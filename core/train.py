# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>


from datetime import datetime as dt
import os
import sys

from tqdm import tqdm
import torch
import torch.backends.cudnn
import torch.utils.data
import pandas as pd

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
from core.test import test_net
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
from core.noDA_epoch_manager import NoDA_EpochManager
from core.CORAL_epoch_manager import CORAL_EpochManager


def train_net(cfg, output_dir):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    # might be bad if you have layers which are only activated when certain conditions are met
    # https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
    # TODO: check this, esp. when doing DA
    torch.backends.cudnn.benchmark = True

    # Set up data augmentation, which is different depending on the dataset
    # TODO: check if these make sense, esp for non-shapenet datasets
    # TODO: Add back in places case and funcionality. rn, DATASET.USE_PLACES must be false
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
 
    # this is always  shapenet
    train_source_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
        utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.RandomFlip(),
        utils.data_transforms.RandomPermuteRGB(),
        utils.data_transforms.ToTensor(),
    ])
    
    if cfg.DATASET.TRAIN_TARGET_DATASET in ["OOWL", "OWILD"]:
        train_target_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])
        target_classes_to_use = [utils.network_utils.shapenet2oowl_name[shapenet_class] for shapenet_class in cfg.DATASET.CLASSES_TO_USE]

    # this is always shapenet
    val_transforms = utils.data_transforms.Compose([
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.ToTensor(),
    ])

    # Set up data loader
    if cfg.DATASET.TRAIN_TARGET_DATASET is None:
        eff_batch_size = cfg.CONST.BATCH_SIZE
        train_target_data_loader = None
    else:
        eff_batch_size = int(cfg.CONST.BATCH_SIZE / 2)
        train_target_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_TARGET_DATASET](cfg)
        train_target_data_loader = torch.utils.data.DataLoader(
            dataset=train_target_dataset_loader.get_dataset(utils.data_loaders.DatasetType.TRAIN,
                                                            cfg.CONST.N_VIEWS_RENDERING, train_target_transforms,
                                                            classes_filter=target_classes_to_use),
                                                            # classes_filter=cfg.DATASET.CLASSES_TO_USE),
            batch_size=eff_batch_size,
            num_workers=cfg.TRAIN.NUM_WORKER,
            pin_memory=True,
            shuffle=True,
            drop_last=True)

    train_source_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_SOURCE_DATASET](cfg)
    train_source_data_loader = torch.utils.data.DataLoader(
        dataset=train_source_dataset_loader.get_dataset(utils.data_loaders.DatasetType.TRAIN,
                                                        cfg.CONST.N_VIEWS_RENDERING, train_source_transforms,
                                                        classes_filter=cfg.DATASET.CLASSES_TO_USE),
        batch_size=eff_batch_size,
        num_workers=cfg.TRAIN.NUM_WORKER,
        pin_memory=True,
        shuffle=True,
        drop_last=True)

    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset_loader.get_dataset(utils.data_loaders.DatasetType.VAL,
                                               cfg.CONST.N_VIEWS_RENDERING, val_transforms,
                                               classes_filter=cfg.DATASET.CLASSES_TO_USE),
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        shuffle=False)

    # Set up networks
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    merger = Merger(cfg)
    print('[DEBUG] %s Parameters in Encoder: %d.' % (dt.now(), utils.network_utils.count_parameters(encoder)))
    print('[DEBUG] %s Parameters in Decoder: %d.' % (dt.now(), utils.network_utils.count_parameters(decoder)))
    print('[DEBUG] %s Parameters in Refiner: %d.' % (dt.now(), utils.network_utils.count_parameters(refiner)))
    print('[DEBUG] %s Parameters in Merger: %d.' % (dt.now(), utils.network_utils.count_parameters(merger)))

    # Initialize weights of networks
    encoder.apply(utils.network_utils.init_weights)
    decoder.apply(utils.network_utils.init_weights)
    refiner.apply(utils.network_utils.init_weights)
    merger.apply(utils.network_utils.init_weights)

    # Set up solver
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(
            filter(lambda p: p.requires_grad, encoder.parameters()),
            lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
            betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.Adam(
            decoder.parameters(), lr=cfg.TRAIN.DECODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        refiner_solver = torch.optim.Adam(
            refiner.parameters(), lr=cfg.TRAIN.REFINER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        merger_solver = torch.optim.Adam(merger.parameters(), lr=cfg.TRAIN.MERGER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(
            filter(lambda p: p.requires_grad, encoder.parameters()),
            lr=cfg.TRAIN.ENCODER_LEARNING_RATE,
            momentum=cfg.TRAIN.MOMENTUM)
        decoder_solver = torch.optim.SGD(
            decoder.parameters(), lr=cfg.TRAIN.DECODER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        refiner_solver = torch.optim.SGD(
            refiner.parameters(), lr=cfg.TRAIN.REFINER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        merger_solver = torch.optim.SGD(
            merger.parameters(), lr=cfg.TRAIN.MERGER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        encoder_solver, milestones=cfg.TRAIN.ENCODER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)
    decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        decoder_solver, milestones=cfg.TRAIN.DECODER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)
    refiner_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        refiner_solver, milestones=cfg.TRAIN.REFINER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)
    merger_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        merger_solver, milestones=cfg.TRAIN.MERGER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)

    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()
        merger = torch.nn.DataParallel(merger).cuda()


    # Load pretrained model if exists
    init_epoch = 0
    best_iou = -1
    best_epoch = -1
    checkpoint = None
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_idx']
        best_iou = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if cfg.NETWORK.USE_REFINER:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])

        print('[INFO] %s Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' 
              % (dt.now(), init_epoch, best_iou, best_epoch))


    # setting up saved items
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    training_record_df = pd.DataFrame()

    # setting up trainer depending on the DA config chosen
    if cfg.TRAIN.USE_DA == "CORAL":
        epoch_manager = CORAL_EpochManager(cfg, encoder, decoder, merger, refiner, checkpoint)
    elif cfg.TRAIN.USE_DA is None:
        epoch_manager = NoDA_EpochManager(cfg, encoder, decoder, merger, refiner, checkpoint)
    else:
        print('[FATAL] %s Invalid DA.' % (dt.now()))
        sys.exit(2)

    # Training loop
    for epoch_idx in tqdm(range(init_epoch, cfg.TRAIN.NUM_EPOCHES), desc="Epoch"):

        # initalize models for the epoch
        encoder.train()
        decoder.train()
        merger.train()
        refiner.train()
        epoch_manager.init_epoch()

        # going through minibatches for epoch
        # each epoch corresponds to the length of the source dataset
        source_iter = iter(train_source_data_loader)
        if train_target_data_loader is not None:
            target_iter = iter(train_target_data_loader)
        n_batches = len(train_source_data_loader)

        for batch_idx in tqdm(range(n_batches), desc="Minibatch", leave=False):

            source_batch_data = next(source_iter)
            if train_target_data_loader is not None:
                # since the target is assumed to be smaller, we enable infinite looping thorugh iter.
                try:
                    target_batch_data = next(target_iter)
                except StopIteration:
                    target_iter = iter(train_target_data_loader)
                    target_batch_data = next(target_iter)
            else:
                target_batch_data = None

            step_record = epoch_manager.perform_step(source_batch_data, target_batch_data, epoch_idx)

            encoder_solver.step()
            decoder_solver.step()
            refiner_solver.step()
            merger_solver.step()

            step_record["Epoch"] = epoch_idx
            step_record["Minibatch"] = batch_idx
            if cfg.PREFERENCES.VERBOSE:
                print(step_record)
            training_record_df = training_record_df.append(step_record, ignore_index=True)

        # Adjust learning rate
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        refiner_lr_scheduler.step()
        merger_lr_scheduler.step()
        epoch_manager.end_epoch()

        #TODO: look at this closer for multiview
        # Update Rendering Views
        #if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING:
        #    n_views_rendering = random.randint(1, cfg.CONST.N_VIEWS_RENDERING)
        #    train_data_loader.dataset.set_n_views_rendering(n_views_rendering)
        #    print('[INFO] %s Epoch [%d/%d] Update #RenderingViews to %d' % 
        #          (dt.now(), epoch_idx + 2, cfg.TRAIN.NUM_EPOCHES, n_views_rendering))

        # Validate the training models
        # TODO: also show iou for other thresholds, now it's only the mean
        iou = test_net(cfg, epoch_idx + 1, output_dir, val_data_loader, encoder, decoder, refiner, merger)[0]

        epoch_record = {"Epoch": epoch_idx, "Minibatch": -1, "IoU": iou}
        print(epoch_record)
        training_record_df = training_record_df.append(epoch_record, ignore_index=True)

        # saving training record
        training_record_df.to_pickle(os.path.join(output_dir, "training_record.pkl"))

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_checkpoints(
                cfg, os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth' % (epoch_idx + 1)),
                epoch_idx + 1, encoder, encoder_solver, decoder, decoder_solver,
                refiner, refiner_solver, merger, merger_solver, best_iou, best_epoch)
        if iou > best_iou:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            best_iou = iou
            best_epoch = epoch_idx + 1
            utils.network_utils.save_checkpoints(
                cfg, os.path.join(ckpt_dir, 'best-ckpt.pth'),
                epoch_idx + 1, encoder, encoder_solver, decoder, decoder_solver,
           