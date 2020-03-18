# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm
import torch
import torch.backends.cudnn
import torch.utils.data
import pandas as pd
import copy

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.test import test_net
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
from models.classifier import Classifier


def train_net(cfg, output_dir):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    # might be bad if you have layers which are only activated when certain conditions are met
    # https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
    # TODO: check this, esp. when doing DA
    torch.backends.cudnn.benchmark = True

    # setting up dataset specific options
    if cfg.DATASET.TEST_DATASET == 'ShapeNet':
        num_classes = cfg.DATASETS.SHAPENET.NUM_CLASSES

    # Set up data augmentation
    # TODO: check if these make sense, esp for non-shapenet datasets
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    if cfg.DATASET.USE_PLACES:
        train_transforms = utils.data_transforms.Compose([
            utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
            #utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
            utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.RandomFlip(),
            utils.data_transforms.RandomPermuteRGB(),
            utils.data_transforms.ToTensor(),
        ])
        val_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            #utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])
    else:
        train_transforms = utils.data_transforms.Compose([
            utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION),
            utils.data_transforms.RandomNoise(cfg.TRAIN.NOISE_STD),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.RandomFlip(),
            utils.data_transforms.RandomPermuteRGB(),
            utils.data_transforms.ToTensor(),
        ])
        val_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    val_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset_loader.get_dataset(utils.data_loaders.DatasetType.TRAIN,
                                                 cfg.CONST.N_VIEWS_RENDERING, train_transforms, classes_filter = cfg.DATASET.CLASSES_TO_USE),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER,
        pin_memory=True,
        shuffle=True,
        drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset_loader.get_dataset(utils.data_loaders.DatasetType.VAL,
                                               cfg.CONST.N_VIEWS_RENDERING, val_transforms, classes_filter = cfg.DATASET.CLASSES_TO_USE),
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        shuffle=False)

    # Set up networks
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    merger = Merger(cfg)
    classifier = Classifier(cfg, num_classes)
    print('[DEBUG] %s Parameters in Encoder: %d.' % (dt.now(), utils.network_utils.count_parameters(encoder)))
    print('[DEBUG] %s Parameters in Decoder: %d.' % (dt.now(), utils.network_utils.count_parameters(decoder)))
    print('[DEBUG] %s Parameters in Refiner: %d.' % (dt.now(), utils.network_utils.count_parameters(refiner)))
    print('[DEBUG] %s Parameters in Merger: %d.' % (dt.now(), utils.network_utils.count_parameters(merger)))
    print('[DEBUG] %s Parameters in Classifier: %d.' % (dt.now(), utils.network_utils.count_parameters(classifier)))

    # Initialize weights of networks
    encoder.apply(utils.network_utils.init_weights)
    decoder.apply(utils.network_utils.init_weights)
    refiner.apply(utils.network_utils.init_weights)
    merger.apply(utils.network_utils.init_weights)
    classifier.apply(utils.network_utils.init_weights)

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
        classifier_solver = torch.optim.Adam(classifier.parameters(), lr=cfg.TRAIN.CLASSIFIER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
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
        classifier_solver = torch.optim.SGD(
            classifier.parameters(), lr=cfg.TRAIN.CLASSIFIER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
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
    classifier_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        classifier_solver, milestones=cfg.TRAIN.CLASSIFIER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)

    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()
        merger = torch.nn.DataParallel(merger).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()

    # Set up loss functions
    bce_loss = torch.nn.BCELoss() # for voxels
    ce_loss = torch.nn.CrossEntropyLoss() # for classification


    # Load pretrained model if exists
    # TODO: gotcha: currently, code does not use/ignores the solver_state_dicts saved
    init_epoch = 0
    best_iou = -1
    best_epoch = -1
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
        if cfg.NETWORK.USE_CLASSIFIER:
            classifier.load_state_dict(checkpoint['classifier_state_dict'])

        print('[INFO] %s Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' \
                 % (dt.now(), init_epoch, best_iou, best_epoch))

    # Summary writer for TensorBoard
    log_dir = os.path.join(output_dir, 'logs')
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(log_dir, 'test'))

    class_mean_features = torch.load(cfg.DIR.MEAN_FEATURES_PATH)

    # dataframe used for training details
    training_record_df = pd.DataFrame()

    # freeze layers if trying residual 
    if cfg.NETWORK.ADD_MEAN_FEATURES:
        for param in merger.parameters():
            param.requires_grad = False
        for param in refiner.parameters():
            param.requires_grad = False
        for param in decoder.parameters():
            param.requires_grad = False

    #prev_decoder = copy.deepcopy(decoder)
    # Training loop
    for epoch_idx in tqdm(range(init_epoch, cfg.TRAIN.NUM_EPOCHES), desc = "Epoch"):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = utils.network_utils.AverageMeter()
        data_time = utils.network_utils.AverageMeter()
        encoder_losses = utils.network_utils.AverageMeter()
        refiner_losses = utils.network_utils.AverageMeter()
        classification_losses = utils.network_utils.AverageMeter()
        l2_losses = utils.network_utils.AverageMeter()

        # switch models to training mode
        encoder.train()
        decoder.train()
        merger.train()
        refiner.train()
        classifier.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, batch_data in enumerate(tqdm(train_data_loader, desc = "Minibatch", leave = False)):

            (taxonomy_id, sample_names, rendering_images,
                ground_truth_volumes, ground_truth_class_labels) = batch_data 
            
            #print("DECODER SAME: {}".format(utils.network_utils.models_equal(prev_decoder, decoder)))
            #prev_decoder = copy.deepcopy(decoder)

            # Measure data time
            data_time.update(time() - batch_end_time)

            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_volumes = utils.network_utils.var_or_cuda(ground_truth_volumes)
            ground_truth_class_labels = utils.network_utils.var_or_cuda(ground_truth_class_labels)

            # Train the encoder, decoder, refiner, merger, and classifier
            image_features = encoder(rendering_images)

            # combine mean shape to encoded image_features as a weighted avg
            batch_mean_features = utils.network_utils.get_batch_mean_features(class_mean_features,
                                                                              image_features.shape, taxonomy_id)
            if cfg.NETWORK.ADD_MEAN_FEATURES:
                image_features = image_features + batch_mean_features
            else:
                image_features = (image_features*(1-cfg.NETWORK.MEAN_FEATURES_WEIGHT) 
                                + batch_mean_features*cfg.NETWORK.MEAN_FEATURES_WEIGHT)

            class_predictions = classifier(image_features)
            raw_features, generated_volumes = decoder(image_features)

            # combining batch and views for ce loss
            class_predictions = class_predictions.reshape(-1,num_classes)
            # each view has the same class
            ground_truth_class_labels = ground_truth_class_labels.repeat(cfg.CONST.N_VIEWS_RENDERING,1).T
            ground_truth_class_labels = ground_truth_class_labels.reshape(-1)
            classification_loss = ce_loss(class_predictions, ground_truth_class_labels) 

            # if the merger is turned off, just compute the mean
            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                generated_volumes = merger(raw_features, generated_volumes)
            else:
                generated_volumes = torch.mean(generated_volumes, dim=1)

            encoder_loss = bce_loss(generated_volumes, ground_truth_volumes) * 10



            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volumes = refiner(generated_volumes)
                refiner_loss = bce_loss(generated_volumes, ground_truth_volumes) * 10
            else:
                refiner_loss = encoder_loss

            # adding l2 norm on residual
            if cfg.NETWORK.ADD_MEAN_FEATURES:
                l2_loss = cfg.NETWORK.RESIDUAL_LAMBDA*torch.norm(image_features)
                refiner_loss = refiner_loss + l2_loss

            # Gradient decent
            encoder.zero_grad()
            decoder.zero_grad()
            refiner.zero_grad()
            merger.zero_grad()
            classifier.zero_grad()

            #TODO: double check the ordering of this
            if cfg.NETWORK.USE_CLASSIFIER and epoch_idx >=cfg.TRAIN.EPOCH_START_USE_CLASSIFIER:
                classification_loss.backward(retain_graph=True)

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                encoder_loss.backward(retain_graph=True)
                refiner_loss.backward()
            else:
                encoder_loss.backward()

            encoder_solver.step()
            decoder_solver.step()
            refiner_solver.step()
            merger_solver.step()
            classifier_solver.step()

            # Append loss to average metrics
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())
            classification_losses.update(classification_loss.item())
            l2_losses.update(l2_loss.item())
            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('EncoderDecoder/BatchLoss', encoder_loss.item(), n_itr)
            train_writer.add_scalar('Refiner/BatchLoss', refiner_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            if cfg.PREFERENCES.VERBOSE:
                print('[INFO] %s [Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) EDLoss=%.4f RLoss=%.4f ClsLoss=%.4f L2Loss= %.4f' % \
                    (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, \
                    batch_time.val, data_time.val, encoder_loss.item(), refiner_loss.item(), classification_loss.item(), l2_loss.item()))

            training_record_df = training_record_df.append({"Epoch": epoch_idx, "Minibatch":batch_idx, "IoU":-1, "L2_Loss":l2_loss.item()}, ignore_index = True)
        # Append epoch loss to TensorBoard
        train_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx + 1)
        train_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx + 1)

        # Adjust learning rate
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()
        refiner_lr_scheduler.step()
        merger_lr_scheduler.step()
        classifier_lr_scheduler.step()

        # Tick / tock
        epoch_end_time = time()
        print('[INFO] %s Epoch [%d/%d] EpochTime = %.3f (s) EDLoss = %.4f RLoss = %.4f ClsLoss=%.4f L2Loss= %.4f' %
            (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, \
                encoder_losses.avg, refiner_losses.avg, classification_losses.avg, l2_losses.avg))

        # Update Rendering Views
        if cfg.TRAIN.UPDATE_N_VIEWS_RENDERING:
            n_views_rendering = random.randint(1, cfg.CONST.N_VIEWS_RENDERING)
            train_data_loader.dataset.set_n_views_rendering(n_views_rendering)
            print('[INFO] %s Epoch [%d/%d] Update #RenderingViews to %d' % \
                (dt.now(), epoch_idx + 2, cfg.TRAIN.NUM_EPOCHES, n_views_rendering))

        # Validate the training models
        iou = test_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, encoder, decoder, refiner, merger, classifier)[0]

        # saving training record
        training_record_df = training_record_df.append({"Epoch": epoch_idx, "Minibatch":-1, "IoU":iou, "L2_Loss":l2_losses.avg}, ignore_index = True)

        training_record_df.to_pickle(os.path.join(output_dir, "training_record.pkl"))
        print(pd.read_pickle(os.path.join(output_dir, "training_record.pkl")))

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            utils.network_utils.save_checkpoints(cfg, \
                    os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth' % (epoch_idx + 1)), \
                    epoch_idx + 1, encoder, encoder_solver, decoder, decoder_solver, \
                    refiner, refiner_solver, merger, merger_solver, classifier, classifier_solver, best_iou, best_epoch)
        if iou > best_iou:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            best_iou = iou
            best_epoch = epoch_idx + 1
            utils.network_utils.save_checkpoints(cfg, \
                    os.path.join(ckpt_dir, 'best-ckpt.pth'), \
                    epoch_idx + 1, encoder, encoder_solver, decoder, decoder_solver, \
                    refiner, refiner_solver, merger, merger_solver, classifier, classifier_solver, best_iou, best_epoch)

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()


    # returning and saving assets/useful data to be used for later visualization
    return
