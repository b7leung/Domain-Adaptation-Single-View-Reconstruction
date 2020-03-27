# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import json
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data
import cv2
from tqdm import tqdm
import pprint

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
import utils.binvox_rw

from datetime import datetime as dt

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger


def test_net(cfg, epoch_idx=-1, output_dir=None, test_data_loader=None,
             encoder=None, decoder=None, refiner=None, merger=None):

    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # setting up dataset-specific properties
    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
    if cfg.DATASET.TEST_DATASET == "ShapeNet":
        has_gt_volume = True
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])
    elif cfg.DATASET.TEST_DATASET == "ShapeNetPlaces":
        has_gt_volume = True
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])
    elif cfg.DATASET.TEST_DATASET in ["OWILD", "OOWL"]:
        has_gt_volume = False
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            #utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])
        cfg.DATASET.CLASSES_TO_USE = [utils.network_utils.shapenet2oowl_name[shapenet_class] for shapenet_class in cfg.DATASET.CLASSES_TO_USE]
    elif cfg.DATASET.TEST_DATASET == "OOWL_SEGMENTED":
        has_gt_volume = False
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])
        cfg.DATASET.CLASSES_TO_USE = [utils.network_utils.shapenet2oowl_name[shapenet_class] for shapenet_class in cfg.DATASET.CLASSES_TO_USE]

    # Set up data loader
    if test_data_loader is None:
        if cfg.TEST.USE_TRAIN_SET:
            partition = utils.data_loaders.DatasetType.TRAIN
        else:
            partition = utils.data_loaders.DatasetType.TEST

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=dataset_loader.get_dataset(partition,
                                               cfg.CONST.N_VIEWS_RENDERING, test_transforms, classes_filter=cfg.DATASET.CLASSES_TO_USE),
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            shuffle=False)

    # Set up networks from saved parameters if not explicitly passed in
    if decoder is None or encoder is None:
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        refiner = Refiner(cfg)
        merger = Merger(cfg)

        if torch.cuda.is_available():
            encoder = torch.nn.DataParallel(encoder).cuda()
            decoder = torch.nn.DataParallel(decoder).cuda()
            refiner = torch.nn.DataParallel(refiner).cuda()
            merger = torch.nn.DataParallel(merger).cuda()

        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if cfg.NETWORK.USE_REFINER:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()  # for voxels

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    encoder_losses = utils.network_utils.AverageMeter()
    refiner_losses = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    # keeps track of how many examples of each class has been saved
    save_count = {}

    # save latent vectors and classification labels for t-SNE embeddings
    latent_vectors = []
    classification_labels = []

    for sample_idx, sample_data in enumerate(tqdm(test_data_loader, desc="Testing", leave=False)):

        (taxonomy_id_arr, sample_name, rendering_images,
            ground_truth_volume, ground_truth_class_labels) = sample_data

        # performing initial processing of minibatch
        taxonomy_id = taxonomy_id_arr[0] if isinstance(taxonomy_id_arr[0], str) else taxonomy_id_arr[0].item()
        sample_name = sample_name[0]
        class_name = taxonomies[taxonomy_id]['taxonomy_name']
        if class_name not in save_count:
            save_count[class_name] = 0
        num_images = rendering_images.shape[1]
        for i in range(num_images):
            classification_labels.append(ground_truth_class_labels.item())

        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volume)
            
            # Test the encoder, decoder, refiner, merger
            image_features = encoder(rendering_images)
            raw_features, generated_volume = decoder(image_features)

            # save latent vectors
            for instance_features in image_features:
                for view_features in instance_features:
                    latent_vectors.append(view_features.reshape(-1))

            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                generated_volume = merger(raw_features, generated_volume)
            else:
                generated_volume = torch.mean(generated_volume, dim=1)

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volume = refiner(generated_volume)

            # only compute shape based losses/metrics if we have access to ground truth volume
            if has_gt_volume:
                encoder_loss = bce_loss(generated_volume, ground_truth_volume) * 10
                if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                    refiner_loss = bce_loss(generated_volume, ground_truth_volume) * 10
                else:
                    refiner_loss = encoder_loss

                # Append loss and accuracy to average metrics
                encoder_losses.update(encoder_loss.item())
                refiner_losses.update(refiner_loss.item())

                # IoU per sample
                sample_iou = []
                for th in cfg.TEST.VOXEL_THRESH:
                    _volume = torch.ge(generated_volume, th).float()
                    intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                    union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                    sample_iou.append((intersection / union).item())

                # IoU per taxonomy
                if taxonomy_id not in test_iou:
                    test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
                test_iou[taxonomy_id]['n_samples'] += 1
                test_iou[taxonomy_id]['iou'].append(sample_iou)

                curr_volumes = [(generated_volume, "reconstructed"), (ground_truth_volume, "ground_truth")]

            else:
                curr_volumes = [(generated_volume, "reconstructed")]
                    
            # save gt reconstruction, estimated reconstruction, and input images 
            if output_dir and (save_count[class_name] < cfg.TEST.SAVE_NUM or cfg.TEST.SAVE_NUM == -1):
                img_dir = os.path.join(output_dir, class_name, sample_name)
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)

                for volume, source in curr_volumes:
                    with open(os.path.join(img_dir, source + ".binvox"), 'wb') as f:
                        # saving voxel
                        th = 0.5
                        vox = utils.binvox_rw.Voxels(volume.ge(th).cpu().squeeze().numpy(), (32,) * 3, (0,) * 3, 1, 'xyz')
                        vox.write(f)
                        # saving voxel renders
                        utils.binvox_visualization.get_volume_views(volume.cpu().numpy(), os.path.join(img_dir, 'test'),
                                                                    epoch_idx, save_path=os.path.join(img_dir, source + ".png"))

                # saving input images
                for i, img in enumerate(rendering_images.squeeze(dim=0)):
                    curr_img = img.cpu().numpy().transpose((1, 2, 0)) * 255
                    cv2.imwrite(os.path.join(img_dir, "input_{}.jpg".format(i)), curr_img)
                
                save_count[class_name] += 1

    latent_vectors = [lv.cpu().numpy() for lv in latent_vectors]
    latent_vectors = np.array(latent_vectors)

    max_iou = 0
    # Shape based results are only possible if we have GT volume
    if has_gt_volume:
        mean_iou = []
        for taxonomy_id in test_iou:
            test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
            mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
        mean_iou = np.sum(mean_iou, axis=0) / n_samples
        print('============================ TEST RESULTS ============================')
        print('Taxonomy', end='\t')
        print('#Sample', end='\t')
        print('Baseline', end='\t')
        for th in cfg.TEST.VOXEL_THRESH:
            print('t=%.2f' % th, end='\t')
        print()
        # Print body
        for taxonomy_id in test_iou:
            print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
            print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
            if 'baseline' in taxonomies[taxonomy_id]:
                print('%.4f' % taxonomies[taxonomy_id]['baseline']['%d-view' % cfg.CONST.N_VIEWS_RENDERING], end='\t\t')
            else:
                print('N/a', end='\t\t')

            for ti in test_iou[taxonomy_id]['iou']:
                print('%.4f' % ti, end='\t')
            print()
        # Print mean IoU for each threshold
        print('Overall ', end='\t\t\t\t')
        for mi in mean_iou:
            print('%.4f' % mi, end='\t')
        print('\n')

        max_iou = np.max(mean_iou)

    # returning assets/useful data to be used for later visualization (like t-SNE)
    return [max_iou, classification_labels, latent_vectors]