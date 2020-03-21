
import sys

import torch
from torch import nn
import numpy as np

import utils.network_utils


class VoxelClassify_EpochManager():

    # train target data loader should be None; it's ignored
    def __init__(self, cfg, encoder, decoder, merger, refiner,
                 checkpoint, total_epochs):
        
        self.encoder = encoder
        self.decoder = decoder
        self.merger = merger
        self.refiner = refiner
        self.cfg = cfg
        self.total_epochs = total_epochs

        # TODO: find a way to fix this magic number
        num_classes = 13
        self.voxel_classifier = nn.Sequential(
            nn.Linear(32 * 32 * 32, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes)
        )

        self.voxel_classifier.apply(utils.network_utils.init_weights)

        # TODO: loading specific pretrained model if it exists 
        # TODO: add option for sgd.
        if cfg.TRAIN.POLICY == 'adam':
            self.voxel_classifier_solver = torch.optim.Adam(self.voxel_classifier.parameters(), lr=cfg.TRAIN.VOXEL_CLASSIFIER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        self.voxel_classifier_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.voxel_classifier_solver, milestones=cfg.TRAIN.VOXEL_CLASSIFIER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)

        if torch.cuda.is_available():
            self.voxel_classifier = torch.nn.DataParallel(self.voxel_classifier).cuda()

        # Set up loss functions
        self.bce_loss = torch.nn.BCELoss()  # for voxels
        self.ce_loss = torch.nn.CrossEntropyLoss()  # for voxel classification


    def get_lam_linear(self, curr_epoch):

        lam = curr_epoch / self.total_epochs

        return lam

    # the lambda value as suggested in the DANN paper
    def get_lam_DANN(self, curr_epoch):
        gamma = 10
        p = curr_epoch / self.total_epochs
        lam = (2 / (1 + np.exp(-gamma * p))) - 1
        return lam

    def init_epoch(self):
        # meant to be called before each epoch

        self.voxel_classifier.train()
        # Batch average meterics
        self.encoder_losses = utils.network_utils.AverageMeter()
        self.refiner_losses = utils.network_utils.AverageMeter()
        self.voxel_classify_losses = utils.network_utils.AverageMeter()


    # returns a record for the step; a dict meant to be a row in a pandas df
    def perform_step(self, source_batch_data, target_batch_data, epoch_idx):

        (s_taxonomy_id, s_sample_names, s_rendering_images,
            s_ground_truth_volumes, s_ground_truth_class_labels) = source_batch_data

        (t_taxonomy_id, t_sample_names, t_rendering_images,
            _, t_ground_truth_class_labels) = target_batch_data 
        
        s_rendering_images = utils.network_utils.var_or_cuda(s_rendering_images)
        s_ground_truth_volumes = utils.network_utils.var_or_cuda(s_ground_truth_volumes)
        s_ground_truth_class_labels = utils.network_utils.var_or_cuda(s_ground_truth_class_labels)
        t_rendering_images = utils.network_utils.var_or_cuda(t_rendering_images)
        t_ground_truth_class_labels = utils.network_utils.var_or_cuda(t_ground_truth_class_labels)


        s_image_features = self.encoder(s_rendering_images)
        t_image_features = self.encoder(t_rendering_images)

        # TODO: incorporate DANN here

        # generating voxel and computing associated losses
        s_raw_features, s_generated_volumes = self.decoder(s_image_features)
        t_raw_features, t_generated_volumes = self.decoder(t_image_features)
        if self.cfg.NETWORK.USE_MERGER and epoch_idx >= self.cfg.TRAIN.EPOCH_START_USE_MERGER:
            s_generated_volumes = self.merger(s_raw_features, s_generated_volumes)
            t_generated_volumes = self.merger(t_raw_features, t_generated_volumes)
        else:
            # if the merger is turned off, just compute the mean
            s_generated_volumes = torch.mean(s_generated_volumes, dim=1)
            t_generated_volumes = torch.mean(t_generated_volumes, dim=1)
        encoder_loss = self.bce_loss(s_generated_volumes, s_ground_truth_volumes) * 10

        if self.cfg.NETWORK.USE_REFINER and epoch_idx >= self.cfg.TRAIN.EPOCH_START_USE_REFINER:
            s_generated_volumes = self.refiner(s_generated_volumes)
            refiner_loss = self.bce_loss(s_generated_volumes, s_ground_truth_volumes) * 10
            t_generated_volumes = self.refiner(t_generated_volumes)
        else:
            refiner_loss = encoder_loss

        # classify voxel
        # this is batch_size x 32 x 32 x 32
        s_t_generated_volumes = torch.cat((s_generated_volumes, t_generated_volumes), axis=0)
        batch_size = s_t_generated_volumes.shape[0]
        s_t_generated_volumes_vec = s_t_generated_volumes.reshape(batch_size, -1)
        voxel_pred_classes = self.voxel_classifier(s_t_generated_volumes_vec)  # batch_size x num_classes
        voxel_gt_classes = torch.cat((s_ground_truth_class_labels, t_ground_truth_class_labels))
        voxel_classify_loss = self.ce_loss(voxel_pred_classes, voxel_gt_classes)

        # Gradient decent
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.refiner.zero_grad()
        self.merger.zero_grad()
        self.voxel_classifier.zero_grad()
        
        encoder_loss.backward(retain_graph=True)
        if self.cfg.NETWORK.USE_REFINER and epoch_idx >= self.cfg.TRAIN.EPOCH_START_USE_REFINER:
            refiner_loss.backward(retain_graph=True)
        voxel_classify_loss.backward()

        # stepping for models introduced by manager
        self.voxel_classifier_solver.step()

        # processing record for this update step
        self.encoder_losses.update(encoder_loss.item())
        self.refiner_losses.update(refiner_loss.item())
        self.voxel_classify_losses.update(voxel_classify_loss.item())
        step_record = {"EDLoss": encoder_loss.item(), "RLoss": refiner_loss.item(), "VoxelClsLoss": voxel_classify_loss.item()}

        return step_record


    def end_epoch(self):
        self.voxel_classifier_lr_scheduler.step()
