
import sys
from datetime import datetime as dt

import torch
from torch import nn
import numpy as np

import utils.network_utils
from utils.gradient_reversal_module import GradientReversal

# This EpochManager implements the Pix2Vox architecture with
# DANN-style adversarial domain adaptation for the latent feature map
# Some the the code is based off this repo:
# https://github.com/jvanvugt/pytorch-domain-adaptation
# Additionally, a classifier is attached to the reconstructed voxel
class VoxelClassify_EpochManager():

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

        # seting up EpochManager specific models, solvers, and schedulers
        self.voxel_classifier = nn.Sequential(
            nn.Linear(32 * 32 * 32, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes)
        )
        
        self.domain_dis = nn.Sequential(
            GradientReversal(),
            nn.Linear(8 * 8 * 256, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

        self.voxel_classifier.apply(utils.network_utils.init_weights)
        self.domain_dis.apply(utils.network_utils.init_weights)

        # TODO: loading specific pretrained model if it exists 
        # TODO: add option for sgd.
        if cfg.TRAIN.POLICY == 'adam':
            self.voxel_classifier_solver = torch.optim.Adam(self.voxel_classifier.parameters(), lr=cfg.TRAIN.VOXEL_CLASSIFIER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
            self.domain_dis_solver = torch.optim.Adam(self.domain_dis.parameters(), lr=cfg.TRAIN.DA.DANN_LEARNING_RATE, betas=cfg.TRAIN.BETAS)

        self.voxel_classifier_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.voxel_classifier_solver, milestones=cfg.TRAIN.VOXEL_CLASSIFIER_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)
        self.domain_dis_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.domain_dis_solver, milestones=cfg.TRAIN.DA.DANN_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)

        if torch.cuda.is_available():
            self.voxel_classifier = torch.nn.DataParallel(self.voxel_classifier).cuda()
            self.domain_dis = torch.nn.DataParallel(self.domain_dis).cuda()

        # Set up loss functions
        self.bce_loss = torch.nn.BCELoss()  # for voxels
        self.ce_loss = torch.nn.CrossEntropyLoss()  # for voxel classification
        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss()  # for domain classification


    # linearly adaptive lambda for voxel classification loss, to making training to smoothly
    def get_lam_linear(self, curr_epoch):

        lam = curr_epoch / self.total_epochs

        return lam


    # exponential adaptive lambda, as suggested in the DANN paper
    def get_lam_DANN(self, curr_epoch):
        gamma = 10
        p = curr_epoch / self.total_epochs
        lam = (2 / (1 + np.exp(-gamma * p))) - 1
        return lam


    # meant to be called before each epoch to initalize modules and other things
    def init_epoch(self):

        self.domain_dis.train()
        self.voxel_classifier.train()
        # Batch average meterics
        self.encoder_losses = utils.network_utils.AverageMeter()
        self.refiner_losses = utils.network_utils.AverageMeter()
        self.voxel_classify_losses = utils.network_utils.AverageMeter()
        self.domain_dis_losses = utils.network_utils.AverageMeter()


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

        # DANN loss
        s_batch_size = s_image_features.shape[0]
        t_batch_size = t_image_features.shape[0]
        s_image_features_vec = s_image_features.reshape(s_batch_size, -1)
        t_image_features_vec = t_image_features.reshape(t_batch_size, -1)
        image_features_vec = torch.cat((s_image_features_vec, t_image_features_vec), 0)
        domain_predictions = self.domain_dis(image_features_vec)
        # source domain has label 0, target domain has label 1
        domain_labels = torch.cat([torch.zeros((s_batch_size, 1)), torch.ones((t_batch_size, 1))], axis=0).cuda()
        domain_dis_loss_unweighted = self.bce_logits_loss(domain_predictions, domain_labels)
        domain_dis_loss = domain_dis_loss_unweighted * self.cfg.TRAIN.DA.DANN_LAMBDA

        # generating voxel and computing associated voxel IoU losses
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

        #  voxel classification loss
        s_t_generated_volumes = torch.cat((s_generated_volumes, t_generated_volumes), axis=0)  # batch_size x 32 x 32 x 32
        batch_size = s_t_generated_volumes.shape[0]
        s_t_generated_volumes_vec = s_t_generated_volumes.reshape(batch_size, -1)
        voxel_pred_classes = self.voxel_classifier(s_t_generated_volumes_vec)  # batch_size x num_classes
        voxel_gt_classes = torch.cat((s_ground_truth_class_labels, t_ground_truth_class_labels))
        if self.cfg.TRAIN.VOXEL_CLASSIFIER_LAMBDA == -1:
            voxel_cls_lam = self.get_lam_linear(epoch_idx)
        elif self.cfg.TRAIN.VOXEL_CLASSIFIER_LAMBDA == -2:
            voxel_cls_lam = self.get_lam_DANN(epoch_idx)
        else:
            print('[FATAL] %s Invalid voxel cls lambda' % (dt.now()))
            sys.exit(2)
        voxel_classify_loss_unweighted = self.ce_loss(voxel_pred_classes, voxel_gt_classes)
        voxel_classify_loss = voxel_classify_loss_unweighted * voxel_cls_lam

        # Gradient decent
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.refiner.zero_grad()
        self.merger.zero_grad()
        self.voxel_classifier.zero_grad()
        self.domain_dis.zero_grad()
        
        domain_dis_loss.backward(retain_graph=True)
        encoder_loss.backward(retain_graph=True)
        if self.cfg.NETWORK.USE_REFINER and epoch_idx >= self.cfg.TRAIN.EPOCH_START_USE_REFINER:
            refiner_loss.backward(retain_graph=True)
        voxel_classify_loss.backward()

        # stepping for models introduced by EpochManager
        self.voxel_classifier_solver.step()
        self.domain_dis_solver.step()

        # processing record for this update step
        self.encoder_losses.update(encoder_loss.item())
        self.refiner_losses.update(refiner_loss.item())
        self.voxel_classify_losses.update(voxel_classify_loss.item())
        self.domain_dis_losses.update(domain_dis_loss.item())
        step_record = {"EDLoss": encoder_loss.item(), "RLoss": refiner_loss.item(),
                       "VoxelClsLoss": voxel_classify_loss_unweighted.item(), "VoxelClsLam": voxel_cls_lam,
                       "DomainDisLoss": domain_dis_loss_unweighted.item()}

        return step_record


    def end_epoch(self):
        # stepping for EpochManager-specific scheduler
        self.domain_dis_lr_scheduler.step()
        self.voxel_classifier_lr_scheduler.step()
