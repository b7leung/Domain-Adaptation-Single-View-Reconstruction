import sys

import torch
from torch import nn

import utils.network_utils
from utils.gradient_reversal_module import GradientReversal
from core.epoch_manager_interface import EpochManager


# This EpochManager implements the Pix2Vox architecture with
# DANN-style adversarial domain adaptation for the latent feature map
# Some the the code is based off this repo:
# https://github.com/jvanvugt/pytorch-domain-adaptation
class DANN_EpochManager(EpochManager):
    def __init__(self, cfg, encoder, decoder, merger, refiner,
                 checkpoint):

        self.encoder = encoder
        self.decoder = decoder
        self.merger = merger
        self.refiner = refiner
        self.cfg = cfg

        # seting up EpochManager specific models, solvers, and schedulers
        # in this case, it's a domain binary classifier
        self.domain_dis = nn.Sequential(
            GradientReversal(),
            nn.Linear(8 * 8 * 256, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

        self.domain_dis.apply(utils.network_utils.init_weights)

        # TODO: add option of loading specific pretrained model if it exists 
        # TODO: add option for sgd.
        if cfg.TRAIN.POLICY == 'adam':
            self.domain_dis_solver = torch.optim.Adam(self.domain_dis.parameters(), lr=cfg.TRAIN.DA.DANN_LEARNING_RATE,
                                                      betas=cfg.TRAIN.BETAS)
        self.domain_dis_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.domain_dis_solver, milestones=cfg.TRAIN.DA.DANN_LR_MILESTONES, gamma=cfg.TRAIN.GAMMA)

        if torch.cuda.is_available():
            self.domain_dis = torch.nn.DataParallel(self.domain_dis).cuda()

        # Set up loss functions
        self.bce_loss = torch.nn.BCELoss()  # for voxels
        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss()  # for domain classification

    # meant to be called before each epoch to initalize modules and other things
    def init_epoch(self):

        self.domain_dis.train()
        # Batch average meterics
        self.encoder_losses = utils.network_utils.AverageMeter()
        self.refiner_losses = utils.network_utils.AverageMeter()
        self.domain_dis_losses = utils.network_utils.AverageMeter()

    # returns step_record; a dict meant to be a row in a pandas dataframe which records
    # information for this current minibatch step
    def perform_step(self, source_batch_data, target_batch_data, epoch_idx):

        (s_taxonomy_id, s_sample_names, s_rendering_images,
         s_ground_truth_volumes, s_ground_truth_class_labels) = source_batch_data

        (t_taxonomy_id, t_sample_names, t_rendering_images,
         _, t_ground_truth_class_labels) = target_batch_data

        s_rendering_images = utils.network_utils.var_or_cuda(s_rendering_images)
        s_ground_truth_volumes = utils.network_utils.var_or_cuda(s_ground_truth_volumes)
        t_rendering_images = utils.network_utils.var_or_cuda(t_rendering_images)

        s_image_features = self.encoder(s_rendering_images)
        t_image_features = self.encoder(t_rendering_images)

        # domain discrinination loss between vectorized source and target features
        s_batch_size = s_image_features.shape[0]
        t_batch_size = t_image_features.shape[0]
        s_image_features_vec = s_image_features.reshape(s_batch_size, -1)
        t_image_features_vec = t_image_features.reshape(t_batch_size, -1)
        image_features_vec = torch.cat((s_image_features_vec, t_image_features_vec), 0)
        domain_predictions = self.domain_dis(image_features_vec)
        # source domain has label 0, target domain has label 1
        domain_labels = torch.cat([torch.zeros((s_batch_size, 1)), torch.ones((t_batch_size, 1))], axis=0).cuda()

        domain_dis_loss = self.bce_logits_loss(domain_predictions, domain_labels) * self.cfg.TRAIN.DA.DANN_LAMBDA

        s_raw_features, s_generated_volumes = self.decoder(s_image_features)

        # if the merger is turned off, just compute the mean
        if self.cfg.NETWORK.USE_MERGER and epoch_idx >= self.cfg.TRAIN.EPOCH_START_USE_MERGER:
            s_generated_volumes = self.merger(s_raw_features, s_generated_volumes)
        else:
            s_generated_volumes = torch.mean(s_generated_volumes, dim=1)
        encoder_loss = self.bce_loss(s_generated_volumes, s_ground_truth_volumes) * 10

        if self.cfg.NETWORK.USE_REFINER and epoch_idx >= self.cfg.TRAIN.EPOCH_START_USE_REFINER:
            s_generated_volumes = self.refiner(s_generated_volumes)
            refiner_loss = self.bce_loss(s_generated_volumes, s_ground_truth_volumes) * 10
        else:
            refiner_loss = encoder_loss

        # Gradient decent
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.refiner.zero_grad()
        self.merger.zero_grad()
        self.domain_dis.zero_grad()

        domain_dis_loss.backward(retain_graph=True)
        if self.cfg.NETWORK.USE_REFINER and epoch_idx >= self.cfg.TRAIN.EPOCH_START_USE_REFINER:
            encoder_loss.backward(retain_graph=True)
            refiner_loss.backward()
        else:
            encoder_loss.backward()

        # stepping for DANN-specific model
        self.domain_dis_solver.step()

        # processing record for this update step
        self.encoder_losses.update(encoder_loss.item())
        self.refiner_losses.update(refiner_loss.item())
        self.domain_dis_losses.update(domain_dis_loss.item())
        step_record = {"EDLoss": encoder_loss.item(), "RLoss": refiner_loss.item(),
                       "DomainDisLoss": domain_dis_loss.item()}

        return step_record

    def end_epoch(self):
        # stepping for EpochManager-specific scheduler
        self.domain_dis_lr_scheduler.step()
