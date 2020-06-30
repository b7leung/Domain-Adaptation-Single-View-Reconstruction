import torch
import sys

import utils.network_utils
from core.epoch_manager_interface import EpochManager


# This EpochManager implements the Pix2Vox architecture with
# a CORAL loss for the latent feature maps between the two domains.
# Some the the code is based off this repo:
# https://github.com/DenisDsh/PyTorch-Deep-CORAL/blob/master/coral.py
class CORAL_EpochManager(EpochManager):

    def __init__(self, cfg, encoder, decoder, merger, refiner,
                 checkpoint):

        self.encoder = encoder
        self.decoder = decoder
        self.merger = merger
        self.refiner = refiner
        self.cfg = cfg

        # Set up loss functions
        self.bce_loss = torch.nn.BCELoss()  # for voxels

    def compute_coral_loss(self, source, target):

        d = source.size(1)  # dim vector

        source_c = self.compute_covariance(source)
        target_c = self.compute_covariance(target)

        loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

        loss = loss / (4 * d * d)
        return loss

    def compute_covariance(self, input_data):
        """
        Compute Covariance matrix of the input data
        """
        n = input_data.size(0)  # batch_size

        # Check if using gpu or cpu
        if input_data.is_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        id_row = torch.ones(n).reshape(1, n).to(device=device)
        sum_column = torch.mm(id_row, input_data)
        mean_column = torch.div(sum_column, n)
        term_mul_2 = torch.mm(mean_column.t(), mean_column)
        d_t_d = torch.mm(input_data.t(), input_data)
        c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

        return c

    # meant to be called before each epoch to initalize modules and other things
    def init_epoch(self):

        # Batch average meterics
        self.encoder_losses = utils.network_utils.AverageMeter()
        self.refiner_losses = utils.network_utils.AverageMeter()
        self.coral_losses = utils.network_utils.AverageMeter()

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

        # CORAL loss between vectorized source and target features
        # TODO: this coral loss currently assumes a single view. not sure if things change w/ multiple views
        s_batch_size = s_image_features.shape[0]
        t_batch_size = t_image_features.shape[0]
        s_image_features_vec = s_image_features.reshape(s_batch_size, -1)
        t_image_features_vec = t_image_features.reshape(t_batch_size, -1)

        coral_loss = self.compute_coral_loss(s_image_features_vec,
                                             t_image_features_vec) * self.cfg.TRAIN.DA.CORAL_LAMBDA

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

        coral_loss.backward(retain_graph=True)
        if self.cfg.NETWORK.USE_REFINER and epoch_idx >= self.cfg.TRAIN.EPOCH_START_USE_REFINER:
            encoder_loss.backward(retain_graph=True)
            refiner_loss.backward()
        else:
            encoder_loss.backward()

        # processing record for this update step
        self.encoder_losses.update(encoder_loss.item())
        self.refiner_losses.update(refiner_loss.item())
        self.coral_losses.update(coral_loss.item())
        step_record = {"EDLoss": encoder_loss.item(), "RLoss": refiner_loss.item(), "CoralLoss": coral_loss.item()}

        return step_record

    def end_epoch(self):
        # nothing to do here since we don't have any EpochManager-specific modules
        pass
