
import torch

import utils.network_utils

# This is the EpochManager which does not do any domain adaptation.
# It just uses the encoder, decoder, merger, and refiner.
class NoDA_EpochManager():

    def __init__(self, cfg, encoder, decoder, merger, refiner,
                 checkpoint):
        
        self.encoder = encoder
        self.decoder = decoder
        self.merger = merger
        self.refiner = refiner
        self.cfg = cfg

        # Set up loss functions
        self.bce_loss = torch.nn.BCELoss()  # for voxels


    # meant to be called before each epoch to initalize modules and other things
    def init_epoch(self):

        # set up batch average metrics 
        self.encoder_losses = utils.network_utils.AverageMeter()
        self.refiner_losses = utils.network_utils.AverageMeter()


    # returns step_record; a dict meant to be a row in a pandas dataframe which records
    # information for this current minibatch step
    def perform_step(self, source_batch_data, target_batch_data, epoch_idx):

        # since no domain adaptation is used, target_batch_data is ignored
        (taxonomy_id, sample_names, rendering_images,
            ground_truth_volumes, ground_truth_class_labels) = source_batch_data 
        
        # Get data from data loader
        rendering_images = utils.network_utils.var_or_cuda(rendering_images)
        ground_truth_volumes = utils.network_utils.var_or_cuda(ground_truth_volumes)
        ground_truth_class_labels = utils.network_utils.var_or_cuda(ground_truth_class_labels)

        image_features = self.encoder(rendering_images)
        raw_features, generated_volumes = self.decoder(image_features)

        # if the merger is turned off, just compute the mean
        if self.cfg.NETWORK.USE_MERGER and epoch_idx >= self.cfg.TRAIN.EPOCH_START_USE_MERGER:
            generated_volumes = self.merger(raw_features, generated_volumes)
        else:
            generated_volumes = torch.mean(generated_volumes, dim=1)
        encoder_loss = self.bce_loss(generated_volumes, ground_truth_volumes) * 10

        if self.cfg.NETWORK.USE_REFINER and epoch_idx >= self.cfg.TRAIN.EPOCH_START_USE_REFINER:
            generated_volumes = self.refiner(generated_volumes)
            refiner_loss = self.bce_loss(generated_volumes, ground_truth_volumes) * 10
        else:
            refiner_loss = encoder_loss

        # Gradient decent
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.refiner.zero_grad()
        self.merger.zero_grad()

        if self.cfg.NETWORK.USE_REFINER and epoch_idx >= self.cfg.TRAIN.EPOCH_START_USE_REFINER:
            encoder_loss.backward(retain_graph=True)
            refiner_loss.backward()
        else:
            encoder_loss.backward()

        # processing record for this update step
        self.encoder_losses.update(encoder_loss.item())
        self.refiner_losses.update(refiner_loss.item())
        step_record = {"EDLoss": encoder_loss.item(), "RLoss": refiner_loss.item()}

        return step_record


    def end_epoch(self):
        # nothing to do here since we don't have any EpochManager-specific modules
        pass
