# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import os
import torch

from datetime import datetime as dt


def var_or_cuda(x):
    if torch.cuda.is_available():
        # TODO: check what non_blocking actually does, it seems like in general we should set it to be true
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


# initalize the class mean shapes for the current batch's classes
#TODO: disentangle this
shapenetID2Name = {
    "02691156":"aeroplane",
    "02828884":"bench",
    "02933112":"cabinet",
    "02958343":"car",
    "03001627":"chair",
    "03211117":"display",
    "03636649":"lamp",
    "03691459":"speaker",
    "04090263":"rifle",
    "04256520":"sofa",
    "04379243":"table",
    "04401088":"telephone",
    "04530566":"watercraft",

    "Airplane_Model":"aeroplane",
    "Car_Model":"car",
    "Monitor":"display",
    "Lamp":"lamp",
    "Telephone":"telephone",
    "Boat_Model":"watercraft"
    }


def get_batch_mean_features(class_mean_features, feature_map_shape, batch_classes):
    num_views = feature_map_shape[1]
    batch_class_mean_features = []
    for batch_idx, class_id in enumerate(batch_classes):
        class_name = shapenetID2Name[class_id]
        y = torch.cat([class_mean_features[class_name] for i in range(num_views)], dim=1)
        batch_class_mean_features.append(y)
    batch_class_mean_features = torch.cat(batch_class_mean_features, dim = 0)
    return batch_class_mean_features


def save_checkpoints(cfg, file_path, epoch_idx, encoder, encoder_solver, \
        decoder, decoder_solver, refiner, refiner_solver, merger, merger_solver, \
        classifier, classifier_solver, best_iou, best_epoch):
    print('[INFO] %s Saving checkpoint to %s ...' % (dt.now(), file_path))
    checkpoint = {
        'epoch_idx': epoch_idx,
        'best_iou': best_iou,
        'best_epoch': best_epoch,
        'encoder_state_dict': encoder.state_dict(),
        'encoder_solver_state_dict': encoder_solver.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'decoder_solver_state_dict': decoder_solver.state_dict()
        }

    if cfg.NETWORK.USE_REFINER:
        checkpoint['refiner_state_dict'] = refiner.state_dict()
        checkpoint['refiner_solver_state_dict'] = refiner_solver.state_dict()
    if cfg.NETWORK.USE_MERGER:
        checkpoint['merger_state_dict'] = merger.state_dict()
        checkpoint['merger_solver_state_dict'] = merger_solver.state_dict()
    if cfg.NETWORK.USE_CLASSIFIER:
        checkpoint['classifier_state_dict'] = classifier.state_dict()
        checkpoint['classifier_solver_state_dict'] = classifier_solver.state_dict()

    torch.save(checkpoint, file_path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def models_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
