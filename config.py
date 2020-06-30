# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import os
from easydict import EasyDict as edict

__C                                         = edict()
cfg                                         = __C

#
# Dataset Config
#
dataset_folder = '/home/svcl-oowl/dataset/'
__C.DATASETS                                = edict()
__C.DATASETS.SOURCE_TRAIN_AVAILABLE         = ["ShapeNet", "ShapeNetPlaces"]
__C.DATASETS.TARGET_TRAIN_AVAILABLE         = ["OWILD", "OOWL", "OOWL_SEGMENTED"]
__C.DATASETS.TEST_AVAILABLE                 = ["ShapeNet", "ShapeNetPlaces", "OWILD", "OOWL", "OOWL_SEGMENTED"]
__C.DATASETS.REBUILD_CACHE                  = False

__C.DATASETS.SHAPENET                         = edict()
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH      = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.RENDERING_PATH          = os.path.join(dataset_folder, "ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png")
__C.DATASETS.SHAPENET.RENDERING_PATH_OnPlaces = os.path.join(dataset_folder, "ShapeNet/ShapeNetRendering_OnPlaces/%s/%s/rendering/%02d.png")
__C.DATASETS.SHAPENET.VOXEL_PATH              = os.path.join(dataset_folder, 'ShapeNet/ShapeNetVox32/%s/%s/model.binvox')
__C.DATASETS.SHAPENET.NUM_CLASSES             = 13

__C.DATASETS.SHAPENETPLACES                         = edict()
__C.DATASETS.SHAPENETPLACES.TAXONOMY_FILE_PATH      = './datasets/ShapeNet.json'

__C.DATASETS.OWILD                          = edict()
__C.DATASETS.OWILD.TAXONOMY_FILE_PATH       = os.path.join(dataset_folder, '3D_ODDS/test_set_info.json')
__C.DATASETS.OWILD.RENDERING_PATH           = os.path.join(dataset_folder, "3D_ODDS/OWILD/%s/%s/%s_01.png")

__C.DATASETS.OOWL                          = edict()
__C.DATASETS.OOWL.TAXONOMY_FILE_PATH       = os.path.join(dataset_folder, '3D_ODDS/test_set_info.json')
__C.DATASETS.OOWL.RENDERING_PATH           = os.path.join(dataset_folder, "3D_ODDS/OOWL/%s/%s/%s_15.png")

__C.DATASETS.OOWL_SEGMENTED                          = edict()
__C.DATASETS.OOWL_SEGMENTED.TAXONOMY_FILE_PATH       = os.path.join(dataset_folder, '3D_ODDS/test_set_info.json')
__C.DATASETS.OOWL_SEGMENTED.RENDERING_PATH           = os.path.join(dataset_folder, "3D_ODDS/OOWL_Segmented/%s/%s/%s_15.png")

#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
__C.DATASET.STD                             = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_SOURCE_DATASET            = 'ShapeNet'
__C.DATASET.TRAIN_TARGET_DATASET            = None
__C.DATASET.USE_PLACES                      = False
__C.DATASET.TEST_DATASET                    = 'ShapeNet'
__C.DATASET.CLASSES_TO_USE                  = None


#
# Common
#
__C.CONST                                   = edict()
__C.CONST.DEVICE                            = '0'
__C.CONST.RNG_SEED                          = 0
__C.CONST.IMG_W                             = 224       # Image width for input
__C.CONST.IMG_H                             = 224       # Image height for input
__C.CONST.N_VOX                             = 32
__C.CONST.BATCH_SIZE                        = 64
__C.CONST.N_VIEWS_RENDERING                 = 1         # Dummy property for Pascal 3D. For shapenet, views are randomly sampled
__C.CONST.CROP_IMG_W                        = 128       # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H                        = 128       # Dummy property for Pascal 3D

#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = './output'
__C.DIR.DATASET_CACHE_PATH                  = './caches/dataset_file_caches'

#
# Network
#
__C.NETWORK                                 = edict()
__C.NETWORK.LEAKY_VALUE                     = .2
__C.NETWORK.TCONV_USE_BIAS                  = False
__C.NETWORK.USE_REFINER                     = True
__C.NETWORK.USE_MERGER                      = True  # if false, just does an average


#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = False
__C.TRAIN.NUM_WORKER                        = 4             # number of data workers
__C.TRAIN.NUM_EPOCHES                       = 250
__C.TRAIN.BRIGHTNESS                        = .4
__C.TRAIN.CONTRAST                          = .4
__C.TRAIN.SATURATION                        = .4
__C.TRAIN.NOISE_STD                         = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
__C.TRAIN.POLICY                            = 'adam'        # available options: sgd, adam
__C.TRAIN.EPOCH_START_USE_REFINER           = 0
__C.TRAIN.EPOCH_START_USE_MERGER            = 0
__C.TRAIN.ENCODER_LEARNING_RATE             = 1e-3
__C.TRAIN.DECODER_LEARNING_RATE             = 1e-3
__C.TRAIN.REFINER_LEARNING_RATE             = 1e-3
__C.TRAIN.MERGER_LEARNING_RATE              = 1e-4
__C.TRAIN.ENCODER_LR_MILESTONES             = [150]
__C.TRAIN.DECODER_LR_MILESTONES             = [150]
__C.TRAIN.REFINER_LR_MILESTONES             = [150]
__C.TRAIN.MERGER_LR_MILESTONES              = [150]
__C.TRAIN.BETAS                             = (.9, .999)
__C.TRAIN.MOMENTUM                          = .9
__C.TRAIN.GAMMA                             = .5
__C.TRAIN.SAVE_FREQ                         = 10            # weights will be overwritten every save_freq epoch
__C.TRAIN.UPDATE_N_VIEWS_RENDERING          = False

__C.TRAIN.VOXEL_CLASSIFIER_LEARNING_RATE    = 1e-3
__C.TRAIN.VOXEL_CLASSIFIER_LR_MILESTONES    = [150]
__C.TRAIN.VOXEL_CLASSIFIER_LAMBDA    = -1

__C.TRAIN.USE_DA = None
__C.TRAIN.DA = edict()
__C.TRAIN.DA.CORAL_LAMBDA = 0
__C.TRAIN.DA.DANN_LAMBDA = 0
__C.TRAIN.DA.DANN_LEARNING_RATE = 1e-3
__C.TRAIN.DA.DANN_LR_MILESTONES = [150]

#
# Testing options
#
__C.TEST                                    = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                       = [.2, .3, .4, .5]
__C.TEST.SAVE_NUM                           = 0
__C.TEST.USE_TRAIN_SET                     = False  # show test results on the training set


# perferencepes
__C.PREFERENCES = edict()
__C.PREFERENCES.VERBOSE = False
