#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import logging
import matplotlib
import multiprocessing as mp
import numpy as np
import os
import sys
import time
# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')

from argparse import ArgumentParser
from datetime import datetime as dt
from pprint import pprint

from config import cfg
from core.train import train_net
from core.test import test_net


def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')

    # initalization arguments
    parser.add_argument('--trial_comment', dest='trial_comment', help='Description of Trial', default="")
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use', default=cfg.CONST.DEVICE, type=str)
    parser.add_argument('--rand', dest='randomize', help='Randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument('--weights', dest='weights', help='Initialize network from a weights file path', default=None)
    parser.add_argument('--verbose', dest='verbose', help='Print more details while running', action='store_true', default=cfg.PREFERENCES.VERBOSE)

    # training/testing arguments
    parser.add_argument('--test_dataset', dest='test_dataset', help='What dataset to test on. Valid options are ShapeNet and ShapeNetPlaces.', default="ShapeNet", type=str)
    parser.add_argument('--train_source_dataset', dest='train_source_dataset', help='Source dataset to train on. Valid options are ShapeNet and ShapeNetPlaces.', default="ShapeNet", type=str)
    parser.add_argument('--train_target_dataset', dest='train_target_dataset', help=("Target dataset to train on. If omitted, no domain adaptation is performed. \
                        valid options are OWILD, OOWL, OOWL_SEGMENTED"), default=None, type=str)
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--use_train_set', dest='use_train_set', help='Use training set when testing', action='store_true', default=False)
    parser.add_argument('--batch_size', dest='batch_size', help='Set batch size during training.', default=cfg.CONST.BATCH_SIZE, type=int)
    parser.add_argument('--epoch', dest='epoch', help='num. of epoches. Only has effect for training', default=cfg.TRAIN.NUM_EPOCHES, type=int)
    parser.add_argument('--save_num', dest='save_num', help='During testing only: Save n reconstructions per class. If -1, save all.', default=cfg.TEST.SAVE_NUM, type=int)
    parser.add_argument('--classes', dest="classes_to_use", nargs='+', help='Classes to use; use shapenet convention names, they will be converted as needed.\
                        Shapenet classes are aeroplane, bench, cabinet, car, chair, display, lamp, speaker, rifle, sofa, table, telephone, watercraft. ', default=None)
    parser.add_argument('--num_views', dest='num_views', help='number of views to use', default=cfg.CONST.N_VIEWS_RENDERING, type=int)

    # general architecture arguments
    parser.add_argument('--no_refiner', dest='no_refiner', help='turn off refiner', action='store_true', default=False)
    parser.add_argument('--no_merger', dest='no_merger', help='turn off merger', action='store_true', default=False)
    parser.add_argument('--DA', dest='da', help='Type of DA to use. Options are CORAL, DANN, VoxelCls', default=None)

    # CORAL arguments
    parser.add_argument('--coral_lam', dest='coral_lam', help='lambda for CORAL loss', default=cfg.TRAIN.DA.CORAL_LAMBDA, type=float)

    # DANN arguments
    parser.add_argument('--DANN_lam', dest='dann_lam', help='lambda for DANN', default=cfg.TRAIN.DA.DANN_LAMBDA, type=float)

    # Voxel Classification arguments
    parser.add_argument('--voxel_lam', dest='voxel_lam', help='lambda for VoxelCls loss. -1 is adaptive linear, -2 is adaptive DANN', default=cfg.TRAIN.VOXEL_CLASSIFIER_LAMBDA, type=int)

    args = parser.parse_args()
    return args


def main():

    # Get args from command line
    args = get_args_from_command_line()
    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if not args.randomize:
        np.random.seed(cfg.CONST.RNG_SEED)
    if args.batch_size is not None:
        cfg.CONST.BATCH_SIZE = args.batch_size
    if args.epoch is not None:
        cfg.TRAIN.NUM_EPOCHES = args.epoch
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
        if not args.test:
            cfg.TRAIN.RESUME_TRAIN = True
    cfg.CONST.N_VIEWS_RENDERING = args.num_views
    cfg.TEST.SAVE_NUM = args.save_num
    cfg.DATASET.CLASSES_TO_USE = args.classes_to_use
    if args.test_dataset in cfg.DATASETS.TEST_AVAILABLE:
        cfg.DATASET.TEST_DATASET = args.test_dataset
    else:
        print('[FATAL] %s Invalid test dataset.' % (dt.now()))
        sys.exit(2)
    if args.train_source_dataset in cfg.DATASETS.SOURCE_TRAIN_AVAILABLE:
        cfg.DATASET.TRAIN_SOURCE_DATASET = args.train_source_dataset
    else:
        print('[FATAL] %s Invalid train source dataset.' % (dt.now()))
        sys.exit(2)
    if args.train_target_dataset is None or args.train_target_dataset in cfg.DATASETS.TARGET_TRAIN_AVAILABLE:
        cfg.DATASET.TRAIN_TARGET_DATASET = args.train_target_dataset
    else:
        print('[FATAL] %s Invalid train target dataset, %s.' % (dt.now(), args.train_target_dataset))
        sys.exit(2)
    if args.train_source_dataset == "ShapeNetPlaces" or args.test_dataset == "ShapeNetPlaces":
        cfg.DATASET.USE_PLACES = True
    cfg.TEST.USE_TRAIN_SET = args.use_train_set
    cfg.PREFERENCES.VERBOSE = args.verbose
    if args.no_refiner:
        cfg.NETWORK.USE_REFINER = False
    if args.no_merger:
        cfg.NETWORK.USE_MERGER = False
    cfg.TRAIN.USE_DA = args.da
    cfg.TRAIN.DA.CORAL_LAMBDA = args.coral_lam
    cfg.TRAIN.DA.DANN_LAMBDA = args.dann_lam
    cfg.TRAIN.VOXEL_CLASSIFIER_LAMBDA = args.voxel_lam
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE  # set GPU


    # creating output folder for this trial run
    mode = "TEST" if args.test else "TRAIN"
    trial_name = time.strftime("%Y_%m_%d--%H_%M_%S")  + "_" + args.trial_comment + "_" + mode
    output_dir = os.path.join(cfg.DIR.OUT_PATH, trial_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        # creating a file called run_cfg_info.txt inside the trial folder to store arguments
        with open(os.path.join(output_dir, "run_cfg_info.txt"), 'wt') as f:
            pprint(args, stream=f)
            f.write("\n\n")
            pprint(cfg, stream=f)


    # Start train/test
    if args.test:
        if 'WEIGHTS' in cfg.CONST and os.path.exists(cfg.CONST.WEIGHTS):
            test_net(cfg, output_dir=output_dir)
        else:
            print('[FATAL] %s Please specify the file path of checkpoint.' % (dt.now()))
            sys.exit(2)
    else:
        train_net(cfg, output_dir=output_dir)


if __name__ == '__main__':
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception("Please follow the installation instruction on 'https://github.com/hzxie/Pix2Vox'")

    # Setup logger
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.WARNING)

    main()
