#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

# MY NOTES
# config
# need matplotlib 3.0.3, tensorBoardX 1.2.0
# torchvision 1.5? to be compatible with pillow
# train cmd
# python runner.py --gpu 2 --rand
# test cmd
# python runner.py --test --gpu 2 --weights=./saved_params/Pix2Vox-A-ShapeNet.pth --out output --save-num 3

# owild test
#python runner.py --test --gpu 2 --weights=./saved_params/Pix2Vox-A-ShapeNet.pth --out output --save-num -1 --trial_comment TestOWILDSaveAllAuthorCkpt

# training
# python runner.py --gpu 1 --epoch 1

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
from tensorboardX import SummaryWriter

from config import cfg
from core.train import train_net
from core.test import test_net

def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [cuda0]', default=cfg.CONST.DEVICE, type=str)
    parser.add_argument('--rand', dest='randomize', help='Randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument(
        '--batch-size', dest='batch_size', help='name of the net', default=cfg.CONST.BATCH_SIZE, type=int)
    parser.add_argument('--epoch', dest='epoch', help='num. of epoches. Only has effect for training', default=cfg.TRAIN.NUM_EPOCHES, type=int)
    parser.add_argument('--num_views', dest='num_views', help='number of views to use', default=cfg.CONST.N_VIEWS_RENDERING, type=int)
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=None)

    parser.add_argument('--trial_comment', dest='trial_comment', help='Description of Trial', default="")
    parser.add_argument('--save_num', dest='save_num', help='Save n samples per class. If -1, save all.', default=cfg.TEST.SAVE_NUM, type=int)
    parser.add_argument('--classify', dest='classify', help='also perform classification', action='store_true', default=cfg.NETWORK.USE_CLASSIFIER)
    parser.add_argument('--classes', dest = "classes_to_use", nargs='+', help='Classes to use', default = None)

    parser.add_argument(
        '--test_dataset', dest='test_dataset', help='What dataset to test on.', default="ShapeNet", type=str)
    parser.add_argument(
        '--train_dataset', dest='train_dataset', help='What dataset to train on.', default="ShapeNet", type=str)    
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
    cfg.NETWORK.USE_CLASSIFIER = args.classify
    cfg.DATASET.CLASSES_TO_USE = args.classes_to_use
    if args.test_dataset in cfg.DATASETS.TEST_AVAILABLE:
        cfg.DATASET.TEST_DATASET = args.test_dataset
    else:
        print('[FATAL] %s Invalid test dataset.' % (dt.now()))
        sys.exit(2)
    if args.train_dataset in cfg.DATASETS.TRAIN_AVAILABLE:
        cfg.DATASET.TRAIN_DATASET = args.train_dataset
    else:
        print('[FATAL] %s Invalid train dataset.' % (dt.now()))
        sys.exit(2)


    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE


    # creating output folder for this trial run
    mode = "TEST" if args.test else "TRAIN"
    trial_name = time.strftime("%Y_%m_%d--%H_%M_%S") +"_"+ args.trial_comment+"_"+mode
    output_dir = os.path.join(cfg.DIR.OUT_PATH, trial_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        with open(os.path.join(output_dir,"run_cfg_info.txt"), 'wt') as f:
            pprint(args, stream=f)
            f.write("\n\n")
            pprint(cfg,stream=f)


    # Start train/test process
    if not args.test:
        train_net(cfg, output_dir = output_dir)
    else:
        if 'WEIGHTS' in cfg.CONST and os.path.exists(cfg.CONST.WEIGHTS):
            #writer = SummaryWriter(os.path.join("runs",trial_name))
            test_net(cfg, output_dir = output_dir)
        else:
            print('[FATAL] %s Please specify the file path of checkpoint.' % (dt.now()))
            sys.exit(2)


if __name__ == '__main__':
    # Check python version
    if sys.version_info < (3, 0):
        raise Exception("Please follow the installation instruction on 'https://github.com/hzxie/Pix2Vox'")

    # Setup logger
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)

    main()
