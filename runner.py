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
#python runner.py --test --gpu 2 --weights=./saved_params/Pix2Vox-A-ShapeNet.pth --out output --save-num 3

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
    parser.add_argument('--epoch', dest='epoch', help='number of epoches', default=cfg.TRAIN.NUM_EPOCHES, type=int)
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=None)
    parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)

    parser.add_argument('--trial_comment', dest='trial_comment', help='Description of Trial', default="")
    parser.add_argument('--save-num', dest='save_num', help='Save n samples per per class. If n=-1, then save all of them', default=0, type=int)
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
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
        if not args.test:
            cfg.TRAIN.RESUME_TRAIN = True

    # Print config
    print('Use config:')
    pprint(cfg)

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    # Start train/test process
    trial_name = time.strftime("%Y_%m_%d--%H_%M_%S") +"_"+ args.trial_comment
    if not args.test:
        train_net(cfg)
    else:
        if 'WEIGHTS' in cfg.CONST and os.path.exists(cfg.CONST.WEIGHTS):
            writer = SummaryWriter(os.path.join("runs",trial_name))
            test_net(cfg, test_writer = writer, output_dir = os.path.join(args.out_path, trial_name), save_num = args.save_num)
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
