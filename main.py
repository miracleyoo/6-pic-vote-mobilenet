# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

from utils.utils import *
from data_loader import *
from torchvision.datasets import ImageFolder
from config import Config
from models import MobileNetV2
from tensorboardX import SummaryWriter
import argparse
import torch
import os


def main():
    # Initializing configs
    folder_init(opt)
    net = None

    # Initialize model
    try:
        if opt.MODEL == 'MobileNetV2':
            net = MobileNetV2.MobileNetV2(opt)
    except KeyError('Your model is not found.'):
        exit(0)
    finally:
        print("==> Model initialized successfully.")

    if opt.LOAD_SAVED_MOD:
        net.load()
        net.to_multi()

    # Initialize Data
    train_data = ImageFolder(opt.TRAIN_PATH)
    eval_data = ImageFolder(opt.EVAL_PATH)
    train_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=opt.BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=opt.NUM_WORKERS)
    eval_loader = torch.utils.data.DataLoader(eval_data,
                                              batch_size=opt.BATCH_SIZE,
                                              shuffle=True,
                                              num_workers=opt.NUM_WORKERS)
    print("==> All datasets are generated successfully.")

    # Instantiation of tensorboard and add net graph to it
    print("==> Adding summaries...")
    writer = SummaryWriter(opt.SUMMARY_PATH)
    dummy_input = torch.rand(opt.BATCH_SIZE, opt.NUM_CHANNEL, opt.WIDTH, opt.LENGTH)

    try:
        writer.add_graph(net, dummy_input)
    except KeyError:
        writer.add_graph(net.module, dummy_input)

    net.fit(train_loader, eval_loader)
    net.predict(eval_loader)


if __name__ == '__main__':
    # Options
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-lsm', '--LOAD_SAVED_MOD', type=str2bool,
                        help='If you want to load saved model')
    parser.add_argument('-gi', '--GPU_INDEX', type=str,
                        help='Index of GPUs you want to use')

    args = parser.parse_args()
    print(args)
    opt = Config()
    for k, v in vars(args).items():
        if v is not None and hasattr(opt, k):
            setattr(opt, k, v)
            print(k, v, getattr(opt, k))
    if args.GPU_INDEX:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_INDEX
    main()
