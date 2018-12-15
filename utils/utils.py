# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import argparse
import torch
import os
import sys
import shutil
import time
import json
import functools
from tensorboardX import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader

# __all__ = ['gen_dataset', 'load_data', 'folder_init', 'divide_func', 'str2bool', 'Timer']


def violent_resize(img, short_len):
    return img.resize((short_len, short_len))


def resize_by_short(img, short_len=128, crop=False):
    """按照短边进行所需比例缩放"""
    (x, y) = img.size
    if x > y:
        y_s = short_len
        x_s = int(x * y_s / y)
        x_l = int(x_s / 2) - int(short_len / 2)
        x_r = int(x_s / 2) + int(short_len / 2)
        img = img.resize((x_s, y_s))
        if crop:
            box = (x_l, 0, x_r, short_len)
            img = img.crop(box)
    else:
        x_s = short_len
        y_s = int(y * x_s / x)
        y_l = int(y_s / 2) - int(short_len / 2)
        y_r = int(y_s / 2) + int(short_len / 2)
        img = img.resize((x_s, y_s))
        if crop:
            box = (0, y_l, short_len, y_r)
            img = img.crop(box)
    return img


def get_center_img(img, short_len=128):
    img = resize_by_short(img, short_len=short_len * 2)
    (x, y) = img.size
    box = (
        x // 2 - short_len * 3 // 4, y // 2 - short_len * 3 // 4, x // 2 + short_len * 3 // 4,
        y // 2 + short_len * 3 // 4)
    img = img.crop(box).resize((short_len, short_len))
    return img


def divide_4_pieces(img, short_len=128, pick=None):
    (x, y) = img.size
    boxs = []
    boxs.append((0, 0, x // 2, y // 2))
    boxs.append((0, y // 2, x // 2, y))
    boxs.append((x // 2, 0, x, y // 2))
    boxs.append((x // 2, y // 2, x, y))
    if pick is not None:
        return img.crop(boxs[pick]).resize((short_len, short_len))
    else:
        imgs = [img.crop(i).resize((short_len, short_len)) for i in boxs]
        return imgs


def get_6_pics(img, short_len=128):
    imgs = []
    imgs.append(violent_resize(img, short_len=short_len))
    imgs.append(get_center_img(img, short_len=short_len))
    imgs.extend(divide_4_pieces(img, short_len=short_len))
    return imgs


def divide_func(index):
    if index == 0:
        return violent_resize
    elif index == 1:
        return get_center_img
    elif 2 <= index <= 5:
        return functools.partial(divide_4_pieces, pick=index - 2)


def div_6_pic(img_path):
    prefix = './source/temp'
    new_root = os.path.join(prefix, img_path.split('/')[-2])
    shutil.rmtree(prefix)
    os.makedirs(new_root)
    img = Image.open(img_path)
    imgs = get_6_pics(img, short_len=128)
    return imgs


# Initialize Data
def load_regular_data(opt, net, loader_type=ImageFolder):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(opt.TENSOR_SHAPE[1]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'eval': transforms.Compose([
            transforms.Resize(opt.TENSOR_SHAPE[1]),
            transforms.CenterCrop(opt.TENSOR_SHAPE[1]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = "../cards_250_7/cards_for_"
    if opt.USE_SP == True:
        dsets = loader_type(data_dir + 'train', opt, data_transforms['train'])
        dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=opt.BATCH_SIZE,
                                                       shuffle=True, num_workers=opt.NUM_WORKERS)
        net.opt.NUM_TRAIN = len(dsets)
        return dset_loaders
    elif loader_type != ImageFolder: # load_type = SixBatch
        opt.BATCH_SIZE = 6
        dsets = {x: loader_type(data_dir + x, opt, data_transforms[x])
                 for x in ['train', 'eval']}
        if opt.TEST_ALL:
            all_datasets = torch.utils.data.ConcatDataset([dsets[key] for key in dsets.keys()])
            all_loader = torch.utils.data.DataLoader(all_datasets, batch_size=opt.BATCH_SIZE,
                                                        num_workers=opt.NUM_WORKERS)
        else:
            all_datasets = dsets["train"]
            all_loader = torch.utils.data.DataLoader(all_datasets, batch_size=opt.BATCH_SIZE,
                                                     num_workers=opt.NUM_WORKERS)
        all_sizes = len(all_datasets)
        net.opt.NUM_EVAL = all_sizes
        return all_loader
    else:
        dsets = {x: loader_type(data_dir + x, data_transforms[x])
                 for x in ['train', 'eval']}
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=opt.BATCH_SIZE,
                                                       shuffle=True, num_workers=opt.NUM_WORKERS)
                        for x in ['train', 'eval']}
        dset_sizes = {x: len(dsets[x]) for x in ['train', 'eval']}
        net.opt.NUM_TRAIN = dset_sizes['train']
        net.opt.NUM_EVAL = dset_sizes['eval']
        dset_classes = dsets['train'].classes
        with open(opt.CLASSES_PATH, "w+") as f:
            json.dump(dset_classes, f)
        return dset_loaders['train'], dset_loaders['eval']


def add_summary(opt, net):
    # Instantiation of tensorboard and add net graph to it
    print("==> Adding summaries...")
    writer = SummaryWriter(opt.SUMMARY_PATH)
    dummy_input = torch.rand(opt.BATCH_SIZE, *opt.TENSOR_SHAPE).to(net.device)

    try:
        writer.add_graph(net, dummy_input)
    except KeyError:
        writer.add_graph(net.module, dummy_input)


def folder_init(opt):
    """
    Initialize folders required
    """
    if not os.path.exists('source'):
        os.mkdir('source')
    if not os.path.exists('source/reference'):
        os.mkdir('source/reference')
    if not os.path.exists('./source/summary/'):
        os.mkdir('./source/summary/')
    if not os.path.exists(opt.NET_SAVE_PATH):
        os.mkdir(opt.NET_SAVE_PATH)


def str2bool(b):
    if b.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print('==> [%s]:\t' % self.name, end='')
        self.time_elapsed = time.time() - self.tstart
        print('Elapsed Time: %s (s)' % self.time_elapsed)
