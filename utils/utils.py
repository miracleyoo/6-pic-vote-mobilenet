# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import argparse
import functools
import json
import os
import shutil
import time

import torch
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


# __all__ = ["gen_dataset", "load_data", "folder_init", "divide_func", "str2bool", "Timer"]

def transforms_fn():
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(opt.TENSOR_SHAPE[1]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(opt.TENSOR_SHAPE[1]),
            transforms.CenterCrop(opt.TENSOR_SHAPE[1]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

# Initialize Data
def load_regular_data(opt, net, loader_type=ImageFolder):
    # data_transforms = {
    #     "train": transforms.Compose([
    #         transforms.RandomResizedCrop(opt.TENSOR_SHAPE[1]),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     "val": transforms.Compose([
    #         transforms.Resize(opt.TENSOR_SHAPE[1]),
    #         transforms.CenterCrop(opt.TENSOR_SHAPE[1]),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    # }
    data_transforms = transforms_fn()

    data_dir = "../cards_250_7/cards_for_"
    if opt.USE_SP:
        dsets = loader_type(data_dir + 'train', opt, data_transforms['train'])
        dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=opt.BATCH_SIZE,
                                                   shuffle=True, num_workers=opt.NUM_WORKERS)
        net.opt.NUM_TRAIN = len(dsets)
        return dset_loaders
    elif loader_type != ImageFolder:
        opt.BATCH_SIZE = 6
        dsets = {x: loader_type(data_dir + x, opt, data_transforms[x])
                 for x in ["train", "val"]}
        if opt.TEST_ALL:
            all_datasets = torch.utils.data.ConcatDataset([dsets[key] for key in dsets.keys()])
            all_loader = torch.utils.data.DataLoader(all_datasets, batch_size=opt.BATCH_SIZE,
                                                     num_workers=opt.NUM_WORKERS)
        else:
            all_datasets = dsets["train"]
            all_loader = torch.utils.data.DataLoader(all_datasets, batch_size=opt.BATCH_SIZE,
                                                     num_workers=opt.NUM_WORKERS)
        all_sizes = len(all_datasets)
        net.opt.NUM_VAL = all_sizes/6
        return all_loader
    else:
        dsets = {x: loader_type(data_dir + x, data_transforms[x])
                 for x in ["train", "val"]}
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=opt.BATCH_SIZE,
                                                       shuffle=True, num_workers=opt.NUM_WORKERS)
                        for x in ["train", "val"]}
        dset_sizes = {x: len(dsets[x]) for x in ["train", "val"]}
        net.opt.NUM_TRAIN = dset_sizes["train"]
        net.opt.NUM_VAL = dset_sizes["val"]
        dset_classes = dsets["train"].classes
        with open(opt.CLASSES_PATH, "w+") as f:
            json.dump(dset_classes, f)
        log("Number of Class:", len(dset_classes), " Top3:", dset_classes[:3])
        return dset_loaders["train"], dset_loaders["val"]


def add_summary(opt, net):
    # Instantiation of tensorboard and add net graph to it
    log("Adding summaries...")
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
    if not os.path.exists("source"):
        os.mkdir("source")
    if not os.path.exists("source/reference"):
        os.mkdir("source/reference")
    if not os.path.exists("./source/summary/"):
        os.mkdir("./source/summary/")
    if not os.path.exists(opt.NET_SAVE_PATH):
        os.mkdir(opt.NET_SAVE_PATH)


def str2bool(b):
    if b.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif b.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print("==> [%s]:\t" % self.name, end="")
        self.time_elapsed = time.time() - self.tstart
        print("Elapsed Time: %s (s)" % self.time_elapsed)


def log(*args, end=None):
    if end is None:
        print(time.strftime("==> [%Y-%m-%d %H:%M:%S]", time.localtime()) + " " + "".join([str(s) for s in args]))
    else:
        print(time.strftime("==> [%Y-%m-%d %H:%M:%S]", time.localtime()) + " " + "".join([str(s) for s in args]),
              end=end)
