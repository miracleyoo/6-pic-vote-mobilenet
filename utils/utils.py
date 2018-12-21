# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import argparse
import time
import pickle
from .data_loader import *
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
# __all__ = ["gen_dataset", "load_data", "folder_init", "divide_func", "str2bool", "Timer"]


def transforms_fn(opt):
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
def load_regular_data(opt, net, train_loader_type=ImageFolder, val_loader_type=ImageFolder):
    data_transforms = transforms_fn(opt)

    if train_loader_type == val_loader_type == ImageFolder:
        train_set = ImageFolder(opt.TRAIN_PATH, data_transforms['train'])
        val_set = ImageFolder(opt.VAL_PATH, data_transforms['val'])
        train_loaders = torch.utils.data.DataLoader(train_set, batch_size=opt.BATCH_SIZE,
                                                    shuffle=True, num_workers=opt.NUM_WORKERS)
        val_loaders = torch.utils.data.DataLoader(val_set, batch_size=opt.VAL_BATCH_SIZE,
                                                  shuffle=False, num_workers=opt.NUM_WORKERS)
        net.opt.NUM_TRAIN = len(train_set)
        net.opt.NUM_VAL = len(val_set)
        net.classes = train_set.classes
        with open(opt.CLASSES_PATH, 'wb+') as f:
            pickle.dump(train_set.classes, f)
        log("Number of Class:", len(net.classes), " Top3:", net.classes[:3])
        return train_loaders, val_loaders
    elif val_loader_type == SixBatch:
        opt.BATCH_SIZE = 6
        train_set = SixBatch(opt.TRAIN_PATH, opt, data_transforms['train'])
        val_set = SixBatch(opt.VAL_PATH, opt, data_transforms['val'])
        if opt.TEST_ALL:
            all_datasets = torch.utils.data.ConcatDataset([train_set, val_set])
        else:
            all_datasets = val_set
        all_loader = torch.utils.data.DataLoader(all_datasets, batch_size=opt.VAL_BATCH_SIZE,
                                                 shuffle=False, num_workers=opt.NUM_WORKERS)
        all_sizes = len(all_datasets)
        net.opt.NUM_VAL = all_sizes / 6
        net.classes = train_set.classes
        log("Number of Class:", len(net.classes), " Top3:", net.classes[:3])
        return all_loader
    else:
        train_set = train_loader_type(opt.TRAIN_PATH, data_transforms['train'])
        val_set = val_loader_type(opt.VAL_PATH, data_transforms['val'])
        train_loaders = torch.utils.data.DataLoader(train_set, batch_size=opt.BATCH_SIZE,
                                                    shuffle=True, num_workers=opt.NUM_WORKERS)
        val_loaders = torch.utils.data.DataLoader(val_set, batch_size=opt.VAL_BATCH_SIZE,
                                                  shuffle=False, num_workers=opt.NUM_WORKERS)
        net.opt.NUM_TRAIN = len(train_set)
        net.opt.NUM_VAL = len(val_set)
        net.classes = train_set.classes
        with open(opt.CLASSES_PATH, 'wb+') as f:
            pickle.dump(train_set.classes, f)
        log("Number of Class:", len(net.classes), " Top3:", net.classes[:3])
        return train_loaders, val_loaders





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
