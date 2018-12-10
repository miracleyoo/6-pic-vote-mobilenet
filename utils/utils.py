# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import os
import scipy.io
import pickle
import time
import argparse
from torch.utils.data import DataLoader
__all__ = ['gen_dataset', 'load_data', 'folder_init', 'str2bool', 'Timer']


def gen_dataset(data_loader, opt, if_all, data_root='./TempData/'):
    train_pairs, test_pairs = load_data(opt, data_root)

    test_dataset = data_loader(test_pairs, opt)
    test_loader = DataLoader(dataset=test_dataset, batch_size=opt.TEST_BATCH_SIZE, shuffle=False,
                             num_workers=opt.NUM_WORKERS, drop_last=False)

    opt.NUM_TEST = len(test_dataset)

    if if_all:
        train_pairs.extend(test_pairs)
        all_dataset = data_loader(train_pairs, opt)
        all_loader = DataLoader(dataset=all_dataset, batch_size=opt.TEST_BATCH_SIZE, shuffle=True,
                                num_workers=opt.NUM_WORKERS, drop_last=False)
        opt.NUM_TRAIN = len(all_dataset)
        return opt, all_loader, test_loader
    else:
        train_dataset = data_loader(train_pairs, opt)
        train_loader = DataLoader(dataset=train_dataset, batch_size=opt.BATCH_SIZE, shuffle=True,
                                  num_workers=opt.NUM_WORKERS, drop_last=False)
        opt.NUM_TRAIN = len(train_dataset)
        return opt, train_loader, test_loader


def load_data(opt, root='./Datasets/'):
    """
    :param opt:
    :Outputs:
        train_pairs : the path of the train  images and their labels' index list
        test_pairs  : the path of the test   images and their labels' index list
        class_names : the list of classes' names
    :param root : the root location of the dataset.

    Data Structure:
    train_data: dictionary, contains X_train and Y_train
    train_data['X_train'] :(6716, 2, 9, 41) num x channel x height x width
    train_data['Y_train'] :(6716, 369)

    """
    if opt.USE_NEW_DATA:
        data_path = [root + 'train_data_2.pkl', root + 'test_data_2.pkl']
        train_pairs = pickle.load(open(data_path[0], 'rb'))
        test_pairs = pickle.load(open(data_path[1], 'rb'))
        print("==> Load train data successfully.")
        print("==> Load test data successfully.")
        return train_pairs, test_pairs
    else:
        data_path = [root + 'train_data.mat', root + 'test_data.mat']

    train_data = scipy.io.loadmat(data_path[0])
    print("==> Load train data successfully.")
    test_data = scipy.io.loadmat(data_path[1])
    print("==> Load test data successfully.")

    train_data = dict((key, value) for key, value in train_data.items() if key == 'X_train' or key == 'Y_train')
    test_data = dict((key, value) for key, value in test_data.items() if key == 'X_test' or key == 'Y_test')
    train_pairs = [(x, y) for x, y in zip(train_data['X_train'], train_data['Y_train'])]
    test_pairs = [(x, y) for x, y in zip(test_data['X_test'], test_data['Y_test'])]

    return train_pairs, test_pairs


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
    if not os.path.exists('./source/val_results/'):
        os.mkdir('./source/val_results/')
    if not os.path.exists('source/simulation_res'):
        os.mkdir('source/simulation_res')
    if not os.path.exists('source/simulation_res/intermediate_file'):
        os.mkdir('source/simulation_res/intermediate_file')
    if not os.path.exists('source/simulation_res/train_data'):
        os.mkdir('source/simulation_res/train_data')
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
