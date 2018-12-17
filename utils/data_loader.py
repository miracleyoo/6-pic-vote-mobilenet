# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import *
from utils.image_generater import divide_func
import random


class SixBatch(DatasetFolder):
    def __init__(self, root, opt, transform=None, target_transform=None, loader=default_loader):
        super(SixBatch, self).__init__(root, loader, IMG_EXTENSIONS,
                                       transform=transform,
                                       target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.opt = opt

    def __len__(self):
        return len(self.samples) * 6

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        ori_index = index // 6
        batch_index = index % 6
        process_func = divide_func(batch_index)

        path, target = self.samples[ori_index]
        sample = process_func(self.loader(path), self.opt.TENSOR_SHAPE[1])
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class SamplePairing(DatasetFolder):
    def __init__(self, root, opt, transform=None, target_transform=None, loader=default_loader):
        super(SamplePairing, self).__init__(root, loader, IMG_EXTENSIONS,
                                            transform=transform,
                                            target_transform=target_transform)
        self.lens = len(self.samples)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.lens

    def __getitem__(self, index):
        path1, label1 = self.samples[index]
        path2, label2 = self.samples[(index + random.randint(1, self.lens)) % self.lens]
        label = random.choice([label1, label2])
        sample1 = self.transform(self.loader(path1))
        sample2 = self.transform(self.loader(path2))
        if self.target_transform is not None:
            label = self.target_transform(label)
        sample = (sample1 + sample2) / 2
        return sample, label


class Template(Dataset):
    def __init__(self, data, opt):
        super(Template, self).__init__()
        self.data = data
        self.opt = opt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs, label = self.data[index]
        return torch.from_numpy(inputs).float(), torch.from_numpy(label).float()

# class Template(Dataset):
#     def __init__(self):
#         super(Template, self).__init__()
#         self.x = np.random.rand(64, 2, 41, 9)
#         self.y = np.ones(64).astype(np.int64)
#
#     def __len__(self):
#         return 64
#
#     def __getitem__(self, index):
#         inputs, label = self.x[index], self.y[index]
#         return np.array(inputs), np.array(label).astype(np.int64)
