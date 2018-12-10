# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch
from torch.utils.data import Dataset
import numpy as np

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