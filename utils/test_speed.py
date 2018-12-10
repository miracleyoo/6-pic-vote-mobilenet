# coding: utf-8
# Author: Zhongyang Zhang# coding: utf-8
# Author: Zhongyang Zhang

import torch
import os
import numpy as np
import time
from torch.autograd import Variable
from models import miracle_net, miracle_wide_net, miracle_weight_wide_net, miracle_lineconv_net
time_elapsed = []


class Config(object):
    def __init__(self):
        self.USE_CUDA            = torch.cuda.is_available()
        self.NET_SAVE_PATH       = "./source/trained_net/"
        self.MODEL               = 'MiracleWeightWideNet'
        self.NUM_CHANNEL         = 2
        self.PROCESS_ID          = 'PADDING_LOSS1-2_WEI4-2-1-1_LESS_LAYER_TRAIN_ALL'
        self.LINER_HID_SIZE      = 1024
        self.LENGTH              = 41
        self.WIDTH               = 9
        self.NUM_CLASSES         = 369


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        global time_elapsed
        if self.name:
            print('==> [%s]:\t' % self.name, end='')
        time_elapsed.append(time.time() - self.tstart)
        print('Elapsed Time: %s (s)' % time_elapsed[-1])


def dl_init():
    opt = Config()
    if opt.MODEL == 'MiracleWeightWideNet':
        net = miracle_weight_wide_net.MiracleWeightWideNet(opt)
    elif opt.MODEL == 'MiracleWideNet':
        net = miracle_wide_net.MiracleWideNet(opt)
    elif opt.MODEL == 'MiracleNet':
        net = miracle_net.MiracleNet(opt)
    elif opt.MODEL == 'MiracleLineConvNet':
        net = miracle_lineconv_net.MiracleLineConvNet(opt)

    NET_SAVE_PREFIX = opt.NET_SAVE_PATH + opt.MODEL + '_' + opt.PROCESS_ID + '/'
    temp_model_name = NET_SAVE_PREFIX + "best_model.dat"
    if os.path.exists(temp_model_name):
        net, *_ = net.load(temp_model_name)
        print("Load existing model: %s" % temp_model_name)
        if opt.USE_CUDA:
            net.cuda()
            print("==> Using CUDA.")
    else:
        raise FileNotFoundError()
    return opt, net


def gen_input(net_charge_f=np.random.randint(3, 9, size=(1, 9, 41))):
    border_cond = np.zeros((1, 9, 41))
    border_cond[0, :, 0] = -0.6
    border_cond[0, :, -1] = 0.6
    model_input_f = np.concatenate((net_charge_f, border_cond), axis=0)
    return model_input_f[np.newaxis, :]


def dl_solver(model_input, net, opt):
    net.eval()
    if opt.USE_CUDA:
        inputs = Variable(torch.Tensor(model_input).cuda())
        outputs = net(inputs)
        outputs = outputs.cpu()
    else:
        inputs = Variable(torch.Tensor(model_input))
        outputs = net(inputs)
    outputs = outputs.data.numpy()
    return outputs


# with Timer('init_dl_core'):
opt, net = dl_init()

for i in range(100):
    with Timer('dl_solver'):
        model_input = gen_input()
        phi = dl_solver(model_input, net, opt)

print("Average time: {} (s)".format(sum(time_elapsed)/100))