# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import sys
sys.path.append('..')
import torch
from torch.autograd import Variable
from torchviz import make_dot
from models import MobileNetV2
from config import Config

opt = Config()

x = Variable(torch.randn(128, 3, 256, 256))  # change 12 to the channel number of network input
model = MobileNetV2.MobileNetV2()
y = model(x)
g = make_dot(y)
g.view()
