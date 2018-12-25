# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from .BasicModule import *

def ConvLayer(in_channel, out_channel, kernel_size, stride, padding):
    '''
    This function is used to apply a combination of conv layer,
    batch_normal layer and Relu layer
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel,
                  out_channels=out_channel,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding),
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )


class PeleeNet(BasicModule):
    def __init__(self, config):
        super(PeleeNet, self).__init__(opt=config)
        self.MODEL_NAME = "PeleeNet"
        self.config = config

        # input shape(N, C_in, H, W)
        # stage0 : apply stem block
        self.stem_block_conv0 = ConvLayer(in_channel=3,
                                          out_channel=32,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1)
        self.stem_block_convl_1 = ConvLayer(32, 16, 1, 1, 0)
        self.stem_block_convl_2 = ConvLayer(16, 32, 3, 2, 1)
        self.stem_block_convr = nn.MaxPool2d(kernel_size=2,
                                             stride=2,
                                             padding=0)
        self.stem_block_convf = ConvLayer(64, 32, 1, 1, 0)

        # stage1 : apply dense block and transition block
        self.dense_block_1 = self.dense_block(in_channel=32,
                                              block_num=3,
                                              k=32)
        self.trasition_layer_1 = self.transition_layer(in_channel=128,
                                                       out_channel=128,
                                                       is_avgpooling=True)

        # stage2 : apply dense block and transition block
        self.dense_block_2 = self.dense_block(in_channel=128,
                                              block_num=4,
                                              k=32)
        self.trasition_layer_2 = self.transition_layer(in_channel=256,
                                                       out_channel=256,
                                                       is_avgpooling=True)

        # stage3 : apply dense block and transition block
        self.dense_block_3 = self.dense_block(in_channel=256,
                                              block_num=8,
                                              k=32)
        self.trasition_layer_3 = self.transition_layer(in_channel=512,
                                                       out_channel=512,
                                                       is_avgpooling=True)

        # stage4 : apply dense block and transition block without avg
        self.dense_block_4 = self.dense_block(in_channel=512,
                                              block_num=6,
                                              k=32)
        self.trasition_layer_4 = self.transition_layer(in_channel=704,
                                                       out_channel=704,
                                                       is_avgpooling=False)

        # stage5 : classification
        self.globel_avg = nn.AvgPool2d(7)
        self.fc = nn.Linear(704,self.config.NUM_CLASSES)

        # Initialize the PeleeNet
        self._initialize_weights()

    def forward(self, input):
        # The input shape (N, C_in, H, W), (batch_size, 3, 224, 224)
        # First get through stem block
        # shape (batch_size, 3, 224, 224)
        output = self.stem_block_conv0(input)
        # shape (batch_size, 32, 112, 112)
        output_l = self.stem_block_convl_1(output)
        output_l = self.stem_block_convl_2(output_l)
        # shape (batch_size, 32, 56, 56)
        output_r = self.stem_block_convr(output)
        # shape (batch_size, 32, 56, 56)
        output = torch.cat([output_l, output_r], 1)
        # shape (batch_size, 64, 56, 56)
        output = self.stem_block_convf(output)
        # shape (batch_size, 32, 56, 56)

        # Apply 4 stages with dense block and transition layer
        for stage, dense_block, transition_block in zip([1,2,3,4],
            [self.dense_block_1, self.dense_block_2, self.dense_block_3, self.dense_block_4],
            [self.trasition_layer_1, self.trasition_layer_2, self.trasition_layer_3, self.trasition_layer_4]):
            # Apply dense block
            for i in range(int(len(dense_block)/2)):
                convl = dense_block[2*i]
                convr = dense_block[2*i+1]
                output_l = convl(output)
                output_r = convr(output)
                output = torch.cat([output, output_l, output_r], 1)
            # Apply transition block
            if stage == 4:
                output = transition_block(output)
            else:
                conv, avg = transition_block
                output = conv(output)
                output = avg(output)

        # Apply classification
        # shape (batch_size, 704, 7, 7)
        output = self.globel_avg(output)
        output = self.fc(output.view((-1, 704)))
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight,
                                       nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def dense_block(self,in_channel, block_num, k=32):
        '''
        This part is used to apply dense block
        :param block_num: the num of dense layers
        :param k: the num of feature maps
        :return: a list contains layers
        '''
        blocks = nn.ModuleList([]) # used to store dense layers
        for i in range(block_num):
            # left channel, this won't change the size of image
            # and output channel is k/2
            convl = nn.Sequential(
                ConvLayer(k*i+in_channel, 2*k, 1, 1, 0),
                ConvLayer(2*k, int(k/2), 3, 1, 1)
            )
            # right channel, this won't change the size of image
            # and output channel is k/2
            convr = nn.Sequential(
                ConvLayer(k*i+in_channel, 2*k, 1, 1, 0),
                ConvLayer(2*k, int(k/2), 3, 1, 1),
                ConvLayer(int(k/2), int(k/2), 3, 1, 1)
            )
            # add to blocks
            blocks.extend([convl, convr])
        return blocks

    def transition_layer(self, in_channel, out_channel, is_avgpooling=True):
        conv0 = ConvLayer(in_channel, out_channel, 1, 1, 0)
        if is_avgpooling:
            avg = nn.AvgPool2d(2, 2, 0)
            return nn.ModuleList([conv0, avg])
        else:
            return conv0