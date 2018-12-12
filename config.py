# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import torch


class Config(object):
    def __init__(self):
        # Action definition
        self.USE_CUDA            = torch.cuda.is_available()
        self.LOAD_SAVED_MOD      = True
        self.SAVE_TEMP_MODEL     = True
        self.SAVE_BEST_MODEL     = True
        self.MASS_TESTING        = False
        self.TRAIN_ALL           = False
        self.TEST_ALL            = False
        self.USE_NEW_DATA        = False
        self.TO_MULTI            = False
        self.SAVE_EVERY          = 1

        # Tensor shape definition
        self.BATCH_SIZE          = 4
        self.EVAL_BATCH_SIZE     = 64
        self.NUM_CHANNEL         = 3
        self.RESIZE              = 448
        self.LINER_HID_SIZE      = 128

        # Program information
        self.CRITERION           = torch.nn.CrossEntropyLoss()
        self.OPTIMIZER           = "Adam"
        self.TRAIN_DATA_RATIO    = 0.7
        self.THREADHOLD          = 0.0005
        self.NUM_EPOCHS          = 500
        self.NUM_CLASSES         = 250
        self.NUM_EVAL            = 1
        self.NUM_TRAIN           = 1
        self.TOP_NUM             = 1
        self.NUM_WORKERS         = 0

        # Hyper parameters
        self.LEARNING_RATE       = 0.001

        # Name and path definition
        self.NET_SAVE_PATH       = "./source/trained_net/"
        self.TRAIN_PATH          = "../cards_250_7/cards_for_train"
        self.EVAL_PATH           = "../cards_250_7/cards_for_eval"
        self.CLASSES_PATH        = "./source/classes.json"
        self.MODEL               = "MobileNetV2"
        self.PROCESS_ID          = "Test03_250_Sigmoid"
        if self.TRAIN_ALL:
            self.PROCESS_ID += '_TRAIN_ALL'
        self.SUMMARY_PATH        = "./source/summary/"+self.MODEL+'_'+self.PROCESS_ID+'_'
