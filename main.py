# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

from utils.utils import *
from data_loader import *
from torchvision.datasets import ImageFolder
from torchvision import transforms
from config import Config
from models import MobileNetV2
from tensorboardX import SummaryWriter
import argparse
import torch
import os


def main():
    # Initializing configs
    folder_init(opt)
    net = None

    # Initialize model
    try:
        if opt.MODEL == 'MobileNetV2':
            net = MobileNetV2.MobileNetV2(opt)
    except KeyError('Your model is not found.'):
        exit(0)
    finally:
        print("==> Model initialized successfully.")

    # Instantiation of tensorboard and add net graph to it
    print("==> Adding summaries...")
    writer = SummaryWriter(opt.SUMMARY_PATH)
    dummy_input = torch.rand(opt.BATCH_SIZE, opt.NUM_CHANNEL, opt.WIDTH, opt.LENGTH)

    try:
        writer.add_graph(net, dummy_input)
    except KeyError:
        writer.add_graph(net.module, dummy_input)

    if opt.LOAD_SAVED_MOD:
        net.load()
    if opt.TO_MULTI:
        net.to_multi()
    else:
        net.to(net.device)

    # Initialize Data
    def load_data(resize):

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomSizedCrop(max(resize)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'eval': transforms.Compose([
                # Higher scale-up for inception
                transforms.Resize(int(max(resize) / 224 * 256)),
                transforms.CenterCrop(max(resize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        data_dir = "../cards_250_7/cards_for_"
        dsets = {x: ImageFolder(data_dir+x, data_transforms[x])
                 for x in ['train', 'eval']}
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=opt.BATCH_SIZE,
                                                       shuffle=True)
                        for x in ['train', 'eval']}
        # dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
        # dset_classes = dsets['train'].classes

        return dset_loaders['train'], dset_loaders['eval']

    train_loader, eval_loader = load_data((224, 224))

    # train_data = ImageFolder(opt.TRAIN_PATH)
    # eval_data = ImageFolder(opt.EVAL_PATH)

    # train_loader = torch.utils.data.DataLoader(train_data,
    #                                         batch_size=opt.BATCH_SIZE,
    #                                         shuffle=True,
    #                                         num_workers=opt.NUM_WORKERS,
    #                                         )
    # eval_loader = torch.utils.data.DataLoader(eval_data,
    #                                           batch_size=opt.BATCH_SIZE,
    #                                           shuffle=True,
    #                                           num_workers=opt.NUM_WORKERS,)
    print("==> All datasets are generated successfully.")
    net.fit(train_loader, eval_loader)
    net.predict(eval_loader)



if __name__ == '__main__':
    # Options
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-lsm', '--LOAD_SAVED_MOD', type=str2bool,
                        help='If you want to load saved model')
    parser.add_argument('-gi', '--GPU_INDEX', type=str,
                        help='Index of GPUs you want to use')

    args = parser.parse_args()
    print(args)
    opt = Config()
    for k, v in vars(args).items():
        if v is not None and hasattr(opt, k):
            setattr(opt, k, v)
            print(k, v, getattr(opt, k))
    if args.GPU_INDEX:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_INDEX
    main()
