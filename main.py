# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

from config import Config
from models import MobileNetV2
from utils.utils import *
from data_loader import SixBatch, SamplePairing


def main():
    # Initializing Configs
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

    if opt.LOAD_SAVED_MOD:
        net.load()
    if opt.TO_MULTI:
        net.to_multi()
    else:
        net.to(net.device)
    if net.pre_epoch == 0 and opt.ADD_SUMMARY and not opt.MASS_TESTING:
        add_summary(opt, net)

    if opt.MASS_TESTING:
        eval_loader = load_regular_data(opt, net, loader_type=SixBatch)
        net.vote_eval(eval_loader)
    else:
        train_omit(opt, net, 500, False)
        for x in range(3):
            train_omit(opt, net, 20, True)
            train_omit(opt, net, 200, False)


def train_omit(opt, net, epochs, use_sp):
    net.opt.NUM_EPOCHS = epochs
    if use_sp == True:
        _, eval_loader = load_regular_data(opt, net, loader_type=ImageFolder)
        net.opt.USE_SP = use_sp
        train_loader = load_regular_data(opt, net, loader_type=SamplePairing)
        print("==> SamplePairing datasets are generated successfully.")
    else:
        net.opt.USE_SP = use_sp
        train_loader, eval_loader = load_regular_data(opt, net, loader_type=ImageFolder)
        print("==> All datasets are generated successfully.")
    net.fit(train_loader, eval_loader)


if __name__ == '__main__':
    # Options
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-lsm', '--LOAD_SAVED_MOD', type=str2bool,
                        help='If you want to load saved model')
    parser.add_argument('-gi', '--GPU_INDEX', type=str,
                        help='Index of GPUs you want to use')
    parser.add_argument('-mt', '--MASS_TESTING', type=str2bool,
                        help='If you want to start mass testing')
    parser.add_argument('-sp', '--START_PREDICT', type=str2bool,
                        help='If you want to start predicting.')
    parser.add_argument('-as', '--ADD_SUMMARY', type=str2bool,
                        help='If you want to add graph to summary.')
    parser.add_argument('-bs', '--BATCH_SIZE', type=int,
                        help='If you want to start mass testing')
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
