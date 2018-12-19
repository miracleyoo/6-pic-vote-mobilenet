# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

from config import Config
from models.MobileNetV2 import *
from utils.utils import *


def main():
    # Initializing Configs
    folder_init(opt)
    net = None

    # Initialize model
    try:
        if opt.MODEL_NAME == 'MobileNetV2':
            net = MobileNetV2(opt)
    except KeyError('Your model is not found.'):
        exit(0)
    finally:
        log("Model initialized successfully.")

    if opt.MASS_TESTING:
        net.load(model_type="best_model.dat")
        net = prep_net(net)
        val_loader = load_regular_data(opt, net, val_loader_type=SixBatch)
        vote_val(net, val_loader)
    elif opt.START_PREDICT:
        net.load(model_type="best_model.dat")
        net = prep_net(net)
        _, val_loader = load_regular_data(opt, net, val_loader_type=ImageFolder)
        predict(net, val_loader)
    else:
        if opt.LOAD_SAVED_MOD:
            net.load()
        net = prep_net(net)
        if net.opt.DATALOADER_TYPE == "SamplePairing":
            train_loader, val_loader = load_regular_data(opt, net, train_loader_type=SamplePairing)
            log("SamplePairing datasets are generated successfully.")
        elif net.opt.DATALOADER_TYPE == "ImageFolder":
            train_loader, val_loader = load_regular_data(opt, net, train_loader_type=ImageFolder)
            log("All datasets are generated successfully.")
        else:
            raise KeyError("Your DATALOADER_TYPE doesn't exist!")
        train_omit(train_loader, val_loader, net, opt.NUM_EPOCHS)


def prep_net(net):
    if opt.TO_MULTI:
        net = net.to_multi()
    else:
        net.to(net.device)
    if net.epoch_fin == 0 and opt.ADD_SUMMARY and not opt.MASS_TESTING:
        add_summary(opt, net)
    return net


def train_omit(train_loader, val_loader, net, epochs):
    net.opt.NUM_EPOCHS = epochs
    fit(net, train_loader, val_loader)


if __name__ == '__main__':
    # Options
    opt = Config()
    parser = argparse.ArgumentParser(description='Training')
    pros = [name for name in dir(opt) if not name.startswith('_')]
    abvs = ['-' + ''.join([j[:2] for j in i.split('_')]).lower()[:3] if len(i.split('_')) > 1 else
            '-' + i.split('_')[0][:3].lower() for i in pros]
    types = [type(getattr(opt, name)) for name in pros]
    for i, abv in enumerate(abvs):
        if types[i] == bool:
            parser.add_argument(abv, '--' + pros[i], type=str2bool)
        else:
            parser.add_argument(abv, '--' + pros[i], type=types[i])
    parser.add_argument('-gi', '--GPU_INDEX', type=str,
                        help='Index of GPUs you want to use')
    args = parser.parse_args()
    log(args)
    opt = Config()
    for k, v in vars(args).items():
        if v is not None and hasattr(opt, k):
            setattr(opt, k, v)
            log(k, v, getattr(opt, k))
    if args.GPU_INDEX:
        print(args.GPU_INDEX)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_INDEX
    main()
