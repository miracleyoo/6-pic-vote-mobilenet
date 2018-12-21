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

    if opt.START_PREDICT or opt.START_VOTE_PREDICT:
        if opt.START_VOTE_PREDICT:
            net.load(model_type="temp_model.dat")
            net = prep_net(net)
            val_loader = load_regular_data(opt, net, val_loader_type=SixBatch)
            vote_val(net, val_loader)
        net.load(model_type="temp_model.dat")
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
        fit(net, train_loader, val_loader)


if __name__ == '__main__':
    # Options and Argparses
    opt = Config()
    parser = argparse.ArgumentParser(description='Training')
    pros = [name for name in dir(opt) if not name.startswith('_')]
    abvs = ['-' + ''.join([j[:2] for j in i.split('_')]).lower()[:3] if len(i.split('_')) > 1 else
            '-' + i.split('_')[0][:3].lower() for i in pros]
    types = [type(getattr(opt, name)) for name in pros]
    with open('./reference/help_file.pkl', 'rb') as f:
        help_file = pickle.load(f)
    for i, abv in enumerate(abvs):
        if pros[i] in help_file.keys():
            help_line = help_file[pros[i]]
        else:
            help_line = "Currently no help doc provided."
        if types[i] == bool:
            parser.add_argument(abv, '--' + pros[i], type=str2bool, help=help_line)
        else:
            parser.add_argument(abv, '--' + pros[i], type=types[i], help=help_line)
    parser.add_argument('-gi', '--GPU_INDEX', type=str,
                        help='Index of GPUs you want to use')
    args = parser.parse_args()
    log(args)

    # Instantiate config
    opt = Config()

    # Overwrite config with input args
    for k, v in vars(args).items():
        if v is not None and hasattr(opt, k):
            setattr(opt, k, v)
            log(k, v, getattr(opt, k))
    if args.GPU_INDEX:
        print(args.GPU_INDEX)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_INDEX
    main()
