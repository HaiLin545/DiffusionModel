import yaml
from easydict import EasyDict

config = EasyDict()
config.DEVICE = "cuda:0"


config.DATASET = EasyDict()
config.DATASET.IMAGE_SIZE = 28
config.DATASET.CHANNELS = 1
config.DATASET.BATCH_SIZE = 128

config.TRAIN = EasyDict()
config.TRAIN.CKPT_DIR = "output/ckpt"
config.TRAIN.OUTPUT_DIR = "output"
config.TRAIN.OUTPUT_FOLDER = "output"
config.TRAIN.EPOCH = 5
config.TRAIN.LR = 1e-3
config.TRAIN.PRINT_CFG = False
config.TRAIN.SAVE_EVERY_ITER = 200
config.TRAIN.SAVE_EVERY_EPOCH = 2

config.DM = EasyDict()
config.DM.TIME_STEPS = 400

config.TEST = EasyDict()


def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def _print_cfg(k, v):
    print(f"{k} = {v}\n")


def print_cfg(cfg=config):
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(k)
            print_cfg(v)
        else:
            _print_cfg(k, v)
