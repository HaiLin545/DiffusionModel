import yaml
from easydict import EasyDict

config = EasyDict()
config.DEVICE = "cuda:0"
config.DEBUG = False
config.CKPT_DIR = "ckpt"
config.OUTPUT_DIR = "output"
config.OUTPUT_FOLDER = "output"
config.LOG_FILE = "result.log"
config.PRINT_CFG = False
config.BACKUP_CODE = False
config.SEED = 123456789

config.DATASET = EasyDict()
config.DATASET.NAME = ""
config.DATASET.ROOT = "data/"
config.DATASET.DIR = ""
config.DATASET.NUM_WORKER = 8
config.DATASET.IMAGE_SIZE = 28
config.DATASET.CHANNELS = 1
config.DATASET.BATCH_SIZE = 128

config.TRAIN = EasyDict()
config.TRAIN.RESUME = None
config.TRAIN.EPOCH = 5
config.TRAIN.LR = 1e-3
config.TRAIN.SAVE_EVERY_ITER = 200
config.TRAIN.SAVE_EVERY_EPOCH = 2
config.TRAIN.SCHEDULER = "StepLR"
config.TRAIN.LR_DECAY = 0.9
config.TRAIN.LR_DECAY_STEP = 1000


config.DM = EasyDict()
config.DM.TIME_STEPS = 400

config.TEST = EasyDict()

config.CUDNN = EasyDict()
config.CUDNN.BENCHMARK = False
config.CUDNN.DETERMINISTIC = True
config.CUDNN.ENABLE = True


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


def _print_cfg(k, v, output_file):
    content = f"{k} = {v}"
    print(content)


def print_cfg(cfg=config, output_file=None):
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(f"------{k}------")
            print_cfg(v, output_file)
        else:
            _print_cfg(k, v, output_file)
