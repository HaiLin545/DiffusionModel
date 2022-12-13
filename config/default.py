import yaml

DEFAULTS = {
    "timesteps": 400,
    "result_folder": "./results",
    "ckpt_folder": "./results/ckpt",
    "image_size": 28,
    "channels": 1,
    "batch_size": 128,
    "save_every": 1000,
    "epoch": 5,
    'lr': 1e-3,
    'printcfg': False
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_default_config():
    config = DEFAULTS
    return config


def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(defaults, config)
    return config


def printcfg(cfg):
    for k, v in cfg:
        print(k, v)
