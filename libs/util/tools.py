import torch
import random
import numpy as np
from torch.backends import cudnn
from inspect import isfunction


def fix_random_seed(cfg):
    seed = cfg.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.enabled = cfg.CUDNN.ENABLE


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
