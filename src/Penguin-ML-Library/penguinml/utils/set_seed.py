import os
import random

import numpy as np
import tensorflow as tf
import torch

from penguinml.utils.logger import get_logger


def seed_base(seed: int = 42):
    """base(random, os, numpy)"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    get_logger(__name__).info(f"set seed: {seed}")


def seed_base_tf(seed: int = 42):
    """base + tensorflow"""
    seed_base(seed)
    tf.random.set_seed(seed)


def seed_base_torch(seed: int = 42):
    """base + torch"""
    seed_base(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
