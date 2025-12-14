import torch
import random
import numpy as np

def set_seed(seed: int):
    """
    Set random seed for reproducibility across python, numpy, and torch.

    Inputs:
        seed (int): the seed value to set
    """
    # python random
    random.seed(seed)

    # numpy random
    np.random.seed(seed)

    # torch cpu and gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # make cudnn deterministic (may slow things down a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
