import os
import pathlib
import random
import numpy as np
import torch


def config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_filenames(filepath):
    filenames = [os.path.join(filepath, f)
                 for f in os.listdir(filepath)
                 if os.path.isfile(os.path.join(filepath, f))]
    return sorted(filenames)


def get_dirs(filepath):
    filenames = [os.path.join(filepath, f)
                 for f in os.listdir(filepath)
                 if os.path.isdir(os.path.join(filepath, f))]
    return sorted(filenames)


def is_file(fname):
    return os.path.exists(fname)


def mkdir(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
