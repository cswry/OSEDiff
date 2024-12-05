import os
import numpy as np
import cv2
import glob
import math
import yaml
import random
from collections import OrderedDict
import torch
import torch.nn.functional as F

from basicsr.data.transforms import augment
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import DiffJPEG, USMSharp, img2tensor, tensor2img
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize, rgb_to_grayscale)

cur_path = os.path.dirname(os.path.abspath(__file__))

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def opt_parse(opt_path):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)  # ignore_security_alert_wait_for_fix RCE

    return opt

class Codeformer_degradation(object):
    def __init__(self, opt_name='params_codeformer.yml', device='cpu'):
        opt_path = f'{cur_path}/{opt_name}'
        self.opt = opt_parse(opt_path)
        