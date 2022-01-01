import copy
from dataclasses import dataclass
from functools import partial
import time
import warnings

import numpy as np
from PIL import Image
import torch
from torch import optim, nn
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.transforms import functional as TF

filename = './images/content/ust4.jpg'
image = Image.open(filename)
image_interp = interpolate(exp_avg, shape, mode='bicubic')
