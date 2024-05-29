import random
import logging

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms as transforms
import kornia.augmentation as K
import kornia.filters as KF
from kornia import image_to_tensor
from kornia.geometry.transform import resize
from utils.my_augment import Kornia_Randaugment

class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""
    def __init__(self, input_size=32):
        super().__init__()
        self.input_size = input_size

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        # x_tmp = np.array(x)  # HxWxC
        # x_out = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        x_out = resize(x.float() / 255.0, (self.input_size, self.input_size))
        return x_out

class DataAugmentation(nn.Module):

    def __init__(self, inp_size, mean, std) -> None:
        super().__init__()
        self.randaugmentation = Kornia_Randaugment(num_ops=2, magnitude=9)
        self.inp_size = inp_size
        self.mean = mean
        self.std = std
        self.preprocess = Preprocess(input_size=256)

        # additional_aug = self.randaugmentation.form_transforms()
        self.transforms = nn.Sequential(
            # K.Resize(size=(inp_size, inp_size)),
            # K.RandomCrop(size=(inp_size, inp_size)),
            K.CenterCrop(size=(inp_size, inp_size)),
            K.Normalize(mean, std)
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor, random=True) -> Tensor:
        # # if x is PIL type
        x = self.preprocess(x)

        if random:
            additional_aug = self.randaugmentation.form_transforms()

            transforms = nn.Sequential(
                # K.Resize(size=(self.inp_size, self.inp_size)),
                K.RandomCrop(size=(self.inp_size, self.inp_size)),
                K.RandomHorizontalFlip(),
                *additional_aug,
                K.Normalize(self.mean, self.std)
            )
        else:
            transforms = self.transforms
        x_out = transforms(x)  # BxCxHxW
        return x_out#.squeeze(0)
