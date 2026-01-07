# -- coding:utf-8 --

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import sys
import math
import random
import cv2
import time
import argparse
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, CenterCrop
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import ImageFolder
from torch.optim import lr_scheduler, SGD
import torch.utils.model_zoo as model_zoo
model_urls = {
            'AlexNet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
        }

def check_path(path):
    if os.path.exists(path):
        pass;
    else:
        os.makedirs(path);
def transform_function(resolution=224, is_train=False):
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    trans_train = transforms.Compose([
        transforms.RandomResizedCrop((resolution, resolution), scale=(0.2, 1.)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    to_pil = transforms.ToPILImage();
    trans_test = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if is_train:
        return trans_train
    else:
        return transform_test


def read_image(img_path):

    got_img = False
    import os
    from PIL import Image
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    img = None
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
    return img


def load_pretrained_weights(model, pretrained_weights, model_name, checkpoint_key="state_dict"):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, weights_only=False, map_location="cpu")

        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("The pretrained weights is not found at {}, so it will be downloaded from internet!".format(pretrained_weights))

        model.load_state_dict(model_zoo.load_url(model_urls[model_name]))



def pretrained_weights(model_choice):
    if model_choice == "AlexNet":
        return "./pretrainedweights/alexnet-owt-4df8aa71.pth"

    else:
        return None


def show_image(heatmap, img_path, name, save_dir, original=False):
    h, w = heatmap.shape
    img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
    ori_height, ori_width, _ = img_array.shape
    if original == True:
        heatmap_originalsize = cv2.resize(heatmap, (ori_width, ori_height), interpolation=cv2.INTER_LINEAR)
        colormap = cv2.applyColorMap(heatmap_originalsize, cv2.COLORMAP_JET);
        result = colormap * 0.4 + img_array * 0.8
    else:
        img_array_sqaure = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_LINEAR)
        colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET);
        result = colormap * 0.4 + img_array_sqaure * 0.8
    cv2.imwrite(os.path.join(save_dir, name + "_NAFlow.jpg"), result)
    return result





if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print("Finish!");
