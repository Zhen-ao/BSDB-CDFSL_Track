# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
from __future__ import division

import torch
from PIL import Image
import numpy as np
import pandas as pd
import os
import random
import torchvision.transforms as transforms
import datasets.additional_transforms as add_transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from torchvision.datasets import ImageFolder

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append("../")
from configs import *


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


identity = lambda x:x
class UnlDataset:
    def __init__(self, dtarget, transform, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform

        self.meta = {}

        self.meta['image_names'] = []
        rnd_id = []

        dtarget = dtarget.upper()
        unl_list = os.path.join('unlabel_list', 'UNSUPERVISED_%s.txt' % dtarget)
        with open(unl_list, 'r') as f:
            count = len(f.readlines())
        if dtarget == 'ISIC':
            unl_path = ISIC_path + "/ISIC2018_Task3_Training_Input/"
        elif dtarget == 'CHESTX':
            unl_path = ChestX_path + "/images/"
            all_id = range(count)
            rnd_id = random.sample(all_id, 5000)
        elif dtarget == 'CROPDISEASE':
            unl_path = CropDisease_path + "/dataset/train/"
        elif dtarget == 'EUROSAT':
            unl_path = EuroSAT_path
        with open(unl_list, 'r') as f:
            for ind, x in enumerate(f.readlines()):
                if rnd_id == [] or ind in rnd_id:
                    img_path = os.path.join(unl_path, x.strip())
                    data = pil_loader(img_path)
                    self.meta['image_names'].append(data)
        print(len(self.meta['image_names']))

    def __getitem__(self, i):

        img = self.transform(self.meta['image_names'][i])

        return img

    def __len__(self):
        return len(self.meta['image_names'])

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size, dtarget):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        self.dtarget = dtarget

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=12,
                                  pin_memory=False)
        dataset_unl = UnlDataset(self.dtarget, transform)
        data_loader_unl = torch.utils.data.DataLoader(dataset_unl, **data_loader_params)
        return data_loader_unl


if __name__ == '__main__':
    pass
