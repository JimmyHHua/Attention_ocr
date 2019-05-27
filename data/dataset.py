#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-05-23 14:35:29

@author: JimmyHua
"""
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage.transform import resize as imresize
import numpy as np
from PIL import Image
import torch
import os
import cv2
import json

class CNDATA(Dataset):
    def __init__(self, img_base, img_transforms, label_transforms):
        super(CNDATA, self).__init__()
        self.img_base = img_base
        self.img_transforms = img_transforms
        self.label_transforms = label_transforms
        
    def __getitem__(self, index):
        info = self.img_base[index].split(' ')
        img_path = info[0]
        img = cv2.imread(img_path)
        H, W, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype('float')/127.5-1
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        label = info[1]
        if self.img_transforms:
            img = self.img_transforms(img)
        if self.label_transforms:
            label = self.label_transforms(label)
        return img, label

    def __len__(self):
        return len(self.img_base)

 # One_hot   
def index_to_onehot(idx, num_class):
    '''idx -> a list of int indices, like [1, 3, 6, 9]
       num_class -> number of classes'''
    assert(max(idx) < num_class)
    return (np.arange(num_class) == np.array(idx)[:, None]).astype(np.float32)   
    

def label_transforms(char_index, num_class, max_seq_len):
    def str_to_index(label):
        return [char_index[_] for _ in label]
    def pad_label(index_label):
        diff_w = max_seq_len - len(index_label)
        return np.array(index_label + [num_class - 1] * diff_w)
    return transforms.Compose([
        transforms.Lambda(str_to_index),
        transforms.Lambda(pad_label),
        transforms.Lambda(lambda label: index_to_onehot(label, num_class)),
        transforms.Lambda(lambda label: label.transpose([1, 0]).astype(np.float32)),
        transforms.Lambda(lambda label: torch.from_numpy(label)),
        ])

def cn_transform(input_h, input_w):
    def resize_with_ratio(x):
        return imresize(x, (input_h,input_w), mode='constant')

    return transforms.Compose([
            transforms.Lambda(resize_with_ratio),
            transforms.Lambda(lambda x: x.transpose([2, 0, 1]).astype(np.float32)),
            transforms.Lambda(lambda x: torch.from_numpy(x))
            ])

def get_label(label_path):
    with open(label_path,'rb') as f:
        lines = f.readlines()
    return np.array([_.decode('utf-8').strip() for _ in lines])


def get_dataset(opt):
    TRAIN_INFO=get_label(opt.TRAIN_DIR)
    TEST_INFO=get_label(opt.TEST_DIR)
    UNIQUE_CHAR = set(',')

    for label in np.hstack((TRAIN_INFO, TEST_INFO)):
        try:
            label_list = list(label.split()[1])
        except:
            print(path)
        for l in label_list:
            if not l in UNIQUE_CHAR:
                UNIQUE_CHAR.add(l)
                
    # Write the unique char to the file
    with open('unique_char.txt','w',encoding='utf-8') as f:
       for char in sorted(list(UNIQUE_CHAR)):
           f.write(char.strip()+'\n')

    char_to_index = {x:y for x, y in zip(
        sorted(list(UNIQUE_CHAR))+['eof'], range(len(UNIQUE_CHAR)+1))}

    index_to_char = {y:x for x, y in zip(
        sorted(list(UNIQUE_CHAR))+[' '], [str(_) for _ in range(len(UNIQUE_CHAR)+1)])}

    # Write index_to_char into json file
    with open('index_to_char.json','r',encoding='utf-8') as f:
        index_to_char = json.load(f)

    n_class = len(UNIQUE_CHAR) + 1

    train_dataset = CNDATA(TRAIN_INFO, img_transforms=cn_transform(opt.input_h, opt.input_w),
                            label_transforms=label_transforms(char_to_index, n_class, opt.max_seq_len))

    test_dataset = CNDATA(TEST_INFO, img_transforms=cn_transform(opt.input_h, opt.input_w), 
                            label_transforms=label_transforms(char_to_index, n_class, opt.max_seq_len))

    train_data = DataLoader(train_dataset, batch_size=opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    test_data = DataLoader(test_dataset, batch_size=opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers, pin_memory=True)

    return train_data, test_data, char_to_index, index_to_char, n_class
