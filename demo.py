#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-05-27 10:13:01

@author: JimmyHua
"""

import cv2
from skimage.transform import resize as imresize
from model.model import Attention_ocr
import time
import glob
import torch
import numpy as np
import json,argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img',  default='image/*', help='the images we want to predict')
parser.add_argument('--input_h', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--input_w', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--use_gpu', action='store_true', help='enables cuda', default=True)
parser.add_argument('--index_to_char', type=str, default='data/index_to_char.json', help='index_to_char')
parser.add_argument('--checkpoints', type=str, default='checkpoints/model_best.pt', help='checkpoints model directory')
opt = parser.parse_args()

def main(opt):
    with open(opt.index_to_char,'r',encoding='utf-8') as f:
        index_to_char = json.load(f)
    n_class = len(index_to_char)
    net = Attention_ocr(use_gpu=True,NUM_CLASS=n_class)
    net.load_state_dict(torch.load(opt.checkpoints)['state_dict'])
    net = torch.nn.DataParallel(net)
    net = net.cuda()
    net.eval()
    print('====== Start Ocr ======')
    img_pathes = sorted(glob.glob(opt.img))
    for path in img_pathes:
        img = cv2.imread(path)
        img = img.astype('float')/127.5-1
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        img = imresize(img, (opt.input_h, opt.input_w), mode='constant')
        img = torch.from_numpy(img.transpose([2,0,1]).astype(np.float32))[None,...]
        img = img.cuda()
        t1 = time.time()
        output = net(img)
        output = output.max(1)[1].squeeze(0)                                        
        text = ''.join([index_to_char[str(_)] for _ in output.tolist()])
        print('Path: ',path,'\t===>>>\t',text,'\t===>>>\t', 'time cost: %3f'%(time.time()-t1))

if __name__ == '__main__':
    main(opt)
