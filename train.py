#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-05-23 16:30:48

@author: JimmyHua
"""
import argparse
import logging
import sys
import torch
from torch.backends import cudnn
from model.model import Attention_ocr
from core import losses
from data.dataset import get_dataset
from core.train_engine import Train_Engine
#import os
#os.environ['CUDA_VISIBLE_DEVICES']="0"

parser = argparse.ArgumentParser()
parser.add_argument('--TRAIN_DIR',  default='/data/huachunrui/datasets/chinese_string_simple/train.txt')
parser.add_argument('--TEST_DIR',  default='/data/huachunrui/datasets/chinese_string_simple/test.txt')
parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--input_h', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--input_w', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--max_seq_len', type=int, default=10, help='the max sequence length')
parser.add_argument('--use_gpu', action='store_true', help='enables cuda', default=True)
parser.add_argument('--epochs', type=int, default=300, help='training epoc')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, default=0.00005')
parser.add_argument('--print_interval', type=int, default=100, help='how many iterations to print')
parser.add_argument('--eval_step', type=int, default=1, help='how many epochs to evaluate')
parser.add_argument('--save_step', type=int, default=1, help='how many epochs to save model')
parser.add_argument('--save_dir', type=str, default='checkpoints', help='save model directory')
opt = parser.parse_args()
print(opt)

logging.basicConfig(
    level = logging.INFO, #打印日志级别数值
    format = '%(asctime)s: %(message)s', #输出时间和信息
    stream=sys.stdout #指定日志的输出流
    )

cudnn.benchmark = True

logging.info('=========== Starting Training ============')
train_data, test_data, char_to_index, index_to_char, n_class = get_dataset(opt)

net = Attention_ocr(use_gpu=opt.use_gpu,NUM_CLASS=n_class)

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999))
criterion = losses.Attention_loss()

net = torch.nn.DataParallel(net)
net = net.cuda()
model = Train_Engine(net)
model.fit(index_to_char,train_data=train_data, test_data=test_data, optimizer=optimizer, criterion=criterion, epochs=opt.epochs, 
         print_interval=opt.print_interval, eval_step=opt.eval_step, save_step=opt.save_step, save_dir=opt.save_dir, use_gpu=opt.use_gpu)