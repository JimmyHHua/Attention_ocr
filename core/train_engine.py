#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-05-23 17:18:26

@author: JimmyHua
"""

import logging
import time
import numpy as np
import torch
from core.meter import AverageValueMeter
import shutil
import os

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
# Save the model        
def save_checkpoint(state,is_best,save_dir,filename='checkpoint.pt'):
    fpath='_'.join((str(state['epoch']),filename))
    fpath=os.path.join(save_dir,fpath)
    make_dir(save_dir)
    torch.save(state,fpath)
    if is_best:
        shutil.copy(fpath,os.path.join(save_dir,'model_best.pt'))

class Train_Engine(object):
    def __init__(self,net):
        self.net = net
        self.loss = AverageValueMeter()
        self.seq_acc = AverageValueMeter()
        self.char_acc = AverageValueMeter()

    def fit(self, index_to_char, train_data, test_data, optimizer, criterion, epochs=300, print_interval=100, eval_step=1, save_step=1, save_dir='checkpoint',use_gpu=True):
        best_test_acc = 0.0
        for epoch in range(0,epochs):
            self.loss.reset()
            self.seq_acc.reset()
            self.char_acc.reset()
            self.net.train()

            tic = time.time()
            btic = time.time()

            for i, data in enumerate(train_data):
                imgs,labels = data
                if use_gpu:
                    labels = labels.cuda()  #torch.Size([1, 5586, 10])
                    imgs = imgs.cuda()
                outputs = self.net(imgs,labels)
                loss = criterion(outputs,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.loss.add(loss.item())
                seq_acc = (outputs.max(1)[1].long()==labels.max(1)[1].long()).prod(1).float().mean()
                char_acc = (outputs.max(1)[1].long()==labels.max(1)[1].long()).float().mean()

                self.seq_acc.add(seq_acc.item())
                self.char_acc.add(char_acc.item())

                if print_interval and (i+1)%(5*print_interval)==0:
                    pred_text = ''.join([index_to_char[str(_)] for _ in outputs[0].max(0)[1].cpu().numpy().tolist()])
                    label_text = ''.join([index_to_char[str(_)] for _ in labels[0].max(0)[1].cpu().numpy().tolist()])
                    # show the predicted text and the gt text
                    logging.info('%-11s ===>  gt: %-11s' % (pred_text, label_text))

                if print_interval and (i+1)%print_interval==0:
                    loss_mean = self.loss.value()[0]
                    seq_acc_mean = self.seq_acc.value()[0]
                    char_acc_mean = self.char_acc.value()[0]
                    logging.info('Epoch: %d\tBatch: %d\tloss=%f\tseq_acc=%f\tchar_acc=%f' 
                                % (epoch, i + 1, loss_mean, seq_acc_mean, char_acc_mean))

            loss_mean = self.loss.value()[0]
            seq_acc_mean = self.seq_acc.value()[0]
            char_acc_mean = self.char_acc.value()[0]

            logging.info('Epoch: %d\ttraining: loss=%f\tepoch_seq_acc=%f\tepoch_char_acc=%f' 
                        % (epoch, loss_mean, seq_acc_mean, char_acc_mean))

            logging.info('Epoch %d\t cost: %f' % (epoch, time.time() - tic))
            
            is_best=False
            if test_data is not None and (epoch+1)%eval_step==0:
                test_seq_acc, test_char_acc = self.val(test_data)
                logging.info('---->> Epoch: %d\ttest_seq_acc=%f\ttest_char_acc=%f' % (epoch, test_seq_acc, test_char_acc))
                is_best = test_seq_acc > best_test_acc
                if is_best:
                    best_test_acc = test_seq_acc
            state_dict = self.net.module.state_dict()
            if not (epoch+1)%save_step:
                save_checkpoint({
                    'state_dict':state_dict,
                    'epoch':epoch+1
                    },is_best=is_best,save_dir=save_dir
                    )
        print('Finished\n')

    # val the model
    def val(self,test_data):
        num_correct = 0
        num_imgs = 0
        self.net.eval()
        for data in test_data:
            imgs,labels=data
            labels = labels.cuda()
            outputs = self.net(imgs)
            char_acc = (outputs.max(1)[1].long()==labels.max(1)[1].long()).float().mean()
            seq_acc = (outputs.max(1)[1].long()==labels.max(1)[1].long()).prod(1).float().mean()

        return seq_acc, char_acc