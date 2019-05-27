#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-05-23 18:07:43

@author: JimmyHua
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision.models import vgg, resnet

DECODER_HIDDEN_SIZE = 255
ENCODER_HIDDEN_SIZE = 256

DECODER_INPUT_SIZE = 256
DECODER_HIDDEN_SIZE = 256

DECODER_OUTPUT_FC = 256
DECODER_OUTPUT_FRAME = 10

V_FC = 50
V_SIZE = 50

class VGG11Base(nn.Module):
    
    def __init__(self):
        super(VGG11Base, self).__init__()
        vgg11 = vgg.vgg11_bn(pretrained=True)
        vgg11.features[14] = nn.MaxPool2d((2, 2), (2, 1), (0, 1)) # batch,256,4,26
        vgg11.features[21] = nn.MaxPool2d((2, 2), (2, 1), (0, 1)) # batch,512,2,27
        vgg11.features[22] = nn.Conv2d(
                512, 512, kernel_size=(2, 2), stride=(2, 1), padding=(0, 0))  # batch,512,1,26
        self.vgg11_base = vgg11.features[:25]
    
    def forward(self, inputs):
        return self.vgg11_base(inputs)
    
    def out_channels(self):
        return self.vgg11_base[-3].out_channels


class Attention_ocr(nn.Module):
    def __init__(self, use_gpu, NUM_CLASS):
        super(Attention_ocr, self).__init__()
        self.base_cnn = VGG11Base()
        self.NUM_CLASS = NUM_CLASS
        FEATURE_C = self.base_cnn.out_channels()  #512
        self.lstm = nn.LSTM(input_size=FEATURE_C, hidden_size=DECODER_HIDDEN_SIZE, 
                            batch_first=True, bidirectional=True)
        self.rnn_cell = nn.GRUCell(input_size=DECODER_INPUT_SIZE, hidden_size=DECODER_HIDDEN_SIZE)
        self.layer_cx = nn.Linear(in_features=NUM_CLASS, out_features=DECODER_INPUT_SIZE)
        self.layer_ux = nn.Linear(in_features=FEATURE_C, out_features=DECODER_INPUT_SIZE)
        self.layer_so = nn.Linear(in_features=DECODER_HIDDEN_SIZE, out_features=DECODER_OUTPUT_FC)
        self.layer_uo = nn.Linear(in_features=FEATURE_C, out_features=DECODER_OUTPUT_FC)
        self.layer_oo = nn.Linear(in_features=DECODER_OUTPUT_FC, out_features=NUM_CLASS)
        self.layer_sa = nn.Linear(in_features=DECODER_HIDDEN_SIZE, out_features=V_FC)
        self.layer_fa = nn.Linear(in_features=DECODER_HIDDEN_SIZE * 2, out_features=V_FC)
        self.layer_va = nn.Linear(in_features=V_FC, out_features=V_SIZE)
        self.layer_aa = nn.Linear(in_features=V_SIZE, out_features=1)
        self.use_gpu = use_gpu
    
    def forward(self, inputs, labels=None, return_alpha=False):
        if self.training:
            assert(labels is not None)
        batch_size = inputs.shape[0]
        '''batch_size * feature_c * (feature_h * feature_w)'''
        f = self.base_cnn(inputs)  #torch.Size([3, 512, 1, 26])
        '''batch_size * (feature_h * feature_w) * feature_c = 
           batch_size * seq_len * feature_c'''
        f = f.view(batch_size, f.shape[1], -1).transpose(1, 2)  # [3,26,512]
        '''batch_size * seq_len * (hidden_size * 2dirs)'''
        f, _ = self.lstm(f)  # [3,26,512]
        '''batch_size * (hidden_size * 2dirs), seq_len'''
        f = f.transpose(1, 2) # [3,512,26]
        #initial c0, s0, ei, ej, do not require gradient
        c = torch.zeros(batch_size, self.NUM_CLASS)
        s = torch.zeros(batch_size, DECODER_HIDDEN_SIZE)
        if self.use_gpu:
            c, s = c.cuda(), s.cuda()
        outputs = []
        alphas = []
        for frame in range(DECODER_OUTPUT_FRAME):
            alpha, u = self._get_alpha_u(f, s)
            alphas.append(alpha.view(batch_size, -1))
            x = self.layer_ux(u) + self.layer_cx(c)  # 3,256
            s = self.rnn_cell(x, s)
            o = self.layer_uo(u) + self.layer_so(s)
            o = self.layer_oo(nn.Tanh()(o))  #torch.Size([3, 37])
            outputs.append(o)

            #update c from o(evaluating) or ground truth(training)
            if self.training:
                c = labels[:, :, frame]
            else:
                c = nn.Softmax(dim=1)(o).detach()
                c = (c == c.max(1, keepdim=True)[0]).float()
        outputs = torch.stack(outputs, dim=-1)  #torch.Size([3, 37, 7])
        if return_alpha:
            alphas = torch.stack(alphas, dim=-1) #torch.Size([3, 26, 7])
            return outputs, alphas
        return outputs
    
    def _get_alpha_u(self, f, s):
        a = self.layer_va(nn.Tanh()(
                self.layer_fa(f.transpose(1, 2)) + self.layer_sa(s).unsqueeze(1)))
        a = self.layer_aa(nn.Tanh()(a)).squeeze(-1)
        alpha = nn.Softmax(dim=1)(a) #[3,26]
        u = (f * alpha.unsqueeze(1)).sum(-1)
        return alpha, u

# check the model
if __name__ == '__main__':   
    x = torch.randn(3,3,32,100)
    vgg_ = vgg.vgg11_bn(pretrained=True)
    vgg_v = OldAttention()
    a=vgg_v(x)