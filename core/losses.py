#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date:   2019-05-23 17:07:20

@author: JimmyHua
"""
import torch

class Attention_loss():
    def __call__(self,outputs, labels, eof_weight=1):
        blank_index = (labels.max(1)[1] == labels.shape[1] - 1)
        char_index = 1 - blank_index
        cl = labels.max(1)[1][char_index]
        co = outputs.transpose(1, 2)[torch.unbind(char_index.nonzero(), 1)]  #[10, 5586]
        cross_entropy = torch.nn.CrossEntropyLoss()
        loss = cross_entropy(co, cl)

        return loss