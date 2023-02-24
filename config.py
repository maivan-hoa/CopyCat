# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:07:59 2023

@author: Mai Van Hoa - HUST
"""
import torch

CLASSES = {'Linh': 0,
           'Minh': 1,
           'Quyet': 2,
           'Thang': 3,
           'Tung': 4}

NUM_CLASSES = len(CLASSES)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')