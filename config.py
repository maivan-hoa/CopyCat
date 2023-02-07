# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:07:59 2023

@author: Mai Van Hoa - HUST
"""
import torch

CLASSES = {'bien': 0,
           'dung': 1,
           'long': 2,
           'thang': 3,
           'thanh': 4}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')