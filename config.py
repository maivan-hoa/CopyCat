# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:07:59 2023

@author: Mai Van Hoa - HUST
"""
import torch

CLASSES = {'Bien': 0,
           'Minh': 1,
           'Quyet': 2,
           'Thang': 3,
           'Tung': 4}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')