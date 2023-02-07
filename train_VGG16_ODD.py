# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:14:11 2023

@author: Mai Van Hoa - HUST
"""
from train import train
from utils.dataloader import FaceDataset
from torch.utils.data import DataLoader
from model.VGG16 import VGG16
from config import *

model = VGG16() #to compile the model
model.to(device) #to send the model for training on either cuda or cpu

epochs = 50
batch_size = 64
lr = 1e-3
milestones = [10, 15, 20, 25, 30, 35]
path_save = './snapshots/Model_VGG_PD_SL.pth'.format(epochs, batch_size, lr)

dataset_path = './data/train_PD_SL'
trainsize = 224

dataset = FaceDataset(dataset_path, trainsize, type_loader='train_loader')
train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True)

dataset = FaceDataset(dataset_path, trainsize, type_loader='val_loader')
val_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True)

train(train_loader, val_loader, model, epochs, batch_size, lr, path_save, milestones)