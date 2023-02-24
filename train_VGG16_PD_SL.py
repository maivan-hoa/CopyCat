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
from testing import test

model = VGG16(num_classes=5) #to compile the model
model.to(device) #to send the model for training on either cuda or cpu

epochs = 15
batch_size = 32
lr = 1e-3
milestones = [10, 15, 20, 25, 30, 35]
path_save = './snapshots/Model_VGG16_PD_SL.pth'

dataset_path = './data/PD_SL'
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

# Testing
model = VGG16(num_classes=5)
# path_to_model = './snapshots/Model_VGG16_PD_SL.pth'
dataset_path = './data/test_OD'

test(model, path_save, dataset_path, trainsize=224, batch_size=2)