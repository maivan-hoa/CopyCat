# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:34 2023

@author: Mai Van Hoa - HUST
"""
from model.VGG16 import VGG16
from config import *
import torch
from utils.dataloader import FaceDataset

path_to_model = './snapshots/Model_VGG_PD_SL.pth'
dataset_path = './data/test'
trainsize = 224
batch_size = 2
model = VGG16().to(device)


if str(device) == 'cpu':
    model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
else:
    model.load_state_dict(torch.load(path_to_model))
    
    
dataset = FaceDataset(dataset_path, trainsize, type_loader='test_loader')
test_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=False)

model.eval()
predicted_true = 0
total_sample = 0

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        y = y.to(device)
        
        output = model(X)
        # print(torch.softmax(output, dim=1))
        # print('label: ', y)
        predicted = torch.argmax(output, dim=1)
        # print('predicted: ', predicted)
        predicted_true += sum(predicted == y)
        total_sample += len(y)

accuracy = predicted_true / total_sample
print('Predicted true: ', predicted_true.item())
print('Total sample: ', total_sample)
print('Accuracy in test: {:.2f}%'.format(accuracy*100))
    
    
    