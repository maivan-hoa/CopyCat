# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:23:08 2023

@author: Mai Van Hoa - HUST
"""

from config import *
from model.VGG16 import VGG16
from utils.dataloader import FaceDataset_SL
from torch.utils.data import DataLoader
import shutil
import os
from collections import defaultdict

CLASSES = {v:k for k,v in CLASSES.items()}

path_to_model = './snapshots/Model_VGG16_OD.pth'
root_save = './data/ImageNet_SL'
dataset_path = './data/ImageNet_OL_aug'
trainsize = 224
batch_size = 32
threshold_confidence = 0

model = VGG16(num_classes=5)
if str(device) == 'cpu':
    model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
else:
    model.load_state_dict(torch.load(path_to_model))

model.to(device)

dataset = FaceDataset_SL(dataset_path, trainsize)
test_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=False)

statistics = defaultdict(int)
total_image = 0

model.eval()
with torch.no_grad():
  # j  = 0
  number_image_success = 0
  for image, path_image in test_loader:
    total_image += len(image)
    image = image.to(device)
    output = model(image)
    
    output = torch.softmax(output, dim=1)
    output, indice = torch.max(output, dim=1)

    output = [o.item() for o in output]
    indice = [i.item() for i in indice]

    label = [CLASSES[index] for index in indice]
    # j += len(image)
    print('Processing {} image'.format(total_image))
    for i, path in enumerate(path_image):
        if output[i] > threshold_confidence:
            number_image_success += 1
            statistics[label[i]] += 1

            path_save = os.path.join(root_save, label[i])
            if not os.path.exists(path_save):
                os.makedirs(path_save)
          
            shutil.copy2(path, path_save)

print()
print('Threshold filter: ', threshold_confidence)
print('Number image stolen label success: {} / {}'.format(number_image_success, total_image))
print()
for k in statistics:
  print('{} has {} image'.format(k, statistics[k]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    