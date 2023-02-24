# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:53:34 2023

@author: Mai Van Hoa - HUST
"""

import torch
import torchvision.transforms.functional as TF
from PIL import Image
import glob
import random
import shutil
import os

root_folder = './data/ImageNet_SL'
number_image_balance = 60000


def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def augmentation(path_image):
    image = rgb_loader(path_image)
    
    if random.random() > 0.5:
        gamma = round(random.uniform(0.8, 1.6), 1)
        image_trans = TF.adjust_gamma(image, gamma=gamma)

    else:
        image_trans = TF.hflip(image)
    
    return image_trans

list_folder = os.listdir(root_folder)

for folder in list_folder:
    print('Processing folder: ', folder)
    list_image = glob.glob(root_folder + '/' + folder + '/*')
    n = len(list_image)
    
    if n == number_image_balance:
        continue
    elif n < number_image_balance:
        number_add = number_image_balance - n
        if number_add <= n:
            list_add = random.sample(list_image, number_add)
        else:
            list_add = [list_image[i] for i in (random.choice(range(n)) for j in range(number_add))]
        
        for i, path in enumerate(list_add):
            ext = os.path.splitext(path)[1]
            image_add = augmentation(path)
            image_add.save(path.replace(ext, '_'+ str(i) + ext))
            
    elif n > number_image_balance:
        number_remove = n - number_image_balance
        list_remove = random.sample(list_image, number_remove)
        
        for path in list_remove:
            os.remove(path)



































