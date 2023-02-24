import torch
import torchvision.transforms.functional as TF
from PIL import Image
import glob
import random
import shutil
import os
import matplotlib.pyplot as plt


folder_augmentation = 'test'


def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


list_images = glob.glob('./data/'+folder_augmentation+'/*/*')

for path_image in list_images:
    norm_path = os.path.normpath(path_image)
    label = norm_path.split(os.sep)[-2]
    name_image = norm_path.split(os.sep)[-1]

    path_save = './data/' + folder_augmentation + '_aug/' + label
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    shutil.copy2(path_image, path_save)

    image = rgb_loader(path_image)
    gamma = round(random.uniform(0.8, 1.6), 1)
    if gamma != 1:
        image_trans_gamma = TF.adjust_gamma(image, gamma=gamma)
        image_trans_gamma.save(path_save + '/1_' + name_image)

    image_trans_hflip = TF.hflip(image)
    image_trans_hflip.save(path_save + '/2_' + name_image)




























