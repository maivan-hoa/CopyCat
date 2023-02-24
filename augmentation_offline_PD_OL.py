import torch
import torchvision.transforms.functional as TF
from PIL import Image
import glob
import random
import shutil
import os
import matplotlib.pyplot as plt


folder_augmentation = './data/PD_OL'


def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


list_images = glob.glob(folder_augmentation+'/*/*')
list_images = [f for f in list_images if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]

for path_image in list_images:
    norm_path = os.path.normpath(path_image)
    label = norm_path.split(os.sep)[-2]
    name_image = norm_path.split(os.sep)[-1]

    path_save = folder_augmentation + '_aug/' + label
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    shutil.copy2(path_image, path_save)

    image = rgb_loader(path_image)

    gamma = round(random.uniform(0.4, 0.9), 1)
    image_trans_gamma = TF.adjust_gamma(image, gamma=gamma)
    image_trans_gamma.save(path_save + '/1_' + name_image)

    gamma = round(random.uniform(1.2, 2.0), 1)
    image_trans_gamma = TF.adjust_gamma(image, gamma=gamma)
    image_trans_gamma.save(path_save + '/2_' + name_image)

    image_trans_hflip = TF.hflip(image)
    image_trans_hflip.save(path_save + '/3_' + name_image)

    image_trans_center_crop = TF.center_crop(image, 100)
    image_trans_center_crop.save(path_save + '/4_' + name_image)



























