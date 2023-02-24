import torch
import torchvision.transforms.functional as TF
from PIL import Image
import glob
import random
import shutil
import os
import matplotlib.pyplot as plt


path_image = './data/NPL_SL/Minh/3_OIP-4afqUHyJVlQjd2nk7EYIiQHaEK.jpeg'


def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

image = rgb_loader(path_image)
print(image)