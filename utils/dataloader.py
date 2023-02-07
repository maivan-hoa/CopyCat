# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:03:43 2023

@author: Mai Van Hoa - HUST
"""
from torch.utils.data import Dataset, DataLoader
import glob
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
from config import *

class FaceDataset(Dataset):
    def __init__(self, image_root, trainsize, type_loader='train_loader'):
        self.trainsize = trainsize
        self.images = glob.glob(image_root + '/*/*')
        
        # random.seed(30031999)
        # self.images = np.array(self.images)
        self.labels = [int(CLASSES[i.split('/')[-2]]) for i in self.images]

        if type_loader != 'test_loader':
            X_train, X_val, y_train, y_val = train_test_split(self.images, self.labels,
                                                              stratify=self.labels, 
                                                              test_size=0.2,
                                                              random_state=1)
        if type_loader == 'train_loader':
            self.images = X_train
            self.labels = y_train
            print('Train Dataset: ', len(self.images))
        elif type_loader == 'val_loader':
            self.images = X_val
            self.labels = y_val
            print('Validation Dataset: ', len(self.images))
        elif type_loader == 'test_loader':
            self.images = self.images
            self.labels = self.labels
            print('Test Dataset: ', len(self.images))

        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        label = self.labels[index]

        # image = np.array(image).astype('float32')
        # image = (image - 127.5) / 128

        # img_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize((self.trainsize, self.trainsize))
        # ])
        img_transform = transforms.Compose(
        [   
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        ])

        image = img_transform(image)

        return image, label

    def __len__(self):
        return self.size

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')