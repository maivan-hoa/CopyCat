# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:54:18 2023

@author: Mai Van Hoa - HUST
"""
import numpy as np
from config import *
from PIL import Image
import torchvision.transforms as transforms

def get_image_transformation(image, trainsize=224):
    img_transform = transforms.Compose(
    [   
        transforms.Resize((trainsize, trainsize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])
    ])


    return img_transform(image)
    

def get_inverse_transformation(image):
    img_transform = transforms.Normalize(
      mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],  # INVERSE normalize images, according to https://pytorch.org/docs/stable/torchvision/models.html
      std=[1/0.229, 1/0.224, 1/0.255]
    )

    return img_transform(image)


def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def load_model(model, path_to_model):
    if str(device) == 'cpu':
        model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(path_to_model))

    model.to(device)
    return model


def lr_warmup_cosine_decay(global_step, # step hiện tại
                           warmup_steps, # số step thực hiện warmup
                           hold = 0, # số step giữ lr bằng target_lr sau khi kết thúc giai đoạn warmup
                           total_steps=0, # tổng số step
                           start_lr=0.0, # lr khi bắt đầu warmup
                           target_lr=1e-3): # lr khởi tạo khi hết quá trình warmup

    # target_lr * progress of warmup (= 1 at the final warmup step)
    warmup_lr = target_lr * (global_step / warmup_steps)

    # Cosine decay = lr_min + 1/2 * (lr_max - lr_min) * (1 + cos(step_current*pi / total_steps)) --> cần dịch step ban đầu là 0
    learning_rate = 0.5 * target_lr * (1 + np.cos(np.pi * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = np.where(global_step > warmup_steps + hold,
                                 learning_rate, target_lr)
    
    learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate


def adjust_lr(optimizer, curr_epoch, epochs, lr_init):
    lr = lr_warmup_cosine_decay(global_step=curr_epoch, warmup_steps=5, hold=0, total_steps=epochs, start_lr=0.0, target_lr=lr_init)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr