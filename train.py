# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:21:33 2023

@author: Mai Van Hoa - HUST
"""
import torch
from torch import optim
import torch.nn as nn
import time
from config import *
import matplotlib.pyplot as plt


def train(train_loader, val_loader, model, epochs, batch_size, lr, path_save, milestones=None):
    # For visual
    arr_train_acc = []
    arr_train_loss = []
    arr_val_acc = []
    arr_val_loss = []  
    
    # optimizer = optim.Adam(model.parameters(), lr= lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr)
    
    if milestones != None:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
      
    criterion = nn.CrossEntropyLoss() # đã bao gồm log softmax
    best_loss = 1e9
    
    for epoch in range(1, epochs+1):
        clock0 = time.time()
        model.train()
        
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
        
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
      
        scheduler.step()
        # adjust_lr(optimizer, curr_epoch=epoch-1, epochs=epochs, lr_init=lr)
          
        clock1 = time.time()
          
        train_acc, train_loss = eval(model, train_loader)
        val_acc, val_loss = eval(model, val_loader)
          
        arr_train_acc.append(train_acc.detach().cpu().numpy())
        arr_train_loss.append(train_loss.detach().cpu().numpy())
        arr_val_acc.append(val_acc.detach().cpu().numpy())
        arr_val_loss.append(val_loss.detach().cpu().numpy())
          
        print('Epoch {:3d}/{} | time: {:5.2f}s | Train acc: {:5.2f}%, train loss: {:7.3f} |  Val acc: {:5.2f}%, val loss: {:7.3f}'.
              format(epoch, epochs, clock1-clock0, train_acc*100, train_loss, val_acc*100, val_loss))
        
        # Save model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), path_save)
    
    plt.figure(figsize=(20, 8))
    plt.style.use("ggplot")
    epochs = [i for i in range(1, epochs+1)]
    plt.plot(epochs, arr_train_acc, label='train_acc')
    plt.plot(epochs, arr_train_loss, label='train_loss')
    plt.plot(epochs, arr_val_acc, label='val_acc')
    plt.plot(epochs, arr_val_loss, label='val_loss')
    
    plt.legend(loc='upper left')
    plt.title('train and valid accuracy per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('./snapshots/learning_curve_' + path_save.split('/')[-1].replace('pth', 'png'))


def eval(model, dataloader):
    model.eval()
    loss = 0
    predict_true = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
      
        with torch.no_grad():
            output = model(X)
        
        loss += criterion(output, y)
        predicted = torch.argmax(output, dim=1)
        predict_true += sum(predicted == y)
        total += X.shape[0]
  
    return predict_true/total, loss/len(dataloader)
