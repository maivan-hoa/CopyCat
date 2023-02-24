# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:34 2023

@author: Mai Van Hoa - HUST
"""
from model.VGG16 import VGG16
from config import *
import torch
from utils.dataloader import FaceDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict

CLASSES = {v:k for k,v in CLASSES.items()}

def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def inference(model, path_model, path_image, trainsize=224):
	if str(device) == 'cpu':
		model.load_state_dict(torch.load(path_model, map_location='cpu'))
	else:
		model.load_state_dict(torch.load(path_model))

	model.to(device)

	image = rgb_loader(path_image)

	img_transform = transforms.Compose(
    [   
        transforms.Resize((trainsize, trainsize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])
    ])

	image = img_transform(image).unsqueeze(0).to(device)

	model.eval()
	with torch.no_grad():
		output = model(image)
		output = torch.softmax(output, dim=1)
		prob, index = torch.max(output, dim=1)

		prob = prob[0].item()
		label = CLASSES[index[0].item()]

	print('Predicted: ', label)
	print('Probability: ', round(prob, 3))


def test(model, path_to_model, dataset_path, trainsize=224, batch_size=2):
	if str(device) == 'cpu':
	    model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
	else:
	    model.load_state_dict(torch.load(path_to_model))

	model.to(device)
	   
	dataset = FaceDataset(dataset_path, trainsize, type_loader='test_loader')
	test_loader = DataLoader(dataset=dataset,
	                          batch_size=batch_size,
	                          shuffle=False)

	model.eval()
	predicted_true = 0
	total_sample = 0
	arr_prob = []
	arr_label_pred = []
	arr_ground_truth = []

	with torch.no_grad():
	    for X, y in test_loader:
	        X = X.to(device)
	        y = y.to(device)
	        
	        output = model(X)
	        output = torch.softmax(output, dim=1)
	        prob, index = torch.max(output, dim=1)
	        
	        arr_prob.extend([round(p.item(), 2) for p in prob])
	        arr_label_pred.extend([CLASSES[i.item()] for i in index])
	        arr_ground_truth.extend([CLASSES[j.item()] for j in y])

	        predicted_true += sum(index == y)
	        total_sample += len(y)

	accuracy = predicted_true / total_sample

	sum_prob = defaultdict(int)
	num_image = defaultdict(int)
	for i in range(len(arr_prob)):
		print('Ground truth: {} | Label predict: {} | Probability: {}'.format(arr_ground_truth[i], arr_label_pred[i], arr_prob[i]))
		sum_prob[arr_label_pred[i]] += arr_prob[i]
		num_image[arr_label_pred[i]] += 1

	print()
	print('Average Probability: ')
	for k in sum_prob:
		print('{}: {}'.format(k, round(sum_prob[k]/num_image[k], 2))) 

	print()
	print('Predicted true: ', predicted_true.item())
	print('Total sample: ', total_sample)
	print('Accuracy in test: {:.2f}%'.format(accuracy*100))
    
    
if __name__ == '__main__':
	# path_image = './data/test_OD/Thang/20230203_085434.jpg'
	path_image = './data/adversarial_example/test.png'
	path_model = './snapshots/Model_VGG16_OD.pth'
	# path_model = './snapshots/Model_VGG16_PD_SL.pth'
	model = VGG16(num_classes=5)
	# dataset_path = './data/train_OD'
	trainsize = 224

	inference(model, path_model, path_image, trainsize=224)

	# model = VGG16(num_classes=5)
	# path_save = './snapshots/Model_VGG16_OD.pth'
	# dataset_path = './data/test_OD'

	# test(model, path_save, dataset_path, trainsize=224, batch_size=2)