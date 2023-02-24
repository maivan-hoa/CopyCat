from model.VGG16 import VGG16
from model.VGG19 import VGG19
from config import *
import torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import numpy as np
from utils.utils import *


CLASSES = {v:k for k,v in CLASSES.items()}
TARGET_LABEL = 4 # chỉ số lớp muốn model nhận diện ra
ORIGIN_LABEL = 1 # chỉ số thực của ảnh đầu vào
EPSILON = 200/255 # lượng maximum nhiễu được phép thêm vào ảnh, nhằm kiểm soát sự khác biệt giữa ảnh gốc và ảnh đã thêm nhiễu

def tensor_to_image(tensor):
	x = tensor.numpy().transpose(1, 2, 0) * 255
	x = np.clip(x, 0, 255)
	return Image.fromarray(x.astype(np.uint8))


# Chèn nhiễu và ghi ảnh ra file
def create_adversarial_example(img_org, path_noise):
	noise = torch.Tensor(np.load(path_noise))

	img_noise = img_org + noise
	img_noise = get_inverse_transformation(img_noise)

	img_noise = tensor_to_image(img_noise)

	img_noise.save('./data/adversarial_example/test.png')


# Chèn nhiễu vào đồng thời với quá trình inference
def test_adversarial_example(model_test, img_org, path_noise):
	print('###############################')
	print('Kết quả với mô hình được kiểm tra khi thêm nhiễu')

	print('	Kết quả khi không có nhiễu: ')
	img = img_org.to(device)[None]
	model_test.to(device).eval()
	output = model_test(img)
	output = torch.softmax(output, 1)
	val, idx = torch.max(output, 1)
	print('	class: {}: {} | probability: {:.3f}'.format(int(idx), CLASSES[int(idx)], float(val)))


	print('	Kết quả khi thêm nhiễu: ')
	noise = torch.Tensor(np.load(path_noise))
	img_noise = (img_org + noise).to(device)[None]
	model_test.to(device).eval()
	output = model_test(img_noise)
	output = torch.softmax(output, 1)
	val, idx = torch.max(output, 1)
	print('	class: {}: {} | probability: {:.3f}'.format(int(idx), CLASSES[int(idx)], float(val)))


# Tạo nhiễu nhằm tấn công dạng Targeted adversarial attacks
def create_adversarial_noise_targeted(model_org, image, path_noise, img_size=224, epochs=200, lr=0.2):
	noise = torch.randn((1, 3, img_size, img_size), requires_grad=True, device=device)

	model_org.to(device).eval()
	image = image.to(device)
	label = torch.LongTensor([TARGET_LABEL]).to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD([noise], lr=lr, momentum=0.9)

	for i in range(epochs):
		noise.data.clamp_(-EPSILON, EPSILON)
		optimizer.zero_grad()

		output = model_org(image + noise)
		# loss = criterion(output, label)


		label_one_hot = torch.eye(output.shape[1], device=device)[label.long()]
		target = torch.sum(output * label_one_hot, 1)
		other, _ = torch.max(output * (1 - label_one_hot), 1)

		# print('Real: ', real.item())
		# print('Other: ', other.item())

		loss = torch.mean(other - target)


		loss.backward()
		optimizer.step()

		output = torch.softmax(output, 1)
		val, idx = torch.max(output, 1)
		print('Epochs: {:3d} | Loss: {:.3f} | class: {}: {} | probability: {:.3f}'.format(i+1, loss, int(idx), CLASSES[int(idx)], float(val)))

	np.save(path_noise, noise.squeeze().detach().cpu().numpy())

	print()
	print('Kết quả nhận diện của mô hình bị tấn công tạo nhiễu: ')
	img_noise = (image + noise).to(device)
	output = model_org(img_noise)
	output = torch.softmax(output, 1)
	val, idx = torch.max(output, 1)
	print('class: {}: {} | probability: {:.3f}'.format(int(idx), CLASSES[int(idx)], float(val)))


# Tạo nhiễu nhằm tấn công dạng Untargeted adversarial attacks
def create_adversarial_noise_untargeted(model_org, image, path_noise, img_size=224, epochs=200, lr=0.5):
	noise = torch.randn((1, 3, img_size, img_size), requires_grad=True, device=device)

	model_org.to(device).eval()
	image = image.to(device)
	label = torch.LongTensor([ORIGIN_LABEL]).to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD([noise], lr=lr, momentum=0.9)

	for i in range(epochs):
		noise.data.clamp_(-EPSILON, EPSILON)
		optimizer.zero_grad()

		output = model_org(image + noise)
		# output = torch.softmax(output, 1)

		label_one_hot = torch.eye(output.shape[1], device=device)[label.long()]
		real = torch.sum(output * label_one_hot, 1)
		# other = torch.sum(output * (1 - label_one_hot), 1)
		other, _ = torch.max(output * (1 - label_one_hot), 1)

		# print('Real: ', real.item())
		# print('Other: ', other.item())

		loss = torch.mean(real - other)

		# loss = criterion(output, label)
		loss.backward()
		optimizer.step()

		output = torch.softmax(output, 1)
		val, idx = torch.max(output, 1)
		print('Epochs: {:3d} | Loss: {:.3f} | class: {}: {} | probability: {:.3f}'.format(i+1, loss, int(idx), CLASSES[int(idx)], float(val)))

	np.save(path_noise, noise.squeeze().detach().cpu().numpy())


if __name__ == '__main__':

	path_image = './data/test_OD/Thang/20230201_093451.jpg'
	path_noise = './snapshots/noise/adversarial_noise.npy'

	image = rgb_loader(path_image)
	image = get_image_transformation(image)

	path_to_model_copy = './snapshots/Model_VGG16_PD_SL.pth'
	model_copy = VGG16(num_classes=5)
	model_copy = load_model(model_copy, path_to_model_copy)
	create_adversarial_noise_targeted(model_copy, image, path_noise)


	path_to_model_org = './snapshots/Model_VGG16_OD.pth'
	model_org = VGG16(num_classes=5)
	model_org = load_model(model_org, path_to_model_org)
	test_adversarial_example(model_org, image, path_noise)


	# path_image = './data/test_OD/Thang/20230203_085434.jpg'
	image = rgb_loader(path_image)
	image = get_image_transformation(image)
	create_adversarial_example(image, path_noise)





































