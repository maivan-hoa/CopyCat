import os
from config import *
from model.VGG16 import VGG16
from utils.dataloader import FaceDataset_SL
import io
from torch.utils.data import DataLoader

def inference(model, dataloader, suffix='OD'):
	model.eval()
	m = []
	v = []

	number_image = 0
	with torch.no_grad():
		for image, path_image in dataloader:
			image = image.to(device)
			output = model(image)

			output = torch.softmax(output, dim=1).cpu().detach().numpy()

			label = [str(CLASSES[i.split('/')[-2]]) + '_' + suffix for i in path_image]

			m.extend(label)
			v.extend(output)

			number_image += len(image)
			print('number image process: ', number_image)

	return m, v


if __name__ == '__main__':
	meta = []
	vecs = []

	trainsize = 224
	batch_size = 32
	dataset_path = './data/train_OD_aug'

	model = VGG16(num_classes=5)
	path_to_model = './snapshots/Model_VGG16_OD.pth'

	if str(device) == 'cpu':
	    model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
	else:
	    model.load_state_dict(torch.load(path_to_model))

	model.to(device)

	dataset = FaceDataset_SL(dataset_path, trainsize)
	OD_loader = DataLoader(dataset=dataset,
	                          batch_size=batch_size,
	                          shuffle=False)

	m, v = inference(model, OD_loader, suffix='OD')
	meta.extend(m)
	vecs.extend(v)

	out_v = io.open('./snapshots/embedding/vecs.tsv', 'w', encoding='utf-8')
	out_m = io.open('./snapshots/embedding/meta.tsv', 'w', encoding='utf-8')

	for idx in range(len(meta)):
		label = meta[idx]
		vector = vecs[idx]

		out_m.write(label + '\n') # Lưu nhãn của mỗi điểm dữ liệu
		out_v.write('\t'.join([str(x) for x in vector]) + '\n') # Lưu tọa độ vector của mỗi điểm dữ liệu
	  
	out_m.close()
	out_v.close()






























