import nni
from dataloader import SupervisedData
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from model import resnest

def parse_args():
	parser = argparse.ArgumentParser(
		description = 'Setup the training settings.',
		formatter_class = argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--train-img-path', default = './dataset/data/p1_data/train_50/',
						help = "The directory that contains training img files")

	parser.add_argument('--valid-img-path', default = './dataset/data/p1_data/val_50/',
						help = "The directory that contains validation img files")

	args = parser.parse_args()
	return args

def train(dataloader, model, device, optimizer, criterion, nni_params, epoch, epochs):
	model.train()
	loss_data = 0
	for img, label in tqdm(dataloader, ncols = 90, desc = '[Train] {:d}/{:d}'.format(epoch, epochs)):
		img, label = img.to(device), label.to(device)
		optimizer.zero_grad()
		pred = model(img)
		loss = criterion(pred, label)
		loss.backward()
		loss_data += loss.item()
		optimizer.step()
	print('classify loss: {}'.format(loss_data / len(dataloader)))

def valid(dataloader, model, device, optimizer, criterion, nni_params, epoch, epochs):
	model.eval()
	loss_data = 0
	hit = 0
	with torch.no_grad():
		for img, label in tqdm(dataloader, ncols = 90, desc = '[Valid] {:d}/{:d}'.format(epoch, epochs)):
			img, label = img.to(device), label.to(device)
			pred = model(img)
			loss = criterion(pred, label)
			loss_data += loss.item()
			hit += sum(torch.argmax(pred, dim = 1) == label)
		print('classify loss: ', loss_data / len(dataloader))
		print('Accuracy: {:.2f}%'.format(hit.true_divide(len(dataloader) * nni_params["batch_size"]) * 100))
		print()
	accuracy = hit.true_divide(len(dataloader) * nni_params["batch_size"])
	return accuracy

def main(nni_params):
	args = parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# load data
	train_dataset = SupervisedData(args.train_img_path)
	valid_dataset = SupervisedData(args.valid_img_path)
	train_dataloader = DataLoader(train_dataset, batch_size = nni_params['batch_size'], shuffle = True, num_workers = 1, pin_memory = True)
	valid_dataloader = DataLoader(valid_dataset, batch_size = nni_params['batch_size'], shuffle = False, num_workers = 1, pin_memory = True)

	# build model
	model = resnest.resnest50(num_classes = 50).to(device).float()
	optimizer = torch.optim.Adam(model.parameters(), lr = nni_params['lr'], betas = (0.5, 0.9))
	criterion = nn.CrossEntropyLoss().to(device).float()

	# train
	for epoch in range(nni_params['epochs']):
		train(train_dataloader, model, device, optimizer, criterion, nni_params, epoch, nni_params['epochs'])
		acc = valid(valid_dataloader, model, device, optimizer, criterion, nni_params, epoch, nni_params['epochs'])
		nni.report_intermediate_result(acc)
	nni.report_final_result(acc)

if __name__ == '__main__':
	nni_params = nni.get_next_parameter()
	main(nni_params)