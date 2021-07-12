import random
import nni
from dataloader import SupervisedData
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from nni.retiarii import serialize
from base_model import BaseModel
import nni.retiarii.strategy as strategy
from nni.retiarii.evaluator import FunctionalEvaluator
import nni.retiarii.evaluator.pytorch.lightning as pl
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment

def parse_args():
	parser = argparse.ArgumentParser(
		description = 'Setup the training settings.',
		formatter_class = argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--train-img-path', default = '../nni/dataset/data/p1_data/train_50/',
						help = "The directory that contains training img files")

	parser.add_argument('--valid-img-path', default = '../nni/dataset/data/p1_data/val_50/',
						help = "The directory that contains validation img files")

	parser.add_argument('--epochs', default = 40, type = int,
						help = "The total training epochs")

	parser.add_argument('--batchsize', default = 20, type = int,
						help = "The training batchsize")

	parser.add_argument('--lr', default = 1e-3, type = float,
						help = "The training learning rate")

	args, unparsed = parser.parse_known_args()
	return args

def train(dataloader, model, device, optimizer, criterion, args, epoch, epochs):
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

def valid(dataloader, model, device, optimizer, criterion, args, epoch, epochs):
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
		print('Accuracy: {:.2f}%'.format(hit.true_divide(len(dataloader) * args.batch_size) * 100))
		print()
	accuracy = hit.true_divide(len(dataloader) * nni_params["batch_size"]).item()
	return accuracy

def main(model_):
	args = parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# load data
	train_dataset = SupervisedData(args.train_img_path)
	valid_dataset = SupervisedData(args.valid_img_path)
	train_dataloader = DataLoader(train_dataset, batch_size = args.batchsize, shuffle = True, num_workers = 1, pin_memory = True)
	valid_dataloader = DataLoader(valid_dataset, batch_size = args.batchsize, shuffle = False, num_workers = 1, pin_memory = True)

	# build model
	model = model_().to(device).float()
	optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas = (0.5, 0.9))
	criterion = nn.CrossEntropyLoss().to(device).float()

	# train
	for epoch in range(1, args.epochs + 1):
		train(train_dataloader, model, device, optimizer, criterion, args, epoch, args.epochs)
		acc = valid(valid_dataloader, model, device, optimizer, criterion, args, epoch, args.epochs)
		nni.report_intermediate_result(acc)
	nni.report_final_result(acc)

if __name__ == '__main__':
	# main(BaseModel)
	chosen_strategy = strategy.RegularizedEvolution()
	evaluator = FunctionalEvaluator(main)
	exp = RetiariiExperiment(serialize(BaseModel), evaluator, None, chosen_strategy)
	exp_config = RetiariiExeConfig('local')
	exp_config.experiment_name = 'NAS_searcher'
	exp_config.trial_concurrency = 1
	exp_config.max_trial_number = 50
	exp_config.experiment_working_directory = "/home/lab402-3090/Desktop/An/NAS-nni/NAS/NAS-experiment"
	exp_config.training_service.use_active_gpu = True
	exp.run(exp_config, 8081 + random.randint(0, 100))
	for model_code in exp.export_top_models(formatter='dict'):
		print(model_code)