#!usr/bin/env python

###########
# Imports #
###########

import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchtext as tt

from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

import datasets

from model import SentimentNet, CLWithLogitsLoss


############
# Function #
############

def eval(
	model: nn.Module,
	loader: data.DataLoader,
	device: torch.device,
	criterion: nn.Module = None,
	optimizer: optim.Optimizer = None
): # -> tuple[list[int], list[int], list[float]]
	r"""Epoch evaluation

	Args:
		model: model
		loader: data loader
		device: CUDA device
		criterion: loss function
		optimizer: model optimizer

	Returns:
		lists of predictions, labels and losses
	"""

	predictions = []
	labels = []
	losses = []

	if optimizer is None:
		model.eval()
	else:
		model.train()

	for inputs, lengths, targets in tqdm(loader):
		outputs = model(inputs.to(device), lengths)

		if criterion is not None:
			loss = criterion(outputs, targets.to(device))
			losses.append(loss.tolist())

		if optimizer is not None:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		predictions.extend(model.prediction(outputs).argmax(dim=1).tolist())
		labels.extend(targets.tolist())

	return predictions, labels, losses


########
# Main #
########

def main(
	output_file: str = 'report.csv',
	dataset: str = 'IMDB',
	vocab_size: int = 25000,
	embedding: str = 'glove.6B.100d',
	net: str = 'RNN',
	hidden_size: int = 256,
	num_layers: int = 1,
	dropout: float = 0,
	bidirectional: bool = False,
	attention: bool = False,
	batch_size: int = 64,
	epochs: int = 10,
	learning_rate: float = 1e-3,
	weight_decay: float = 0.,
	step_size: int = 5,
	gamma: float = 1e-1,
	workers: int = 4,
	seed: int = 31415
):
	torch.manual_seed(seed)

	# Device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Vocabulary
	tokenizer = datasets.Tokenizer()
	frequencer = datasets.Frequencer(tokenizer)

	if dataset == 'IMDB':
		labels, splits = datasets.IMDB()
	else: # dataset == 'SST'
		labels, splits = datasets.SST()

	traindata, testdata = splits['train'], splits['test']

	vocab = tt.vocab.Vocab(
		counter=frequencer([text for _, text in traindata]),
		max_size=vocab_size
	)

	# Embedding
	pretrained = datasets.load_embedding(embedding)
	vocab.set_vectors(
		pretrained.stoi,
		pretrained.vectors,
		dim=pretrained.dim
	)
	del pretrained

	# Datasets
	trainset = datasets.SentimentDataset(traindata, vocab, tokenizer)
	testset = datasets.SentimentDataset(testdata, vocab, tokenizer)

	# DataLoaders
	collator = datasets.SeqCollator(pad_idx=vocab.stoi['<pad>'])

	trainloader = data.DataLoader(
		trainset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=workers,
		pin_memory=True,
		collate_fn=collator
	)

	testloader = data.DataLoader(
		testset,
		batch_size=batch_size,
		num_workers=workers,
		pin_memory=True,
		collate_fn=collator
	)

	# Model
	model = SentimentNet(
		embedding_shape=vocab.vectors.shape,
		output_size=len(labels) - 1,
		pad_idx=vocab.stoi['<pad>'],
		net=net,
		hidden_size=hidden_size,
		num_layers=num_layers,
		dropout=dropout,
		bidirectional=bidirectional,
		attention=attention
	)
	model.embedding.weight.data = vocab.vectors.clone()
	model.embedding.weight.requires_grad = False
	model.to(device)

	# Criterion
	criterion = CLWithLogitsLoss()
	criterion.to(device)

	# Optimizer
	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

	# Report
	stats = []

	for epoch in range(epochs):
		## Train
		start = time.time()
		predictions, targets, losses = eval(
			model,
			trainloader,
			device,
			criterion,
			optimizer
		)
		stop = time.time()

		losses = np.array(losses)
		accuracy = accuracy_score(targets, predictions)
		report = classification_report(targets, predictions, zero_division=0, output_dict=True)['weighted avg']

		stats.append({
			'dataset': dataset,
			'vocab_size': vocab_size,
			'embedding': embedding,
			'net': net,
			'hidden_size': hidden_size,
			'num_layers': num_layers,
			'dropout': dropout,
			'bidirectional': int(bidirectional),
			'attention': int(attention),
			'type': 'train',
			'epoch': epoch,
			'time': stop - start,
			'speed': len(testloader) / (stop - start),
			'loss_mean': losses.mean(),
			'loss_std': losses.std(),
			'precision': report['precision'],
			'recall': report['recall'],
			'f1-score': report['f1-score'],
			'accuracy': accuracy
		})

		scheduler.step()

		## Test
		start = time.time()
		predictions, targets, losses = eval(
			model,
			testloader,
			device,
			criterion
		)
		stop = time.time()

		losses = np.array(losses)
		accuracy = accuracy_score(targets, predictions)
		report = classification_report(targets, predictions, zero_division=0, output_dict=True)['weighted avg']

		stats.append({
			'dataset': dataset,
			'vocab_size': vocab_size,
			'embedding': embedding,
			'net': net,
			'hidden_size': hidden_size,
			'num_layers': num_layers,
			'dropout': dropout,
			'bidirectional': int(bidirectional),
			'attention': int(attention),
			'type': 'test',
			'epoch': epoch,
			'time': stop - start,
			'speed': len(testloader) / (stop - start),
			'loss_mean': losses.mean(),
			'loss_std': losses.std(),
			'precision': report['precision'],
			'recall': report['recall'],
			'f1-score': report['f1-score'],
			'accuracy': accuracy
		})

	# Export
	pd.DataFrame(stats).to_csv(output_file, mode='a', index=False)


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Perform training of various network architectures for sentiment analysis, and report statistics.')

	parser.add_argument('-o', '--output', default='report.csv', help='output file')

	parser.add_argument('-dataset', default='IMDB', choices=['IMDB', 'SST'], help='dataset')
	parser.add_argument('-vsize', type=int, default=25000, help='vocab size')
	parser.add_argument('-embedding', default='glove.6B.100d', choices=list(datasets.aliases.keys()), help='embedding alias')

	parser.add_argument('-net', default='RNN', choices=['RNN', 'LSTM', 'GRU'], help='recurrent neural network type')
	parser.add_argument('-hidden', type=int, default=256, help='hidden memory size')
	parser.add_argument('-layers', type=int, default=1, help='number of layers in RNN')
	parser.add_argument('-dropout', type=float, default=0, help='dropout in RNN')
	parser.add_argument('-bidirectional', default=False, action='store_true', help='bidirectional RNN')
	parser.add_argument('-attention', default=False, action='store_true', help='attention in RNN')

	parser.add_argument('-bsize', type=int, default=64, help='batch size')
	parser.add_argument('-epochs', type=int, default=10, help='number of epochs')
	parser.add_argument('-lrate', type=float, default=1e-3, help='learning rate')
	parser.add_argument('-wdecay', type=float, default=0., help='weight decay')
	parser.add_argument('-step-size', type=int, default=5, help='scheduler step size')
	parser.add_argument('-gamma', type=float, default=1e-1, help='scheduler gamma')
	parser.add_argument('-workers', type=int, default=4, help='number of workers')
	parser.add_argument('-seed', type=int, default=31415, help='seed')

	args = parser.parse_args()

	main(
		output_file=args.output,
		dataset=args.dataset,
		vocab_size=args.vsize,
		embedding=args.embedding,
		net=args.net,
		hidden_size=args.hidden,
		num_layers=args.layers,
		dropout=args.dropout,
		bidirectional=args.bidirectional,
		attention=args.attention,
		batch_size=args.bsize,
		epochs=args.epochs,
		learning_rate=args.lrate,
		weight_decay=args.wdecay,
		step_size=args.step_size,
		gamma=args.gamma,
		workers=args.workers,
		seed=args.seed
	)
