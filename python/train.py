#!usr/bin/env python

###########
# Imports #
###########

import collections
import glob
import numpy as np
import os
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchtext as tt

from sklearn.metrics import confusion_matrix
from tqdm import tqdm


###########
# Classes #
###########

class SentimentNet(nn.Module):
	def __init__(
		self,
		vocab_size: int,
		embedding_size: int,
		hidden_size: int,
		output_size: int,
		pad_idx: int,
		num_layers: int = 1,
		dropout: float = 0,
		bidirectional: bool = False,
		recnet: str = 'RNN'
	):
		super().__init__()

		self.embedding = nn.Embedding(
			vocab_size,
			embedding_size,
			padding_idx=pad_idx
		)

		self.rec = (nn.RNN if recnet == 'RNN' else nn.LSTM)(
			input_size=embedding_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			dropout=dropout,
			bidirectional=bidirectional,
			batch_first=True
		)
		self.get = (lambda x: x[1]) if recnet == 'RNN' else (lambda x: x[1][0])

		self.lin = nn.Linear(
			hidden_size * num_layers * (2 if bidirectional else 1),
			output_size
		)

	def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
		x = self.embedding(input)
		x = nn.utils.rnn.pack_padded_sequence(
			x,
			lengths=lengths,
			batch_first=True
		)
		x = self.get(self.rec(x))

		x = self.lin(x.permute(1, 0, 2).flatten(1)).squeeze(1)

		return x


class Tokenizer:
	def __init__(self, lang: str = 'en_core_web_sm'):
		self.tokenizer = spacy.load(lang).tokenizer

	def __call__(self, text: str): # -> generator[str]
		for t in self.tokenizer(text):
			yield t.text.lower()


class SentimentDataset(data.Dataset):
	def __init__(
		self,
		data: list,
		vocab: tt.vocab.Vocab,
		tokenizer, # callable[str] -> iterable[str]
		positive: str = 'pos'
	):
		super().__init__()

		self.data = data
		self.vocab = vocab
		self.tokenizer = tokenizer
		self.positive = positive

	def __len__(self) -> int:
		return len(self.data)

	def __getitem__(self, idx): # -> tuple[torch.Tensor, int]
		text, label = self.data[idx]

		return (
			torch.tensor([self.vocab[t] for t in self.tokenizer(text)]),
			1. if label == self.positive else 0.
		)


class Collator:
	def __init__(self, pad_idx: int, batch_first: bool = True):
		self.pad_idx = pad_idx
		self.batch_first = batch_first

	def __call__(self, batch: list):
		texts, labels = zip(*batch)

		lengths = torch.tensor([len(x) for x in texts])

		order = torch.argsort(lengths, descending=True)

		lengths = lengths[order]
		texts = nn.utils.rnn.pad_sequence(
			texts,
			batch_first=self.batch_first,
			padding_value=self.pad_idx
		)[order]
		labels = torch.tensor(labels)[order]

		return texts, lengths, labels


############
# Function #
############

def IMDB(
	root: str = '.data',
	splits=['train', 'test'], # list[str]
	labels=['neg', 'pos'] # list[str]
): # -> list[list[tuple[str, str]]]
	# Download
	path = tt.datasets.IMDB.download(root)

	# Train
	data = []

	for split in splits:
		data.append([])

		for label in labels:
			for file in glob.glob(os.path.join(path, split, label, '*.txt')):
				with open(file, encoding='utf8') as f:
					data[-1].append((f.read(), label))

	return data


def freqs(
	data, # list[tuple[str, str]]
	tokenizer # callable[str] -> iterable[str]
) -> collections.Counter:
	counter = collections.Counter()

	for text, _ in tqdm(data):
		counter.update(tokenizer(text))

	return counter


def epoch(
	model: nn.Module,
	loader: data.DataLoader,
	device: torch.device,
	criterion: nn.Module = None,
	optimizer: optim.Optimizer = None,
	head: nn.Module = nn.Sigmoid()
): # -> tuple[list, list, list]
	classes = []
	predictions = []
	losses = []

	if optimizer is None:
		model.eval()
	else:
		model.train()

	for inputs, lengths, labels in tqdm(loader):
		outputs = model(inputs.to(device), lengths)

		if criterion is not None:
			loss = criterion(outputs, labels.to(device))
			losses.append(loss.tolist())

		if optimizer is not None:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		outputs = head(outputs).round()

		classes.extend(labels.tolist())
		predictions.extend(outputs.tolist())

	return classes, predictions, losses


########
# Main #
########

def main(
	vocab_size: int = 25000,
	embedding_size: int = 50,
	hidden_size: int = 256,
	num_layers: int = 1,
	dropout: float = 0,
	bidirectional: bool = False,
	recnet: str = 'RNN',
	batch_size: int = 64,
	epochs: int = 5,
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

	# Dataset
	tokenizer = Tokenizer()

	traindata, testdata = IMDB()

	vocab = tt.vocab.Vocab(
		counter=freqs(traindata, tokenizer),
		max_size=vocab_size
	)

	trainset = SentimentDataset(traindata, vocab, tokenizer)
	testset = SentimentDataset(testdata, vocab, tokenizer)

	# Embedding
	glove = tt.vocab.GloVe('6B', dim=embedding_size)
	vocab.set_vectors(glove.stoi, glove.vectors, dim=embedding_size)

	# DataLoaders
	collator = Collator(pad_idx=vocab.stoi['<pad>'])

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
		vocab_size=vocab_size,
		embedding_size=embedding_size,
		hidden_size=hidden_size,
		output_size=1,
		pad_idx=vocab.stoi['<pad>'],
		num_layers=num_layers,
		dropout=dropout,
		bidirectional=bidirectional,
		recnet=recnet
	)
	model.embedding.weight.data = vocab.vectors.clone()
	model.to(device)

	# Criterion
	criterion = nn.BCEWithLogitsLoss()
	criterion.to(device)

	# Optimizer
	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

	# Train
	for _ in range(epochs):
		## Train
		classes, predictions, losses = epoch(
			model,
			trainloader,
			device,
			criterion,
			optimizer
		)

		losses = np.array(losses)

		print('Training loss = {} +- {}'.format(losses.mean(), losses.std()))
		print('Training confusion matrix =\n', confusion_matrix(classes, predictions))

		scheduler.step()

	# Test
	classes, predictions, losses = epoch(
		model,
		testloader,
		device,
		criterion
	)

	losses = np.array(losses)

	print('Test loss = {} +- {}'.format(losses.mean(), losses.std()))
	print('Test confusion matrix =\n', confusion_matrix(classes, predictions))


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Train Sentiment Analysis')

	parser.add_argument('-vsize', type=int, default=25000, help='vocab size')
	parser.add_argument('-bsize', type=int, default=64, help='batch size')

	parser.add_argument('-hidden', type=int, default=256, help='hidden memory size')
	parser.add_argument('-layers', type=int, default=1, help='number of layers in RNN')
	parser.add_argument('-dropout', type=float, default=0, help='dropout in RNN')
	parser.add_argument('-bidirectional', default=False, action='store_true', help='bidirectional RNN')
	parser.add_argument('-recnet', default='RNN', choices=['RNN', 'LSTM'], help='recurrent neural network type')

	parser.add_argument('-epochs', type=int, default=5, help='number of epochs')
	parser.add_argument('-lrate', type=float, default=1e-3, help='learning rate')
	parser.add_argument('-wdecay', type=float, default=0., help='weight decay')
	parser.add_argument('-step-size', type=int, default=5, help='scheduler step size')
	parser.add_argument('-gamma', type=float, default=1e-1, help='scheduler gamma')
	parser.add_argument('-workers', type=int, default=4, help='number of workers')
	parser.add_argument('-seed', type=int, default=31415, help='seed')

	args = parser.parse_args()

	main(
		vocab_size=args.vsize,
		batch_size=args.bsize,
		hidden_size=args.hidden,
		num_layers=args.layers,
		dropout=args.dropout,
		bidirectional=args.bidirectional,
		recnet=args.recnet,
		epochs=args.epochs,
		learning_rate=args.lrate,
		weight_decay=args.wdecay,
		step_size=args.step_size,
		gamma=args.gamma,
		workers=args.workers,
		seed=args.seed
	)
