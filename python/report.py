#!usr/bin/env python

###########
# Imports #
###########

import collections
import glob
import numpy as np
import os
import pandas as pd
import spacy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchtext as tt

from nltk.tree import Tree
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


##########
# Global #
##########

aliases = tt.vocab.pretrained_aliases


###########
# Classes #
###########

class SentimentNet(nn.Module):
	def __init__(
		self,
		embedding_shape, # tuple[int, int]
		output_size: int,
		pad_idx: int = 1,
		net: str = 'RNN',
		hidden_size: int = 256,
		num_layers: int = 1,
		dropout: float = 0,
		bidirectional: bool = False,
		attention: bool = False
	):
		super().__init__()

		self.embedding = nn.Embedding(
			embedding_shape[0],
			embedding_shape[1],
			padding_idx=pad_idx
		)

		if net == 'LSTM':
			self.rec = nn.LSTM
		elif net == 'GRU':
			self.rec = nn.GRU
		else:
			self.rec = nn.RNN

		self.rec = self.rec(
			input_size=embedding_shape[1],
			hidden_size=hidden_size,
			num_layers=num_layers,
			dropout=dropout,
			bidirectional=bidirectional,
			batch_first=True
		)

		self.attention = attention
		self.num_directions = 2 if bidirectional else 1

		self.lin = nn.Linear(
			hidden_size * self.num_directions,
			output_size
		)

		self.scale = float(np.sqrt(self.lin.in_features))

		self.softmax = nn.Softmax(dim=-1)
		self.softplus = nn.Softplus()

	def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
		# Embedding
		vector = self.embedding(input)
		vector = nn.utils.rnn.pack_padded_sequence(
			vector,
			lengths=lengths,
			batch_first=True
		)

		# Encoder
		outputs, hidden = self.rec(vector)

		if type(hidden) is tuple: # LSTM
			hidden = hidden[0]

		hidden = hidden[-self.num_directions:].transpose(0, 1).flatten(1) # batch_first

		# Attention
		if self.attention:
			outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
				outputs,
				batch_first=True
			)

			energy = torch.bmm(hidden.unsqueeze(1), outputs.transpose(1, 2))
			energy = self.softmax(energy / self.scale)

			hidden = torch.bmm(energy, outputs).squeeze(1)

		# Decoder
		logits = self.lin(hidden)

		logits = torch.cat([
			logits[:, :1],
			self.softplus(logits[:, 1:])
		], dim=1).cumsum(dim=1)

		return logits

	@staticmethod
	def prediction(output: torch.Tensor) -> torch.Tensor:
		output = torch.cat([
			torch.zeros((output.size(0), 1), device=output.device),
			torch.sigmoid(output),
			torch.ones((output.size(0), 1), device=output.device)
		], dim=1)

		return output[:, 1:] - output[:, :-1]


class CLWithLogitsLoss(nn.Module):
	r"""Cumulative link with logits loss

	References:
		[1] On the consistency of ordinal regression methods
		(Pedregosa et al., 2017)
		https://dl.acm.org/doi/abs/10.5555/3122009.3153011

	Note:
		For (N, 1) inputs, CLWithLogitsLoss is equivalent to BCEWithLogitsLoss.
	"""

	def __init__(self, reduction: str = 'mean'):
		super().__init__()

		self.reduction = reduction

	def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		exps = input.exp()
		logs = (1. + exps).log()

		neg_likelihoods = torch.cat([
			logs[:, :1] - input[:, :1], # -log(p(0 < x < 1))
			logs[:, 1:] + logs[:, :-1] - (exps[:, 1:] - exps[:, :-1]).log(), # -log(p(i-1 < x < i))
			logs[:, -1:] # -log(p(k-1 < x < k))
		], dim=1)

		loss = torch.gather(neg_likelihoods, 1, target.unsqueeze(1)).squeeze(1)

		if self.reduction == 'mean':
			loss = loss.mean()
		elif self.reduction == 'sum':
			loss = loss.sum()

		return loss


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
		tokenizer # callable[str] -> iterable[str]
	):
		super().__init__()

		self.data = data
		self.vocab = vocab
		self.tokenizer = tokenizer

	def __len__(self) -> int:
		return len(self.data)

	def __getitem__(self, idx): # -> tuple[torch.Tensor, int]
		target, text = self.data[idx]

		return (
			torch.tensor([self.vocab[t] for t in self.tokenizer(text)]),
			target
		)


class Collator:
	def __init__(self, pad_idx: int, batch_first: bool = True):
		self.pad_idx = pad_idx
		self.batch_first = batch_first

	def __call__(self, batch: list):
		texts, targets = zip(*batch)

		lengths = torch.tensor([len(x) for x in texts])

		order = torch.argsort(lengths, descending=True)

		lengths = lengths[order]
		texts = nn.utils.rnn.pad_sequence(
			texts,
			batch_first=self.batch_first,
			padding_value=self.pad_idx
		)[order]
		targets = torch.tensor(targets)[order]

		return texts, lengths, targets


############
# Function #
############

def IMDB(root: str = '.data'): # -> tuple[list[str], dict[str, list[tuple[int, str]]]]
	# Download
	path = tt.datasets.IMDB.download(root)

	# Load
	splits = {'train': [], 'test': []}
	labels = ['neg', 'pos']

	for split in splits:
		for i, label in enumerate(labels):
			for file in glob.glob(os.path.join(path, split, label, '*.txt')):
				with open(file, encoding='utf8') as f:
					splits[split].append((i, f.read()))

	return labels, splits


def SST(root: str = '.data'):
	# Download
	path = tt.datasets.SST.download(root)

	# Load
	splits = {'train': [], 'test': []}
	labels = ['very-negative', 'negative', 'neutral', 'positive', 'very-positive']

	for split in splits:
		with open(os.path.join(path, split + '.txt')) as f:
			for line in f:
				tree = Tree.fromstring(line)
				splits[split].append((int(tree.label()), ' '.join(tree.leaves())))

	return labels, splits


def freqs(
	data, # list[tuple[int, str]]
	tokenizer # callable[str] -> iterable[str]
) -> collections.Counter:
	counter = collections.Counter()

	for _, text in tqdm(data):
		counter.update(tokenizer(text))

	return counter


def eval(
	model: nn.Module,
	loader: data.DataLoader,
	device: torch.device,
	criterion: nn.Module = None,
	optimizer: optim.Optimizer = None
): # -> tuple[list, list, list]
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

	# Vocabulary
	tokenizer = Tokenizer()

	if dataset == 'IMDB':
		labels, splits = IMDB()
	else: # dataset == 'SST'
		labels, splits = SST()

	traindata, testdata = splits['train'], splits['test']

	vocab = tt.vocab.Vocab(
		counter=freqs(traindata, tokenizer),
		max_size=vocab_size
	)

	# Embedding
	pretrained = aliases[embedding]()
	vocab.set_vectors(
		pretrained.stoi,
		pretrained.vectors,
		dim=pretrained.vectors.size(1)
	)

	# Datasets
	trainset = SentimentDataset(traindata, vocab, tokenizer)
	testset = SentimentDataset(testdata, vocab, tokenizer)

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
			'time': (stop - start) / len(trainloader),
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
			'time': (stop - start) / len(testloader),
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

	parser = argparse.ArgumentParser(description='Train Sentiment Analysis')

	parser.add_argument('-o', '--output', default='report.csv', help='output file')

	parser.add_argument('-dataset', default='IMDB', choices=['IMDB', 'SST'], help='dataset')
	parser.add_argument('-vsize', type=int, default=25000, help='vocab size')
	parser.add_argument('-embedding', default='glove.6B.100d', choices=list(aliases.keys()), help='embedding alias')

	parser.add_argument('-net', default='RNN', choices=['RNN', 'LSTM', 'GRU'], help='recurrent neural network type')
	parser.add_argument('-hidden', type=int, default=256, help='hidden memory size')
	parser.add_argument('-layers', type=int, default=1, help='number of layers in RNN')
	parser.add_argument('-dropout', type=float, default=0, help='dropout in RNN')
	parser.add_argument('-bidirectional', default=False, action='store_true', help='bidirectional RNN')
	parser.add_argument('-attention', default=False, action='store_true', help='attention in RNN')

	parser.add_argument('-bsize', type=int, default=64, help='batch size')
	parser.add_argument('-epochs', type=int, default=5, help='number of epochs')
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
