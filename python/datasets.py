#!usr/bin/env python

###########
# Imports #
###########

import collections
import gensim.downloader
import glob
import os
import spacy
import torch
import torch.nn as nn
import torch.utils.data as data
import torchtext as tt

from functools import partial
from nltk.tree import Tree
from tqdm import tqdm


############
# Datasets #
############

def IMDB(root: str = '.data'): # -> tuple[list[str], dict[str, list[tuple[int, str]]]]
	r"""Load/download the IMDB dataset.

	Args:
		root: where to load/download the dataset

	Returns:
		list of labels, dataset splits

	References:
		[1] Learning Word Vectors for Sentiment Analysis
		(Maas et al., 2011)
		http://www.aclweb.org/anthology/P11-1015
	"""

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
	r"""Load/download the SST dataset.

	Args:
		root: where to load/download the dataset

	Returns:
		list of labels, dataset splits

	References:
		[1] Recursive deep models for semantic compositionality over a sentiment treebank
		(Socher et al., 2013)
		https://www.aclweb.org/anthology/D13-1170/
	"""

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


##############
# Embeddings #
##############

class Word2Vec(tt.vocab.Vectors):
	r"""Word2Vec embedding

	Args:
		key: embedding key in Word2Vec.table
		unk_init: intializer of unknown word vectors
		max_vectors: max number of pre-trained vectors loaded

	Note:
		Based on 'gensim.downloader' module.
	"""

	table = {
		'google': 'word2vec-google-news-300'
	}

	def __init__(
		self,
		key: str = 'google',
		unk_init=None, # callable -> torch.Tensor
		max_vectors: int = None
	):
		model = gensim.downloader.load(self.table[key])
		vectors = model.vectors
		tokens = model.vocab.keys()

		if max_vectors is None:
			max_vectors = len(vectors)

		self.vectors = torch.tensor(vectors[:max_vectors])
		self.dim = self.vectors.size(1)
		self.itos = []
		self.stoi = {}

		for i, token in zip(range(max_vectors), tokens):
			self.itos.append(token)
			self.stoi[token] = i

		self.unk_init = torch.Tensor.zero_ if unk_init is None else unk_init


aliases = tt.vocab.pretrained_aliases
aliases['word2vec.google.300d'] = partial(Word2Vec, key='google', max_vectors=int(2e5))

def load_embedding(embedding: str):
	return aliases[embedding]()


###########
# Classes #
###########

class SentimentDataset(data.Dataset):
	r"""Sentiment Analysis Dataset

	Args:
		data: list of (label, text) pairs
		vocab: mapping from word to index
		tokenizer: text tokenizer
	"""

	def __init__(
		self,
		data, # list[tuple[int, str]]
		vocab: tt.vocab.Vocab,
		tokenizer # callable[str] -> iterable[str]
	):
		super().__init__()

		self.data = data
		self.vocab = vocab
		self.tokenizer = tokenizer

	def __len__(self) -> int:
		return len(self.data)

	def __getitem__(self, idx: int): # -> tuple[torch.Tensor, int]
		target, text = self.data[idx]

		return (
			torch.tensor([self.vocab[t] for t in self.tokenizer(text)]),
			target
		)


class SeqCollator:
	r"""Sequence collator

	Args:
		pad_idx: index of the '<pad>' token in the embedding
		batch_first: whether the batch should (T, B) or (B, T)
	"""

	def __init__(self, pad_idx: int, batch_first: bool = True):
		self.pad_idx = pad_idx
		self.batch_first = batch_first

	def __call__(self, batch: list): # -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
		r"""Collate a batch of (label/target, seq) pairs together.

		Returns:
			sequences tensor (B, max(T)), lengths tensor (B,) and targets tensors (B,)

		Note:
			Since, sequences have different lengths, they have to be padded
			to the longest length max(T).
		"""

		seqs, targets = zip(*batch)

		lengths = torch.tensor([len(x) for x in seqs])

		order = torch.argsort(lengths, descending=True)

		lengths = lengths[order]
		seqs = nn.utils.rnn.pad_sequence(
			seqs,
			batch_first=self.batch_first,
			padding_value=self.pad_idx
		)[order]
		targets = torch.tensor(targets)[order]

		return seqs, lengths, targets


class Tokenizer:
	r"""Spacy text tokenizer wrapper.

	Args:
		lang: language

	Note:
		One should execute
			python -m spacy download en
		beforehand.
	"""

	def __init__(self, lang: str = 'en_core_web_sm'):
		self.tokenizer = spacy.load(lang).tokenizer

	def __call__(self, text: str): # -> generator[str]
		for t in self.tokenizer(text):
			yield t.text.lower()


class Frequencer:
	r"""Token frequence calculation wrapper.

	Args:
		tokenizer: text tokenizer
	"""

	def __init__(self, tokenizer):
		self.tokenizer = tokenizer # callable[str] -> iterable[str]

	def __call__(self, texts): # list[str] -> collections.Counter
		freqs = collections.Counter()

		for text in tqdm(texts):
			freqs.update(self.tokenizer(text))

		return freqs


########
# Main #
########

def main(
	datasets=[], # list[str]
	embeddings=[], # list[str]
	root: str = '.data'
):
	# Datasets
	for dataset in datasets:
		if dataset == 'IMDB':
			IMDB(root)
		elif dataset == 'SST':
			SST(root)

	# Embeddings
	for embedding in embeddings:
		load_embedding(embedding)

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Download datasets and embeddings.')

	parser.add_argument('-d', '--datasets', default=[], nargs='*', choices=['IMDB', 'SST'], help='datasets')
	parser.add_argument('-e', '--embeddings', default=[], nargs='*', choices=list(aliases.keys()), help='embeddings')
	parser.add_argument('-root', default='.data', help='datasets root path')

	args = parser.parse_args()

	main(
		datasets=args.datasets,
		embeddings=args.embeddings,
		root=args.root
	)
