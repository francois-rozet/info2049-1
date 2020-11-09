#!usr/bin/env python

###########
# Imports #
###########

import numpy as np
import torch
import torch.nn as nn


###########
# Classes #
###########

class SentimentNet(nn.Module):
	r"""Modular sentiment analysis network

	Args:
		embedding_shape: shape of the embedding
		output_size: output size (labels - 1, L - 1)
		pad_idx: index of the '<pad>' token in the embedding
		net: recurrent network name
		hidden_size: size of the hidden vectors
		num_layers: number of layers in the network
		dropout: dropout rate
		bidirectional: whether the network is bidirectional
		attention: whether the network has attention

	References:
		[1] Attention is all you need
		(Vaswani et al., 2017)
		https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

	Note:
		In SST and IMDB the labels are ordered from worst to best.
		Therefore, this network perform Ordinal (Text) Classification.
	"""

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

		self.num_directions = 2 if bidirectional else 1

		self.lin = nn.Linear(
			hidden_size * self.num_directions,
			output_size
		)

		self.attention = attention
		self.scale = float(np.sqrt(self.lin.in_features))

		self.softmax = nn.Softmax(dim=-1)
		self.softplus = nn.Softplus()

	def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
		r"""Forward pass

		Args:
			input: padded sequences, (B, T)
			lengths: sequence lengths, (B, 1)

		Returns:
			logits of the cumulative probabilites, (B, L - 1)
				output[i][j] = logit(P(y[i] <= j | x[i]))
			for each sequence 1 <= i <= B, and label 1 <= j < L
		"""

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

		hidden = hidden.transpose(0, 1) # batch_first
		hidden = hidden[:, -self.num_directions:].flatten(1) # last layers only

		# Scaled dot-product attention
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
	def prediction(logits: torch.Tensor) -> torch.Tensor:
		r"""Computes the label probabilities from the logits of the cumulative probabilites.

		Args:
			logits: logits of the cumulative probabilites, (B, L - 1)

		Returns:
			label probabilities (B, L)
		"""

		prob = torch.cat([
			torch.zeros((logits.size(0), 1), device=logits.device),
			torch.sigmoid(logits),
			torch.ones((logits.size(0), 1), device=logits.device)
		], dim=1)

		return prob[:, 1:] - prob[:, :-1]


class CLWithLogitsLoss(nn.Module):
	r"""Cumulative link with logits loss

	Args:
		reduction: reduction type ('none', 'mean', 'sum')

	References:
		[1] On the consistency of ordinal regression methods
		(Pedregosa et al., 2017)
		https://dl.acm.org/doi/abs/10.5555/3122009.3153011

	Note:
		For (B, 1) inputs, CLWithLogitsLoss is equivalent to BCEWithLogitsLoss.
	"""

	def __init__(self, reduction: str = 'mean'):
		super().__init__()

		self.reduction = reduction

	def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		r"""Forward pass

		Args:
			logits: logits of the cumulative probabilites, (B, L - 1)
			target: target labels, (B,)

		Returns:
			negative loglikelihood of target label probabilities, (1,) or (B,)
			See Eq. 11 in [1]

		Note:
			For stability reasons, the transformation
				log(sigmoid(x)) = x - log(1 + exp(x))
			is used.
		"""

		exps = logits.exp()
		logs = (1. + exps).log()

		neg_likelihoods = torch.cat([
			logs[:, :1] - logits[:, :1], # -log(p(0 < x < 1))
			logs[:, 1:] + logs[:, :-1] - (exps[:, 1:] - exps[:, :-1]).log(), # -log(p(i-1 < x < i))
			logs[:, -1:] # -log(p(k-1 < x < k))
		], dim=1)

		loss = torch.gather(neg_likelihoods, 1, target.unsqueeze(1)).squeeze(1)

		if self.reduction == 'mean':
			loss = loss.mean()
		elif self.reduction == 'sum':
			loss = loss.sum()

		return loss
