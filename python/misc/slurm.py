workers = 8
time = '1:00:00'

content = """#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name={} # Name of the job
#SBATCH --export=ALL # Export all environment variables
#SBATCH --output={}.log # Log-file
#SBATCH --cpus-per-task={} # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --time={} # Max execution time
#

# Activate your Anaconda environment
conda activate sentiment

# Run your Python script
cd ~/info2049-1/python
python report.py {}
"""

def slurmwrite(run: str):
	biflag = '-bidirectional' if bidirectional else ''
	atflag = '-attention' if attention else ''

	name = f'{net}-{hidden}-{layers}-{dropout}-{int(bidirectional)}-{int(attention)}-{embedding}-{dataset}.{vsize}-{task}'

	filename = f'{name}.sbatch'
	with open(filename, 'w') as f:
		f.write(content.format(
			name,
			name,
			workers,
			time,
			f'-o {output} -net {net} -hidden {hidden} -layers {layers} -dropout {dropout} {biflag} {atflag} -embedding {embedding} -dataset {dataset} -vsize {vsize} -workers {workers}'
		))

	return run + f'sbatch {filename}\n'

run = ''

# Testing embeddings

task = 'a'
output = '~/embeddings.csv'
hidden = 256
layers = 2
dropout = 0.5
bidirectional = True
attention = True

for dataset in ['IMDB', 'SST']:
	for vsize in [25000, 50000]:
		for embedding in ['word2vec.google.300d', 'glove.6B.100d', 'glove.6B.300d', 'fasttext.simple.300d']:
			for net in ['RNN', 'LSTM', 'GRU']:
				run = slurmwrite(run)

# Testing hyperparameters

task = 'b'
output = '~/hyperparameters.csv'
dataset = 'IMDB'
vsize = 50000
embedding = 'glove.6B.300d'

for net in ['RNN', 'LSTM', 'GRU']:
	for hidden in [64, 256]:
		for layers in [1, 2]:
			for dropout in [0, 0.5]:
				for bidirectional in [False, True]:
					for attention in [False, True]:
						run = slurmwrite(run)

# Run file

with open('run.sh', 'w') as f:
	f.write(run)
