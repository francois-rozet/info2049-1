workers = 8
time = '1:00:00'
output = '~/scripts/stats.csv'

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
python train.py {}
"""

run = ''

for dataset in ['IMDB', 'SST']:
	for vsize in [25000, 50000]:
		for embedding in ['glove.6B.100d', 'glove.6B.300d', 'fasttext.simple.300d']:
			for net in ['RNN', 'LSTM', 'GRU']:
				for hidden in [64, 256]:
					for layers in [1, 2]:
						for dropout in [0, 0.5]:
							for bidirectional in [False, True]:
								biflag = '-bidirectional' if bidirectional else ''

								name = f'{net}-{hidden}-{layers}-{dropout}-{int(bidirectional)}-{embedding}-{dataset}.{vsize}'
								formatted = content.format(
									name,
									name,
									workers,
									time,
									f'-o {output} -net {net} -hidden {hidden} -layers {layers} -dropout {dropout} {biflag} -embedding {embedding} -dataset {dataset} -vsize {vsize} -workers {workers}'
								)

								filename = f'{name}.sbatch'
								with open(filename, 'w') as f:
									f.write(formatted)

								run += f'sbatch {filename}\n'

with open('run.sh', 'w') as f:
	f.write(run)
