# Web and text analytics

Project realized as part of the *Web and text analytics* course given by **Ashwin Ittoo** to graduate CS engineering students at the [University of Liège](https://www.uliege.be/) during the academic year 2020-2021.

The project consists in analyzing the influence of various word embeddings and network architectures/hyperparameters on the [Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis) task.

All the raw measures that were performed are provided in the [`embeddings.csv`](results/csv/embeddings.csv) and [`hyperparameters.csv`](results/csv/hyperparameters.csv) files. They are also showcased and analyzed in a [notebook](results/notebook.ipynb).

## Dependencies

The models and scripts are implemented using `Python` and some of its packages, including `torch`, `torchtext`, `spacy`, `sklearn`, etc. and their dependencies.

A `conda` environment file is provided, from which you can create a fresh environment.

```bash
conda env create -f environment.yml
conda activate sentiment
cd python
```

It is also necessary to download the english `spacy` dictionnary.

```bash
python -m spacy download en
```

Optionally, you can pre-download the datasets and pretrained embeddings required. For example,

```bash
python datasets.py -d IMDB -e glove.6B.100d fasttext.simple.300d
```

> The exhaustive lists of available datasets and embeddings are displayed with the flag `--help`.

## Usage

The main script is [`report.py`](python/report.py). It takes settings and parameters as arguments and produces a `CSV` report of training statistics such as speed (iter/sec), loss, accuracy, etc.

```bash
python report.py output.csv -dataset IMDB -net GRU -embedding glove.6B.100d -bidirectional -attention
```

> Use the `--help` flag for more information about the arguments.

## Authors

* **Maxime Meurisse** - [meurissemax](https://github.com/meurissemax)
* **François Rozet** - [francois-rozet](https://github.com/francois-rozet)
* **Valentin Vermeylen** - [ValentinVermeylen](https://github.com/ValentinVermeylen)
