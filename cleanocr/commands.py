"""Command line interface."""

from importlib.resources import path

import click
from cleanocr.utils import HParams
from cleanocr.train import Trainer

# pylint: disable=unnecessary-pass;
# pylint: disable=line-too-long;

@click.group()
def cli():
    """A system for training and using deep encoder-decodrs for ocr correction.

    The included model has been trained on 6000 newspaper articles downloaded
    from the Trove database, used for the ALTA-2017 challenge. If you have your
    own training data, you can use it either to improve the included model,
    or to train your own model from scratch (note, this will delete and
    replace the included model)."""
    pass

@cli.command()
@click.option('-j', '--json', help='path to json configuration file, defaults to file included in package', type=str, default='use included')
@click.option('-e', '--embedding_dim', help='embedding dimension of model', type=int)
@click.option('-u', '--units', help='dimensionality of hidden state of encoder and decoder', type=int)
@click.option('-K', '--K', help='number of attention vectors to calculate', type=int)
@click.option('-o', '--optimizer', help='optimizer to use, defaults to Adam', type=str, default='adam')
@click.option('-b', '--max_batch_size', help='maximum size of training batches', type=int)
@click.option('-t', '--tolerance', help='tolerance parameter for sentence splitter, defaults to 1.2', type=float, default=1.2)
@click.option('-l', '--max_len', help='maximum allowed length of training sequences, defaults to 50', type=int, default=50)
@click.option('-e', '--epochs', help='number of epochs to train for', type=int)
@click.option('-nl', '--norm_lim', help='norm limit for gradient clipping, defaults to 5', type=int, default=5)
@click.option('-tp', '--teacher_force_prob', help='initial probability of teacher forcing', type=float)
@click.option('-td', '--teacher_force_decay', help='decay factor for teacher forcing')
@click.option('-nc', '--num_chars', help='number of characters in vocabulary; if not supplied, it is inferred from the data', type=int, default=None)
@click.option('-c', '--checkpoint_dir', help='directory where model data should be saved, defaults to included model', type=str, default='included_model')
@click.option('-d', '--data_dir', help='directory where data is stored, defaults to current working directory', type=str, default='/')
@click.option('-x', '--x_path', help='file name of input sequences, if loading raw training data')
@click.option('-y', '--y_path', help='file name of output sequences, if loading raw training data')
@click.option('--saved-model/--new-model', help='use saved model, or create new one? defaults to saved-model', default=True)
@click.option('--saved-data/--raw-data', help='load preprocessed training data, or load from raw? defaults to saved-data', default=True)
def train(**kwargs):
    """Train an ocr model on data you supply.

    You can supply the hyperparemters in a json file, or use
    the config.json included in this package."""

    print(kwargs)

    # Get options, store in HParams
    if kwargs['json'] == 'use included':
        with path('cleanocr', 'config.json') as json_path:
            hparams = HParams.from_json(json_path=json_path)
    elif kwargs['json']:
        hparams = HParams.from_json(json_path=kwargs['json'])
    else:
        hparams = HParams(**kwargs)

    # Instantiate Trainer object
    trainer = Trainer(hparams, kwargs['saved_model'], kwargs['saved_data'])

    print(trainer)

@cli.command()
def clean():
    """Clean some OCR text using a trained model."""
    click.echo('Not implemented yet!')

if __name__ == '__main__':
    cli()
