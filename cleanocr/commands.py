"""Command line interface."""

import click

# pylint: disable=unnecessary-pass;

@click.group()
def cli():
    """Passes off to subcommands."""
    pass

@cli.command()
@click.option('--print', help='something to print')
@click.option('--add', help='something to add')
def train():
    """Train an ocr model on the supplied data."""
    click.echo('Train the model on some given data')

@cli.command()
def clean():
    """Clean some OCR text using a trained model."""
    click.echo('Not implemented yet!')


if __name__ == '__main__':
    cli()
