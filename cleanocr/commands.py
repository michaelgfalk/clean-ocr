import click

@click.group()
def cli():
    pass

@click.command()
def train():
    click.echo('Train the model on some given data')
    pass

@click.command()
def clean():
    click.echo('Use a trained model to clean some OCR\'d text.')
    pass

if __name__ == '__main__':
    cli()