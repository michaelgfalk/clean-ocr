from setuptools import setup, find_packages

setup(
    name = 'clean-ocr',
    version = '0.1',
    packages = find_packages(),
    author = 'Michael Falk',
    author_email = 'm.falk@westernsydney.edu.au',
    description = 'Uses a seq2seq model to clean messy OCR.',
    license = 'MIT',
    keywords = 'RNN GRU attention OCR text-mining',
    entry_points = {
        'console_scripts': [
            'clean-ocr = cleanocr.commands:cli'
        ]
    }

)