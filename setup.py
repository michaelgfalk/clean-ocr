"""Setup file for cleanocr"""

from setuptools import setup, find_packages

setup(
    name='clean-ocr',
    version='0.1',
    packages=find_packages(),
    author='Michael Falk',
    author_email='m.falk@westernsydney.edu.au',
    description='Use a deep encoder-decoder to clean messy OCR.',
    license='MIT',
    keywords='RNN GRU attention OCR text-mining',
    entry_points={
        'console_scripts': [
            'ocr-cleaner = cleanocr.commands:cli'
        ]
    },
    install_requires=[
        'Click',
        'tensorflow>=2.0',
        'importlib-resources',
        'scipy'
    ]

)
