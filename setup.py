# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ars',
    version='0.1.0',
    description='A deep learning text summarizer for Amazon reviews in tensorflow.',
    long_description=readme,
    author='Junior Teudjio Mbativou',
    author_email='teudjiombativou@gmail.com',
    url='https://github.com/teudjio',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'nose',
        'sphinx',
        'matplotlib',
        'scipy',
        'numpy',
        'tensorflow>=1.4.0',
        'joblib',
        'nltk>=3.2.2',
        'tqdm',
        'pyprind',
    ]
)

