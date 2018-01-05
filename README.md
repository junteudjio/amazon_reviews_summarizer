#  Amazon Reviews Summarizer (ars)
A deep learning text sumarizer for Amazon reviews in tensorflow.
The project has several dependencies that have to be satisfied before running the code. You can install them using your preferred method -- we list here the names of the packages using `pip`.

# Requirements

The starter code provided pressuposes a working installation of Python 2.7, as well as a TensorFlow 0.14.0.

It should also install all needed dependnecies through
`pip install -r requirements.txt`.
Or
`python setup.py develop`.



# Running your ars

You can get started by downloading the datasets and doing some basic preprocessing:

$ code/get_started.sh

Note that you will always want to run your code from the "ars" directory, not the code directory, like so:

$ python code/train.py

This ensures that any files created in the process don't pollute the code directoy.
