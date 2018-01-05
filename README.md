#  Amazon Reviews Summarizer (ars)
A deep learning based text summarizer for Amazon reviews in tensorflow.

## Synopsis

- In this project, we use the amazon fine food reviews dataset to implement a text summarizers for these reviews.
- The code is meant to be high quality, clean and flexible enough to handle testing different kinds of architecture easily.
- To achieve this goal we use the same coding style as in the beautiful official tensorflow's example for neural machine translation,
from which we have used/adapted several code snippets.


## Architecture overview

- A seq2seq model with bi-directional, multi layers RNN/GRU/LSTM cells.
- Attention mechanism on the decoder.
- 3 different graphs for train, evaluation and test modes (more work but makes code clean and fast).
- used the beautiful dataset/iterator for input data feeding.
- used glove vectors for embeddings initialization.


## Requirements

Install all needed dependnecies through
`pip install -r requirements.txt`.
Or
`python setup.py develop`.



## Running the code


- You can get started by downloading the datasets and doing some basic preprocessing:

$ code/get_started.sh

Note that you will always want to run your code from the "ars" directory, not the code directory, like so:

$ python code/train.py

This ensures that any files created in the process don't pollute the code directoy.

- Now train/evaluate/test the model by running :

$ python code/run_ars.py

change the cmd line args to try different architecture flavours.


## Contributors

- junior Teudjio Mbativou : https://www.linkedin.com/in/junior-teudjio-3a125b8a


# BibTex and Acknowledgment

```
@article{luong17,
  author  = {Minh{-}Thang Luong and Eugene Brevdo and Rui Zhao},
  title   = {Neural Machine Translation (seq2seq) Tutorial},
  journal = {https://github.com/tensorflow/nmt},
  year    = {2017},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
