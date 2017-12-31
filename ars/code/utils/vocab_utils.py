# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility to handle vocabularies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import codecs
import os
import tensorflow as tf
import re
import argparse

from tensorflow.python.ops import lookup_ops
from tqdm import *
import numpy as np
from os.path import join as pjoin


from ars.code.utils import misc_utils as utils

utils.check_tensorflow_version()

__all__ = ['load_vocab', 'load_embed_txt', 'check_vocab', 'create_vocab_file', 'create_vocab_tables']

reload(sys)
sys.setdefaultencoding('utf8')

UNK = u"<unk>"
SOS = u"<s>"
EOS = u"</s>"

_START_VOCAB = [UNK, SOS, EOS]

UNK_ID = 0
SOS_ID = 1
EOS_ID = 2

def _setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    vocab_dir = os.path.join("data", "ars")
    corpus_dir = os.path.join("data", "ars")
    parser.add_argument("--corpus_dir", default=corpus_dir)
    parser.add_argument("--vocab_dir", default=vocab_dir)
    parser.add_argument("--random_init", default=True, type=bool)
    return parser.parse_args()


def load_vocab(vocab_file):
  vocab = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
    vocab_size = 0
    for word in f:
      vocab_size += 1
      vocab.append(word.strip())
  return vocab, vocab_size


def check_vocab(vocab_file, out_dir, check_special_token=True, sos=None,
                eos=None, unk=None):
  """Check if vocab_file doesn't exist, create from corpus_file."""
  if tf.gfile.Exists(vocab_file):
    utils.print_out("# Vocab file %s exists" % vocab_file)
    vocab, vocab_size = load_vocab(vocab_file)
    if check_special_token:
      # Verify if the vocab starts with unk, sos, eos
      # If not, prepend those tokens & generate a new vocab file
      if not unk: unk = UNK
      if not sos: sos = SOS
      if not eos: eos = EOS
      assert len(vocab) >= 3
      if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
        utils.print_out("The first 3 vocab words [%s, %s, %s]"
                        " are not [%s, %s, %s]" %
                        (vocab[0], vocab[1], vocab[2], unk, sos, eos))
        vocab = [unk, sos, eos] + vocab
        vocab_size += 3
        new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
        with codecs.getwriter("utf-8")(
            tf.gfile.GFile(new_vocab_file, "wb")) as f:
          for word in vocab:
            f.write("%s\n" % word)
        vocab_file = new_vocab_file
  else:
    raise ValueError("vocab_file '%s' does not exist." % vocab_file)

  vocab_size = len(vocab)
  return vocab_size, vocab_file


def create_vocab_table(vocab_file):
  """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
  vocab_table = lookup_ops.index_table_from_file(
    vocab_file, default_value=UNK_ID)
  return vocab_table


def load_embed_txt(embed_file):
  """Load embed_file into a python dictionary.

  Note: the embed_file should be a Glove formated txt file. Assuming
  embed_size=5, for example:

  the -0.071549 0.093459 0.023738 -0.090339 0.056123
  to 0.57346 0.5417 -0.23477 -0.3624 0.4037
  and 0.20327 0.47348 0.050877 0.002103 0.060547

  Args:
    embed_file: file path to the embedding file.
  Returns:
    a dictionary that maps word to vector, and the size of embedding dimensions.
  """
  emb_dict = dict()
  emb_size = None
  with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, 'rb')) as f:
    for line in f:
      tokens = line.strip().split(" ")
      word = tokens[0]
      vec = list(map(float, tokens[1:]))
      emb_dict[word] = vec
      if emb_size:
        assert emb_size == len(vec), "All embedding size should be same."
      else:
        emb_size = len(vec)
  return emb_dict, emb_size


def _basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]

   
def create_vocab_file(vocabulary_path, corpus_paths, tokenizer=None):
    if not tf.gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, str(corpus_paths)))
        vocab = {}
        for path in corpus_paths:
            with codecs.getreader("utf-8")(tf.gfile.GFile(path, mode="rb")) as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("processing line %d" % counter)
                    tokens = tokenizer(line) if tokenizer else _basic_tokenizer(line)
                    for w in tokens:
                        if w in vocab:
                            vocab[w] += 1
                        else:
                            vocab[w] = 1
        vocab_list = sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab_list))
        with codecs.getreader("utf-8")(tf.gfile.GFile(vocabulary_path, mode="wb")) as vocab_file:
            for w in _START_VOCAB:
                vocab_file.write(w + u"\n")
            for w in vocab_list:
                vocab_file.write(w + u"\n")
        print('Vocabulary file created successfully in file: %s' %vocabulary_path)
                

if __name__ == '__main__':
  args = _setup_args()

  vocab_path = pjoin(args.vocab_dir, "vocab.dat")
  create_vocab_file(vocab_path,
                    [pjoin(args.corpus_dir, "all.review"),
                     pjoin(args.corpus_dir, "all.summary")
                     ])
  