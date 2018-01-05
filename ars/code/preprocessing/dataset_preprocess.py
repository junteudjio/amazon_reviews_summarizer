from __future__ import print_function
import argparse
import linecache
import nltk
import numpy as np
import os
import codecs
import sys
from tqdm import tqdm
import random
import csv
import re
import zipfile
from collections import Counter

from six.moves.urllib.request import urlretrieve
from nltk.corpus import stopwords as nltk_stopwords
import tensorflow as tf

from ars.code.preprocessing import contractions as _contractions
from ars.code.utils import misc_utils as utils

utils.check_tensorflow_version()

__all__ = ["maybe_download", "clean_sentence", "tokenize"]

reload(sys)
sys.setdefaultencoding('utf8')
random.seed(42)
np.random.seed(42)

raw_dataset_base_url = "https://www.kaggle.com/snap/amazon-fine-food-reviews/downloads/"
raw_filesize_expected = None
raw_filename = "Reviews.csv"
raw_file_numlines = 568454L


# Size train: 30288272
# size dev: 4854279
def _setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    download_prefix = os.path.join("download", "ars")
    data_prefix = os.path.join("data", "ars")

    parser.add_argument("--download_prefix", default=download_prefix)
    parser.add_argument("--data_prefix", default=data_prefix)

    parser.add_argument("--train_percentage", default=0.85, type=float)
    parser.add_argument("--val_percentage", default=0.10, type=float)
    parser.add_argument("--test_percentage", default=0.05, type=float)

    parser.add_argument("--shuffle_during_split", default=True, type=bool)
    parser.add_argument("--remove_stopwords", default=False, type=bool)
    parser.add_argument("--replace_contractions", default=True, type=bool)
    return parser.parse_args()


def _reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
            Number of blocks just transferred [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def _check_if_not_kaggle_url(func):
    def decorator(*pargs, **kwargs):
        url = pargs[0]
        downloaded = os.path.exists(os.path.join(pargs[2], pargs[1]))
        not_kaggle_url = not 'kaggle' in url
        assert downloaded or not_kaggle_url, 'For kaggle dataset download them with browser in dir: %s' % (pargs[2])
        return func(*pargs, **kwargs)

    return decorator


@_check_if_not_kaggle_url
def maybe_download(url, filename, prefix, num_bytes=None):
    """Takes an URL, a filename, and the expected bytes, download
    the contents and returns the filename
    num_bytes=None disables the file size check."""
    local_filename = None
    if not os.path.exists(os.path.join(prefix, filename)):
        try:
            print("Downloading file {}...".format(url + filename))
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                local_filename, _ = urlretrieve(url + filename, os.path.join(prefix, filename),
                                                _reporthook=_reporthook(t))
        except AttributeError as e:
            print("An error occurred when downloading the file! Please get the dataset using a browser.")
            raise e
    # We have a downloaded file
    # Check the stats and make sure they are ok
    file_stats = os.stat(os.path.join(prefix, filename))
    if num_bytes is None or file_stats.st_size == num_bytes:
        print("File {} successfully loaded".format(filename))
    else:
        raise Exception("Unexpected dataset size. Please get the dataset using a browser.")

    return local_filename


def tokenize(sentence):
    # Format words and remove unwanted characters
    # sentence = sentence.lower()
    sentence = re.sub(u'https?:\/\/.*[\r\n]*', '', sentence, flags=re.MULTILINE)
    sentence = re.sub(u'\<a href', ' ', sentence)
    sentence = re.sub(u'&amp;', '', sentence)
    # sentence = re.sub(u'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', sentence)
    sentence = re.sub(u'<br />', ' ', sentence)
    sentence = re.sub(u'\'', ' ', sentence)
    tokens = [token.replace(u"``", u'"').replace(u"''", u'"') for token in nltk.word_tokenize(sentence)]
    return tokens


def clean_sentence(sentence_str, contractions_dict=None, stopwords_set=None):
    if contractions_dict:
        sentence = []
        for word in sentence_str.split(u' '):
            if word in contractions_dict:
                sentence.append(contractions_dict[word])
            else:
                sentence.append(word)
        sentence_str = u' '.join(sentence)

    sentence = tokenize(sentence_str)

    if stopwords_set:
        sentence = [w for w in sentence if not w in stopwords_set]

    return u' '.join(sentence)


def _save_files(prefix, tier, indices):
    with open(os.path.join(prefix, tier + '.review'), 'wb') as review_file, \
            open(os.path.join(prefix, tier + '.summary'), 'wb') as summary_file:
        print('Creating %s split ...' % tier)
        for i in tqdm(indices, total=len(indices)):
            review_file.write(linecache.getline(os.path.join(prefix, 'all.review'), i))
            summary_file.write(linecache.getline(os.path.join(prefix, 'all.summary'), i))


def _split_tier(prefix, percents, shuffle=False):
    total_percentages = percents['train'] + percents['val'] + percents['test']
    assert 0.0 <= (1.0 - total_percentages) <= 1e-3, 'spliting percentages must sum to 1.0'

    review_filename = os.path.join(prefix, 'all' + '.review')
    # Get the number of lines
    with open(review_filename) as current_file:
        num_lines = sum(1 for line in current_file)

    # Get indices and split into 3 files
    indices_train = range(num_lines)[:int(num_lines * percents['train'])]
    indices_other = range(num_lines)[int(num_lines * percents['train'])::]
    if shuffle:
        np.random.shuffle(indices_train)
    _save_files(prefix, 'train', indices_train)

    indices_val = indices_other[:int(num_lines * percents['val'])]
    if shuffle:
        np.random.shuffle(indices_val)
        print("Shuffling...")
    _save_files(prefix, 'val', indices_val)

    indices_test = indices_other[int(num_lines * percents['val'])::]
    if shuffle:
        np.random.shuffle(indices_test)
        print("Shuffling...")
    _save_files(prefix, 'test', indices_test)


def _read_clean_save_data_from_csv(csvfilepath, save_dir, replace_contractions, remove_stopwords):
    save_review_path = os.path.join(save_dir, 'all.review')
    save_summary_path = os.path.join(save_dir, 'all.summary')
    if os.path.exists(save_review_path) and os.path.exists(save_summary_path): return

    stopwords = set(nltk_stopwords.words("english")) if remove_stopwords else None
    contractions = _contractions.contractions if replace_contractions else None

    with codecs.getreader("utf-8")(tf.gfile.GFile(csvfilepath, mode="rb")) as raw_csvfile, \
            codecs.getwriter("utf-8")(tf.gfile.GFile(save_review_path, mode="wb")) as review_file, \
            codecs.getwriter("utf-8")(tf.gfile.GFile(save_summary_path, mode="wb")) as summary_file:

        reader = csv.DictReader(raw_csvfile, delimiter=',')
        print('Cleaning raw dataset ...')
        for row in tqdm(reader, total=raw_file_numlines):
            review = clean_sentence(row['Text'], contractions, stopwords)
            summary = clean_sentence(row['Summary'], contractions, stopwords)
            if review and summary:
                review_file.write('%s\n' % review)
                summary_file.write('%s\n' % summary)


if __name__ == '__main__':
    args = _setup_args()

    download_prefix = args.download_prefix
    data_prefix = args.data_prefix
    percents = {
        'train': args.train_percentage,
        'val': args.val_percentage,
        'test': args.test_percentage,
    }

    print("Downloading datasets into {}".format(download_prefix))
    print("Preprocessing datasets into {}".format(data_prefix))

    if not os.path.exists(download_prefix):
        os.makedirs(download_prefix)
    if not os.path.exists(data_prefix):
        os.makedirs(data_prefix)

    # Download and extract the raw dataset
    raw_data_zip = maybe_download(raw_dataset_base_url, raw_filename + '.zip', download_prefix, raw_filesize_expected)
    raw_data_zip_ref = zipfile.ZipFile(os.path.join(download_prefix, raw_filename + '.zip'), 'r')
    raw_data_zip_ref.extractall(download_prefix)
    raw_data_zip_ref.close()

    # Read, clean the dataset and save to all.review and all.summary
    _read_clean_save_data_from_csv(os.path.join(download_prefix, raw_filename),
                                   save_dir=data_prefix,
                                   replace_contractions=args.replace_contractions,
                                   remove_stopwords=args.remove_stopwords
                                   )

    # Split train into train, validation and test into 85-10-5
    # Shuffle train, validation, test
    print("Splitting the dataset into train, validation and test")
    _split_tier(data_prefix, percents, shuffle=args.shuffle_during_split)