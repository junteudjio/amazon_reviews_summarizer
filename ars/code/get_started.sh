#!/usr/bin/env bash
# Downloads raw data into ./download
# and saves preprocessed data into ./data
# Get directory containing this script

CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && cd .. && pwd )"

export PYTHONPATH=$PYTHONPATH:$CODE_DIR
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR

pip install -r $CODE_DIR/requirements.txt

# download punkt, perluniprops
if [ ! -d "/usr/local/share/nltk_data/tokenizers/punkt" ]; then
    python2 -m nltk.downloader punkt
fi

# download stopwords
if [ ! -d "/usr/local/share/nltk_data/tokenizers/stopwords" ]; then
    python2 -m nltk.downloader stopwords
fi



# dataset_preprocess is in charge of downloading
# and formatting the data to be consumed later
DATA_DIR=data
DOWNLOAD_DIR=download
mkdir -p $DATA_DIR
rm -rf $DATA_DIR
python2 $CODE_DIR/preprocessing/dataset_preprocess.py

# Download distributed word representations
python2 $CODE_DIR/preprocessing/dwr.py
