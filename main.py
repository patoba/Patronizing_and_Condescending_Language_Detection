from random import seed
import numpy as np
import pandas as pd

from model_selection import process_bag_of_words, process_sentiment_analysis

# establish seed

SEED = 42
seed(SEED)
np.random.seed(SEED)

# Files

file_url = "datasets/dontpatronizeme_pcl.tsv"

def read_csv(file_url):
    data = pd.read_csv(file_url, skiprows=4, sep='\t', header=None,
                       names=['index','id_new','word_key', 'country','text','label'])
    mapeo = lambda x : 0 if x < 2 else 1 
    data['label'] = data['label'].apply(mapeo)
    data = data[~data.text.isna()]
    X = data.text
    y = data.label
    return X, y

def bag_of_words(file_url = file_url):
    X, y = read_csv(file_url)
    return process_bag_of_words(X, y)

def sentiment_analysis(file_url = file_url):
    X, y = read_csv(file_url)
    return process_sentiment_analysis(X, y)

if __name__ == '__main__':
    bag_of_words(file_url)
    sentiment_analysis(file_url)

