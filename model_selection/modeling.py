import numpy as np

from sklearn.pipeline import Pipeline

from preprocessing import GetSentenceEmbedding, Doc2Vec, UnTokenize, \
                          AnalisisSentimiento
from sklearn.feature_extraction.text import TfidfVectorizer

# BAG OF WORDS

inicio = "modeling__"

modelings_bag_words = [
                        ("bag_of_words", TfidfVectorizer())
                      ]

# SENTIMENT ANALISIS

inicio = "column__" + inicio

param_word2vec = {
    inicio + "word2vec__method": [np.mean, np.sum], 
}

param_doc2vec = {
    inicio + "vector_size": [5, 100, 300]
}

param_sentiment_analisis = {}

word2vec = Pipeline([("untokenize", UnTokenize()), 
                    ("word2vec", GetSentenceEmbedding())])

modelings_sentiment_analysis = [
                                ("doc2vec", Doc2Vec(), param_doc2vec),
                                ("word2vec", word2vec, param_word2vec),
                                ]