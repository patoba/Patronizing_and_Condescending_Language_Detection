import numpy as np

from imblearn.pipeline import Pipeline

from preprocessing import GetSentenceEmbedding, Doc2Vec, UnTokenize, \
                          AnalisisSentimiento
from sklearn.feature_extraction.text import TfidfVectorizer

# BAG OF WORDS

inicio = "modeling__"

modelings_bag_words = [
                        ("bag_of_words", TfidfVectorizer())
                      ]

# SENTIMENT ANALISIS

inicio = "" + inicio

param_word2vec = {
    "word2vec__method": [np.mean, np.sum], 
}

param_doc2vec = {}

param_sentiment_analisis = {}

doc2vec_pipe = [("untokenize", UnTokenize()), 
                ("doc2vec", Doc2Vec())]

modelings_sentiment_analysis = [
                                ("doc2vec", doc2vec_pipe, param_doc2vec),
                                #("word2vec", [GetSentenceEmbedding()], param_word2vec),
                                ]