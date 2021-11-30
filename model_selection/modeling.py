import numpy as np

from sklearn.pipeline import Pipeline

from preprocessing import GetSentenceEmbedding, Doc2Vec, UnTokenize

inicio = "__"

param_word2vec = {
    inicio + "method": [np.mean, np.sum], 
}

param_doc2vec = {
    inicio + "vector_size": [5, 100, 300]
}

bag_of_words_pipe = Pipeline([("untokenize", UnTokenize()), 
                             ("", )])

modelings = [("bag_of_words", bag_of_words_pipe, param_)
             ("doc2vec", Doc2Vec(), param_doc2vec),
             ("word2vec", GetSentenceEmbedding(), param_word2vec),
             ("sentiment_analysis", )
             ]