import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class Doc2Vec(BaseEstimator, TransformerMixin):

    def __init__(self, vector_size = 5):
        self.vector_size = vector_size
    
    def fit(self, X, y = None):
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X)]
        model = Doc2Vec(documents, vector_size = 5, window = 2, min_count = 1)
        self.model_ = model
        return self

    def transform(self, X):
        transf = X.str.split(" ").apply(self.model_.infer_vector)
        return np.array(transf.tolist())
