import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import Doc2Vec as new_Doc2Vec, TaggedDocument

class Doc2Vec(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    
    def fit(self, X, y = None):
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X)]
        model = new_Doc2Vec(documents = documents)
        self.model_ = model
        return self

    def transform(self, X):
        transf = X.str.split(" ").apply(self.model_.infer_vector)
        return np.array(transf.tolist())
