from re import split
import pandas as pd  
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class Tokenize(BaseEstimator, TransformerMixin):
    def __init__(self, caracteres_a_quitar = ['"', ',', '!', ':', '\d', '\.', "\'", ' ', '', '\d+' ]):
        self.caracteres_a_quitar = caracteres_a_quitar

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        print(X[X.isna()])
        return X.str.split(' ') \
               .dropna() \
               .apply(lambda l: [w for w in l if w not in self.caracteres_a_quitar])

class UnTokenize(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return X.apply(lambda l: ' '.join(l))