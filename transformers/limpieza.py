from re import split
import pandas as pd  
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class Limpieza(BaseEstimator, TransformerMixin):
    def __init__(self, caracteres_a_quitar = ['"', ',', '!', ':', '\d', '\.', "\'"]):
        self.caracteres_a_quitar = caracteres_a_quitar

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        caracteres_separados = "|".join(self.caracteres_a_quitar)
        return pd.Series(X).str.replace(" n\'t", "n\'t") \
               .str.replace(" \'re", "\'re") \
               .str.replace(" 's", "'s") \
               .str.replace("\([ A-Za-z%]+\)", "") \
               .str.replace("--", "") \
               .str.replace("<h>", "") \
               .str.replace(" \? s ", "'s") \
               .str.replace(" (@|#)\w+ ", " ") \
               .str.replace(caracteres_separados, '') \
               .str.replace("\\", " ") \
               .str.replace("  +", " ")