from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.utils.validation import check_is_fitted

class RemoveWords(BaseEstimator, TransformerMixin):
    def __init__(self, unwanted_words = stopwords.words("english")):
        self.unwanted_words = unwanted_words
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        #check_is_fitted(self)
        X_ = X.copy()
        X_ = X_.apply(lambda line: [w for w in line if not w.lower() in self.unwanted_words])
        return X_