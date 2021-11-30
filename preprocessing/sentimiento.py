from sklearn.base import BaseEstimator, TransformerMixin
from textblob import TextBlob
import pandas as pd

class AnalisisSentimiento(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def transform(self, X, y = None):
         #check_is_fitted(self)
         sent_pol = X.apply(lambda line: TextBlob(line).sentiment.polarity)
         sent_sub = X.apply(lambda line: TextBlob(line).sentiment.subjectivity)
         df = pd.DataFrame([sent_pol,sent_sub]).T.reset_index(drop=True)
         df.columns = [0, 1]
         return df
    
    def fit(self, X, y=None):
        return self