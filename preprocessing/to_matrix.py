import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class ToMatrix(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        df = pd.DataFrame({"x0": X, "x1": X})
        return df

class ApplyColumn(BaseEstimator, TransformerMixin):
    def __init__(self, col = 0, trans = ToMatrix()):
        self.col = col
        self.trans = trans
    
    def fit(self, X, y = None):
        if type(self.trans) == list:
            self.trans = Pipeline(self.trans)
        return self

    def transform(self, X):
        if type(X) == pd.Series:
            X = pd.DataFrame({0: X.values})
        if self.col == -1:
            self.col = len(X.columns) - 1
        X_ = X.copy()
        
        X = X.iloc[:, self.col]
        X_.drop(columns = self.col, inplace = True)
        df = pd.DataFrame(self.trans.fit_transform(X))
        if len(X_.columns) > 0:
            df = pd.concat([X_, df], axis=1)
        df.columns = list(range(len(df.columns)))
        df.reset_index(drop = True)
       
        return df