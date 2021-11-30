import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

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
        return self

    def transform(self, X):
        X_ = X.copy()
        X = X.loc[:, self.col]
        X_.drop(columns = self.col, inplace = True)
        df = self.trans.fit_transform(X)
        df = pd.concat([X_, df], axis=1)
        df.columns = list(range(len(df.columns)))
        return df