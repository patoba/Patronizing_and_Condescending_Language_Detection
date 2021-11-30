from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import PorterStemmer, WordNetLemmatizer 
from sklearn.utils.validation import check_is_fitted

class GetBase(BaseEstimator, TransformerMixin):
    def __init__(self, transf = 'Stemming'):
        self.transf = transf
        if transf not in {'Lematizar','Lemmatization','Stemming', 'Enraizar'}:
            raise Warning("""'Lematizar','Lemmatization','Stemming' and 'Enraizar' are the only
            possible options for transf paramater.""")
    
    def transform(self, X, y = None):
         check_is_fitted(self)
         words = X.copy()
         words = words.apply(lambda line: [self.transf_f_(w) for w in line])
         return words
    
    def fit(self, X, y=None):
        if self.transf == 'Lematizar' or self.transf == 'Lemmatization':
            lemmatizer = WordNetLemmatizer()
            self.transf_f_ = lemmatizer.lemmatize  
        elif self.transf == 'Stemming' or self.transf == 'Enraizar':
            ps = PorterStemmer()
            self.transf_f_ = ps.stem
        return self