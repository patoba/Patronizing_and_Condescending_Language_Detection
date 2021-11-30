from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sklearn.utils.validation import check_is_fitted

all_tag_list = {'CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD',
                'NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR',
                'RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ',
                'WDT','WP','WP$','WRB','RBS'}    
common_tag_list = {'JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP','VBZ'}

class GetTags(BaseEstimator, TransformerMixin):
    def __init__(self, tag_list = common_tag_list):
        self.tag_list = tag_list
        if not tag_list.issubset(all_tag_list):
            raise Warning("tag list provided does not match, look for the correct Peen Tree Bank POS tags")
        
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X, y=None):
        #check_is_fitted(self)
        X_ = X.copy()
        X_ = X_.apply(lambda line: pos_tag(line))
        X_ = X_.apply(lambda line: [w for w, pos in line if pos in self.tag_list])
        return X_