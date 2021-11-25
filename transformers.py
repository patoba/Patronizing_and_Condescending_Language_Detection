from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer 
from sklearn.utils.validation import check_is_fitted

class RemoveWords(BaseEstimator, TransformerMixin):
    def __init__(self, unwanted_words = stopwords.words("english")):
        self.unwanted_words = unwanted_words
        
    #def remove_from_word_list(words_lst):
    #    return [w for w in words_lst if not w in self.palabras_remover]
    
    def fit(self, X, y=None):
        #print("List of some word to remove {}".format(self.palabras_remover[:10]))
        return self
    
    def transform(self, X, y=None):
        check_is_fitted(self)
        X_ = X.copy()
        X_ = X_.apply(lambda line: [w for w in line if not w.lower() in self.unwanted_words])
        return X_
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
        print("POS tags to be used: {}".format(self.tag_list))
        return self
    
    def transform(self, X, y=None):
        check_is_fitted(self)
        X_ = X.copy()
        X_ = X_.apply(lambda line: pos_tag(line))
        X_ = X_.apply(lambda line: [w for w, pos in line if pos in self.tag_list])
        return X_

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
    
    def fit(self, X, y):
        if self.transf == 'Lematizar' or self.transf == 'Lemmatization':
            lemmatizer = WordNetLemmatizer()
            self.transf_f_ = lemmatizer.lemmatize  
        elif self.transf == 'Stemming' or self.transf == 'Enraizar':
            ps = PorterStemmer()
            self.transf_f_ = ps.stem
        return self
