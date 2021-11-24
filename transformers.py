from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer 

class RemoveWords(BaseEstimator, TransformerMixin):
    def __init__(self, unwanted_words = stopwords.words("english")):
        self.unwanted_words = unwanted_words
        
    #def remove_from_word_list(words_lst):
    #    return [w for w in words_lst if not w in self.palabras_remover]
    
    def fit(self, X, y=None):
        #print("List of some word to remove {}".format(self.palabras_remover[:10]))
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        X_ = X_.apply(lambda line: [w for w in line if not w.lower() in self.unwanted_words])
        return X_

class GetTags(BaseEstimator, TransformerMixin):
    def __init__(self,tag_list=['JJ','JJR','JJS','VB','VBD','VBG','VBN','VBP','VBZ'],use_all_tags=False):
        self.tag_list = tag_list
        self.all_tags = set(['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD',
                             'NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS',
                            'RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ',
                            'WDT','WP','WP$','WRB'])
        #self.default_tags_dic = {'ADJ':['JJ','JJR','JJS'],'VERB':['VB','VBD','VBG','VBN','VBP','VBZ']}
        self.use_all_tags = use_all_tags
        if self.use_all_tags and len(tag_list)>0:
            print("Warning: input tag list will be ignored") 
        
    def fit(self, X, y=None):
        print("POS tags to be used: {}".format(self.tag_list))
        return self
    
    def transform(self, X, y=None):
        if set(self.tag_list).issubset(self.all_tags) and not self.use_all_tags:
            X_ = X.copy()
            X_ = X_.apply(lambda line: pos_tag(line))
            X_ = X_.apply(lambda line: [w for w, pos in line if pos in self.tag_list])
        elif self.use_all_tags:
            X_ = X.copy()
        else:
            raise Warning("tag list provided does not match, look for the correct Peen Tree Bank POS tags")
        return X_
class Get_base(BaseEstimator, TransformerMixin):
    def __init__(self, transf= None):
        self.transf = transf
    
    def transform(self, X, y = None):
         words = X.copy
         if self.transf=='Lematizar' or self.transf=='Lemmatization':
            lemmatizer = WordNetLemmatizer()
            words = words.apply(lambda line: [lemmatizer.lemmatize(w) for w in line])
         elif self.transf=='Stemming' or self.transf=='Enraizar':
            ps = PorterStemmer()
            words = words.apply(lambda line: [ps.stem(w) for w in line])
         else:
            raise Warning("""'Lematizar','Lemmatization','Stemming' and 'Enraizar' are the only
            possible options for transf paramater.""")
         return words
    
    def fit(self, X, y):
        return self
