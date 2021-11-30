from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import numpy as np
import pickle

subset_file = "datasets/en_embeddings_subset.p"
en_embeddings_subset = pickle.load(open(subset_file, "rb"))

class GetSentenceEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_dic = en_embeddings_subset, method=np.sum, 
                 replace = False):
        """
        embedding_dic: Dictionary or gensim KeyedVectors with the word as key and its embedding vector as value.
        method: Decide how to aggregate word vectors for a sentence, np.sum or np.mean.
        replace: If true, replace words not found in embedding_dic with word "unknown" otherwise they are not considered
        """
        self.embedding_dic = embedding_dic
        self.method = method
        self.replace = replace
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        #check_is_fitted(self)
        X_ = X.copy()
        word_in_embedding = list(self.embedding_dic.keys())[0] 
        embedding_dim = len(self.embedding_dic[word_in_embedding])
        if self.replace:
            X_ = X_.apply(lambda line: 
                          [self.embedding_dic[word] 
                           if word in self.embedding_dic 
                           else self.embedding_dic['unknown']
                          for word in line])
        else:
            X_ = X_.apply(lambda line: [self.embedding_dic[word] for word in line if word in self.embedding_dic])
        X_[X_.apply(lambda l: len(l) == 0)] = 0
        X_[X_.apply(lambda l: l == 0)] = X_[X_.apply(lambda l: l == 0)].map({0:[np.zeros(embedding_dim)]}) 
        X_ = X_.apply(lambda line: self.method(np.array(line), axis=0).tolist())
        X_ = np.array(X_.tolist())
        return X_
