import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer 

def obtener_palabras_df(df,text_col):
    # Tokenización
    lines = []
    for line in df[text_col]:
        lines.append(word_tokenize(line))
    # Remover signos de puntuación
    words = []
    for line in lines:
        words += [word for word in line if word.isalpha()]
    return words
    
def limpiar_palabras(words,remove_all_stopword=True,relevant_words=None,POS_tags=None,transf=None):
    # Remover palabras_vacías
    stops_words = stopwords.words('english')
    if not remove_all_stopword:
        if relevant_words == None:
            raise Warning("You need the relevant words of dataset")
        else:
            bad_sw = set(stop_words) - set(relevant_words)
            words = [w for w in words if not w in bad_sw]
    else:
        words = [w for w in words if not w in stops_words]
    # Remover categorías gramaticales
    if POS_tags is not None:
        tagged_words = pos_tag(words)
        words = [word for word,pos in tagged_words if pos not in POS_tags]
    # Convertir a minúsculas si es necesario
    tagged_words = pos_tag(words)
    words = [w.lower() for w,pos in tagged_words if pos != 'NNP']
    # Lematizar o estamatizar
    if transf is not None:
        if transf=='Lematizar' or transf=='Lemmatization':
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(w) for w in words]
        elif transf=='Stemming' or transf=='Enraizar':
            ps = PorterStemmer()
            words = [ps.stem(w) for w in words]
        else:
            raise Warning("""'Lematizar','Lemmatization','Stemming' and 'Enraizar' are the only
            possible options for transf paramater.""")
    return words
