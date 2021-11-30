from preprocessing import GetBase

bassers = [
         ("lemmatizator", GetBase(transf = "Lemmatization")),
         ("stemmer", GetBase(transf = "Stemming"))
         ]