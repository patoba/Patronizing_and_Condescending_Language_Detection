from time import time
from copy import deepcopy

import pandas as pd
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from preprocessing import UnTokenize, ApplyColumn, AnalisisSentimiento
from .preprocessing import preprocessesors
from .bassing import bassers
from .modeling import modelings_bag_words, modelings_sentiment_analysis
from .training import classifiers
from .selection import analyze_results
from .selection import min_max

def process_bag_of_words(X, y, n_jobs = -1):
    X_train, X_test, y_train, y_test = train_test_split(
                X, y, random_state=42)
    tiempos = dict()
    results = dict()
    for name_preprocessor, preprocessor in preprocessesors:
        for name_base, basser in bassers:
            for name_modeling, modeling in modelings_bag_words:
                for name_classifier, classifier, params_classifier in classifiers:
                    print(name_preprocessor, name_base, name_modeling, name_classifier)
                    tiempo_inicio = time()
                    pipe = Pipeline(preprocessor + 
                                    [("basser", basser),
                                     ("untokenize", UnTokenize()),
                                     ("modeling", modeling),
                                     ("resampling", SMOTE()),
                                     ("classifier", classifier)])
                    grid = GridSearchCV(pipe, params_classifier, 
                                scoring = "f1", 
                                n_jobs = n_jobs,
                                verbose = 2)
                    grid.fit(X_train, y_train) 
                    tiempo_total = time() - tiempo_inicio
                    tiempos[name_preprocessor, name_base, name_modeling, name_classifier] = tiempo_total
                    results[name_preprocessor, name_base, name_modeling, name_classifier] = deepcopy(grid)
    df_models = analyze_results(results, X_train, X_test, y_train, y_test)
    print(df_models)
    best_model = min_max(df_models)
    return results, tiempos, df_models, best_model

def process_sentiment_analysis(X, y, n_jobs = -1):
    X_train, X_test, y_train, y_test = train_test_split(
                X, y, random_state=42)
    tiempos = dict()
    results = dict()
    for name_preprocessor, preprocessor in preprocessesors:

        # 2 modelajes
        for name_modeling, modeling, params_modeling in modelings_sentiment_analysis:
            for name_classifier, classifier, params_classifier in classifiers:
                print(name_preprocessor, name_modeling, None, name_classifier)
                pipe = Pipeline(preprocessor + \
                                modeling + \
                                [("resampling", SMOTE()),
                                ("classifier", classifier)])
                param_grid = {**params_modeling, **params_classifier}
                tiempo_inicio = time()
                grid = GridSearchCV(pipe, param_grid, 
                                    scoring = "f1", 
                                    n_jobs = n_jobs)
                grid.fit(X_train, y_train) 
                tiempo_total = time() - tiempo_inicio
                print(grid.score(X_train, y_train), grid.score(X_test, y_test))
                tiempos[name_preprocessor, name_modeling, None, name_classifier] = tiempo_total
                results[name_preprocessor, name_modeling, None, name_classifier] = deepcopy(grid)

        # Analisis sentimiento individual
        for name_classifier, classifier, params_classifier in classifiers:
            print(name_preprocessor, None, "sentiment_analysis", name_classifier)
            pipe = Pipeline(preprocessor + \
                            [("untokenize", UnTokenize()),
                             ("sentiment_analysis", AnalisisSentimiento()),
                             ("resampling", SMOTE()),
                             ("classifier", classifier)])
            tiempo_inicio = time()
            grid = GridSearchCV(pipe, params_classifier, 
                                scoring = "f1", 
                                n_jobs = n_jobs)
            grid.fit(X_train, y_train) 
            tiempo_total = time() - tiempo_inicio
            print(grid.score(X_train, y_train), grid.score(X_test, y_test))
            tiempos[name_preprocessor, None, "sentiment_analysis", name_classifier] = tiempo_total
            results[name_preprocessor, None, "sentiment_analysis", name_classifier] = deepcopy(grid)

        # 2 modelajes y analisis sentimiento individual
        for name_modeling, modeling, params_modeling in modelings_sentiment_analysis:
            for name_classifier, classifier, params_classifier in classifiers:
                print(name_preprocessor, name_modeling, "sentiment_analysis", name_classifier)
                pipe = Pipeline(preprocessor + \
                                [("copy", ApplyColumn()),
                                 ("modeling", ApplyColumn(1, modeling)),
                                 ("untokenize", ApplyColumn(0, UnTokenize())),
                                 ("sentiment_analysis", ApplyColumn(-1, AnalisisSentimiento())),
                                 ("resampling", SMOTE()),
                                 ("classifier", classifier)])
                tiempo_inicio = time()
                grid = GridSearchCV(pipe, params_classifier, 
                                    scoring = "f1", 
                                    n_jobs = n_jobs)
                grid.fit(X_train, y_train) 
                tiempo_total = time() - tiempo_inicio
                print(grid.score(X_train, y_train), grid.score(X_test, y_test))
                tiempos[name_preprocessor, name_modeling, "sentiment_analysis", name_classifier] = tiempo_total
                results[name_preprocessor, name_modeling, "sentiment_analysis", name_classifier] = deepcopy(grid)
        
    df_models = analyze_results(results, X_train, y_train, X_test, y_test)
    print(df_models)
    best_model = min_max(df_models)
    return results, tiempos, df_models, best_model
