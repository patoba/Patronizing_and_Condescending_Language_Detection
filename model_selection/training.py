from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

param_grid_logistic_regression = {
    "classifier__penalty": ['l1', 'l2', 'elasticnet', 'none']
}

# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html

param_grid_knn = {
    "classifier__n_neighbors": [3, 5, 10, 15, 25, 40],
    "classifier__metric": ["minkowski"]
}

# Checar hiperparametros del paper

param_svm = {
    "classifier__C": [1, 5, 10, 15],
}

param_random_forest = {
    "classifier__n_estimators": [10, 50, 100],
}

classifiers = [
              ("logistic_regression", LogisticRegression(), param_grid_logistic_regression),
              ("knn", KNeighborsClassifier(), param_grid_knn),
              ("svm", SVC(), param_svm),
              ("random_forest", RandomForestClassifier(), param_random_forest)
              ]
