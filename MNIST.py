# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 21:01:22 2020

@author: augus
"""

""" 
Module Importation
"""

from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt 
import numpy  as np 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.base import BaseEstimator



"""
Fonction definition
"""

class Never5Classifier(BaseEstimator):
    def fit(self,X, y=None):
        pass
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)

def shuffle_sample(sample):
    shuffle_index = np.random.permutation(60000)
    sample = sample[shuffle_index]
    return(sample)
    
def cross_va_score(sample_train,sample_label_train,classifier):
    skfolds = StratifiedKFold(n_splits=3,random_state = 42)
    for train_index, test_index in skfolds.split(sample_train,sample_label_train):
        clone_clf = clone(classifier)
        X_train_folds = sample_train[train_index]
        y_train_folds = (sample_label_train[train_index])
        X_test_fold = sample_train[test_index]
        y_test_fold = (sample_label_train[test_index])
        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct/ len(y_pred))
    
    

"""
Script
"""


if __name__=='__main__':
    
    
    mnist = fetch_openml('mnist_784')
    X,y = mnist["data"],mnist["target"]
    some_digit = X[36000]
    some_digit_image = some_digit.reshape(28,28)
    X_train, X_test , y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    X_train = shuffle_sample(X_train)
    y_train = shuffle_sample(y_train)
    y_train_5  = (y_train == "5")
    y_test_5 = (y_test == "5")
    compteur_5 = 0
    compteur_autre = 0
    sgd_clf = SGDClassifier(random_state  =  42)
    sgd_clf.fit(X_train,y_train_5)
    never_5_clf = Never5Classifier()
    print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring = "accuracy"))
    print(cross_val_score(never_5_clf,X_train, y_train_5, cv=3, scoring = "accuracy"))
    
    
    
