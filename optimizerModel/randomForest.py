import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn import metrics
import os, sys
from matplotlib import pyplot as plt

class RandomforestOptimizer(object):
    def __init__(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        
    def optimize():
        print("start default parameters check...")
        rf0 = RandomForestClassifier(oob_score=True, min_samples_split=100,
                                        min_samples_leaf=20, max_depth=8, max_features='sqrt', random_state=10)
        rf0.fit(X,y)
        default_oob_score=rf0.oob_score_
        y_predprob = rf0.predict_proba(X)[:,1]
        

        metrics.roc_auc_score(y, y_predprob)
        print("AUC Score (Train): %f; rf0.oob_score_: %f" % (metrics.roc_auc_score(y, y_predprob, default_oob_score))
       
        # start n_estimators parameter optimize...
        print("start n_estimators parameter optimize...")
        param_test1 = {'n_estimators':range(10,71,10)}
        gsearch1 = GridSearchCV(estimator = rf0), 
                            param_grid = param_test1, scoring='roc_auc',cv=5)
        gsearch1.fit(X,y)
        gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_
        default_n_estimators = rf0.n_estimators
        best_n_estimators = gsearch1.best_params_["n_estimators"]
        rf0 = RandomForestClassifier(oob_score=True, n_estimators=best_n_estimators, min_samples_split=100,
                                        min_samples_leaf=20, max_depth=8, max_features='sqrt', random_state=10)
        rf0.fit(X, y)
        op_oob_score=rf0.oob_score_

        # if can't find better oob score, roll back to default value
        if(default_oob_score > op_oob_score):
            best_n_estimators = default_n_estimators
        # start max_depth & min_samples_split parameters optimize...
        print("start max_depth & min_samples_split parameters optimize...")
        param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
        gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= best_n_estimators, 
                                        min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10),
        param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
        gsearch2.fit(X,y)
        gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_
        
        best_max_depth = gsearch2.best_params_["max_depth"]
        best_min_samples_split = gsearch2.best_params_["min_samples_split"]
    
    def trainTheGridSearch(parameters, rf0, X, y):
        gsearch1 = GridSearchCV(estimator = rf0), 
                            param_grid = parameters, scoring='roc_auc',cv=5)
        gsearch1.fit(X,y)
        gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_
        
        rf0.fit(X, y)
        op_oob_score=rf0.oob_score_

        better_flag = default_oob_score < op_oob_score
        return (gsearch1.best_params_, better_flag)