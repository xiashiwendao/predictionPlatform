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
        
    def optimize(self):
        print("start default parameters check...")
        rf0 = RandomForestClassifier(oob_score=True, min_samples_split=100,
                                        min_samples_leaf=20, max_depth=8, max_features='sqrt', random_state=10)
        rf0.fit(X,y)
        default_oob_score = rf0.oob_score_
        default_n_estimators = rf0.n_estimators
        self.oob_score = default_oob_score # 这个全局变量oob_score将会不断的用于后续的评分

        y_predprob = rf0.predict_proba(X)[:,1]
        metrics.roc_auc_score(y, y_predprob)
        print("AUC Score (Train): %f; rf0.oob_score_: %f" % (metrics.roc_auc_score(y, y_predprob, default_oob_score)))
       
        # start n_estimators parameter optimize...
        print("start n_estimators parameter optimize...")
        param_test1 = {'n_estimators':range(10,71,10)}
        (best_params, better_flag) = self.trainTheGridSearch(param_test1, rf0, self.X, self.y)
        best_n_estimators = default_n_estimators

        if better_flag == True:
            best_n_estimators = best_params["n_estimators"]
        
        # start max_depth & min_samples_split parameters optimize...
        print("start max_depth & min_samples_split parameters optimize...")
        param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
        rf0 = RandomForestClassifier(n_estimators= best_n_estimators, 
                                        min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10)
        best_max_depth = rf0.max_depth
        best_min_samples_split = rf0.min_samples_split

        (best_params, better_flag) = self.trainTheGridSearch(param_test1, rf0, self.X, self.y)
        # if can't find better oob score, roll back to default value
        if better_flag == True:
            best_max_depth = best_params["max_depth"]
            best_min_samples_split = best_params["min_samples_split"]
    
    def trainTheGridSearch(self, parameters, rf, X, y):
        gsearch = GridSearchCV(estimator = rf, param_grid = parameters, scoring='roc_auc',cv=5)
        gsearch.fit(X,y)
        gsearch.cv_results_, gsearch.best_params_, gsearch.best_score_
        # test for get oob score, to see if better
        rf.fit(X, y)
        op_oob_score=rf.oob_score_
        better_flag = self.oob_score < op_oob_score
        return (gsearch.best_params_, better_flag)