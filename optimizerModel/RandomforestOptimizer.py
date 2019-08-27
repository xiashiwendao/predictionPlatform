import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn import metrics
import os, sys
from matplotlib import pyplot as plt

class RandomforestOptimizer(object):
    def __init__(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        self.surfix = "********* "
        self.surfix_2 = "+++++ " + "+++++ "
        self.surfix_3 = "-----------------------"
        
    def toString(self):
        print("I'm RandomforestOptimizer")

    def getOptimizedModel(self, X, y):
        print("start default parameters check123...")
        rf0 = RandomForestRegressor(oob_score=True, random_state=10)
        # X = self.X
        # y = self.y
        
        # print("x is: \n", X)
        # print("y is: \n", y)
        rf0.fit(X, y)
        default_oob_score = rf0.oob_score_
        default_n_estimators = rf0.n_estimators
        self.oob_score = default_oob_score # 这个全局变量oob_score将会不断的用于后续的评分
        best_n_estimators = default_n_estimators
        # start n_estimators parameter optimize...
        print(self.surfix, "start n_estimators parameter optimize...")
        param_test1 = {'n_estimators':range(10,71,10)}
        (best_params, better_flag) = self.trainTheGridSearch(param_test1, rf0, self.X, self.y)

        rf0 = RandomForestRegressor(n_estimators= best_n_estimators,oob_score=True, random_state=10)
        rf0.fit(X, y)
        op_oob_score = rf0.oob_score_
        better_flag = self.oob_score < op_oob_score
        print(self.surfix_2, "default oob_score: ", self.oob_score, "; best oob_score: ", op_oob_score)
        self.oob_score = op_oob_score if better_flag else self.oob_score

        if better_flag == True:
            best_n_estimators = best_params["n_estimators"]
        print(self.surfix_2, "default estimators: ", default_n_estimators, "; best estimators: ", best_n_estimators)
        
        
        
        # start max_depth & min_samples_split parameters optimize...
        print(self.surfix, "start max_depth & min_samples_split parameters optimize...")
        param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
        
        default_max_depth = rf0.max_depth
        default_min_samples_split = rf0.min_samples_split
        best_max_depth = default_max_depth
        best_min_samples_split = default_min_samples_split

        (best_params, better_flag) = self.trainTheGridSearch(param_test2, rf0, self.X, self.y)


        rf0 = RandomForestRegressor(n_estimators=best_n_estimators,max_depth=best_max_depth, min_samples_split=best_min_samples_split,
                                    oob_score=True, random_state=10)
        rf0.fit(X, y)
        op_oob_score = rf0.oob_score_
        better_flag = self.oob_score < op_oob_score
        print(self.surfix_2, "default oob_score: ", self.oob_score, "; best oob_score: ", op_oob_score)
        self.oob_score = op_oob_score if better_flag else self.oob_score

        # if can't find better oob score, roll back to default value
        if better_flag == True:
            best_max_depth = best_params["max_depth"]
            best_min_samples_split = best_params["min_samples_split"]

        print(self.surfix_2, "default max_depth: ", default_max_depth, "; best max_depth: ", best_max_depth)
        print(self.surfix_2, "default_min_samples_split: ", default_min_samples_split, "; best_min_samples_split: ", best_min_samples_split)





        # 配对优化min_samples_split和min_samples_leaf两个参数值
        param_test3 = {'min_samples_split':range(80,150,20), 'min_samples_leaf':range(10,60,10)}

        default_min_samples_split = rf0.min_samples_split
        default_min_samples_leaf = rf0.min_samples_leaf
        best_min_samples_split = default_min_samples_split
        best_min_samples_leaf = default_min_samples_leaf

        (best_params, better_flag) = self.trainTheGridSearch(param_test3, rf0, self.X, self.y)


        rf0 = RandomForestRegressor(n_estimators=best_n_estimators,max_depth=best_max_depth, min_samples_split=best_min_samples_split,min_samples_leaf=best_min_samples_leaf,
                                    oob_score=True, random_state=10)
        rf0.fit(X, y)
        op_oob_score = rf0.oob_score_
        better_flag = self.oob_score < op_oob_score
        print(self.surfix_2, "default oob_score: ", self.oob_score, "; best oob_score: ", op_oob_score)
        self.oob_score = op_oob_score if better_flag else self.oob_score

        # if can't find better oob score, roll back to default value
        if better_flag == True:
            best_min_samples_leaf = best_params["best_min_samples_leaf"]
            best_min_samples_split = best_params["min_samples_split"]
        print(self.surfix_2, "default min_samples_split: ", default_min_samples_split, "; best min_samples_split: ", best_min_samples_split)
        print(self.surfix_2, "default min_samples_leaf: ", default_min_samples_leaf, "; best min_samples_leaf: ", best_min_samples_leaf)
        
        
        
        
        
        # 最后是max_features参数进行调优
        param_test4 = {'max_features':range(3,11,2)}
        default_max_features = rf0.max_features
        best_max_features = default_max_features

        (best_params, better_flag) = self.trainTheGridSearch(param_test4, rf0, self.X, self.y)
        
        
        rf0 = RandomForestRegressor(n_estimators=best_n_estimators,max_depth=best_max_depth, min_samples_split=best_min_samples_split,min_samples_leaf=best_min_samples_leaf,max_features=best_max_features,
                                    oob_score=True, random_state=10)
        rf0.fit(X, y)
        op_oob_score = rf0.oob_score_
        better_flag = self.oob_score < op_oob_score
        print(self.surfix_2, "default oob_score: ", self.oob_score, "; best oob_score: ", op_oob_score)
        self.oob_score = op_oob_score if better_flag else self.oob_score
        
        
        # if can't find better oob score, roll back to default value
        if better_flag == True:
            best_best_max_features = best_params["best_max_features"]
        
        print(self.surfix_2, "default max_features: ", default_max_features, "; best max_features: ", best_max_features)

    def trainTheGridSearch(self, parameters, rf, X, y):
        gsearch = GridSearchCV(estimator = rf, param_grid = parameters, scoring='neg_mean_squared_error',cv=5)
        gsearch.fit(X,y)
        print(self.surfix_3,"gsearch.cv_results_ \n", gsearch.cv_results_)
        print(self.surfix_3,"gsearch.best_params_ \n",gsearch.best_params_)
        print(self.surfix_3,"gsearch.best_score_ \n", gsearch.best_score_)
        # test for get oob score, to see if better
        #rf.fit(X, y)
        op_oob_score=rf.oob_score_
        better_flag = self.oob_score < op_oob_score
        self.oob_score = op_oob_score if better_flag else self.oob_score
        print(self.surfix_2, "oob_score: ", self.oob_score, "current oob_score: ", op_oob_score)
        return (gsearch.best_params_, better_flag)