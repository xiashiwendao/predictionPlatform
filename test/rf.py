import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn import metrics
import os, sys
from matplotlib import pyplot as plt
basepath= os.path.abspath('dataset')
# filePath = os.path.join(basepath, "train_modified.csv")
filePath = os.path.join(basepath, "banner_group.csv")
train = pd.read_csv(filePath)

target='Disbursed' # Disbursed的值就是二元分类的输出
IDcol = 'ID'
train['Disbursed'].value_counts()


x_columns = [x for x in train.columns if x not in [target, IDcol]]
X = train[x_columns]
y = train['Disbursed']

rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(X,y)
print(rf0.oob_score_)
y_predprob = rf0.predict_proba(X)[:,1]

metrics.roc_auc_score(y, y_predprob)
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

param_test1 = {'n_estimators':range(10,71,10)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(X,y)
gsearch1.best_estimator_
cvres = gsearch1.cv_results_
cvres

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(mean_score), params)
print("complete")
gsearch1.best_params_, gsearch1.best_score_

param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60, 
                                  min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10),
   param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(X,y)
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_




rf1 = RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=110,
                                  min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10)
rf1.fit(X,y)
print(rf1.oob_score_)

# 配对优化min_samples_split和min_samples_leaf两个参数值
param_test3 = {'min_samples_split':range(80,150,20), 'min_samples_leaf':range(10,60,10)}
gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60, max_depth=13,
                                  max_features='sqrt' ,oob_score=True, random_state=10),
   param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(X,y)
gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_

# 最后是max_features参数进行调优
param_test4 = {'max_features':range(3,11,2)}
gsearch4 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,
                                  min_samples_leaf=20 ,oob_score=True, random_state=10),
   param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(X,y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


rf2 = RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,
                                  min_samples_leaf=20,max_features=7 ,oob_score=True, random_state=10)
rf2.fit(X,y)
print(rf2.oob_score_)