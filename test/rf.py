import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn import metrics
import os, sys
from matplotlib import pyplot as plt
basepath="C:\\Users\\wenyang.zhang\\Documents\\MySpace\\practice\\github\\predictionPlatform\\dataset"
basepath="..\dataset"
basepath= os.path.abspath('dataset')
filePath = os.path.join(basepath, "train_modified.csv")
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
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_