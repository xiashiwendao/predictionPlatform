import numpy as np
import pandas as pd
#import pymysql
import os, sys

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from matplotlib import pyplot as plt

import time
import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def real_predict_curve(model, X, y):
    X_train = X[0:24]
    y_train = y[0:24]
    reg = model.fit(X_train, y_train)
    yHat = reg.predict(X)
    print("yHat value is: ")
    for val in yHat:
        print(val)
    plt.plot(range(len(df_merge_clean)), labels, "b-")
    plt.plot(range(len(df_merge_clean)), yHat, "r--")
    
    mse = mean_squared_error(yHat[-12:], y[-12:])
    print("mse: ", mse, "rmse: ", np.sqrt(mse))
    plt.show()

basePath = "dataset"
st = time.time()
filePath = os.path.join(basePath, "banner_carr_month.csv")
df_merge=pd.read_csv(filePath)
labels = df_merge["QLI"].values
df_merge_clean = df_merge.copy()
df_merge_clean = df_merge_clean.drop("QLI", 1)
X = df_merge_clean.copy()
y = labels.copy()
X_train = X[0:24]
y_train = y[0:24]
X_test = X[24:]
y_test = y[24:]
# lr = LinearRegression()
# real_predict_curve(lr, X, y)

rf = RandomForestRegressor()
# real_predict_curve(rf, df_merge_clean, labels)
# X_train = X[0:24]
# y_train = y[0:24]
reg = rf.fit(X_train, y_train)
#joblib.dump(reg, filename="rf.m")
feature_import = rf.feature_importances_
yHat = reg.predict(X_test)
# print("yHat value is: ")
# for val in yHat:
#     print(val)
mse = mean_squared_error(yHat[-12:], y[-12:])
print("mse: ", mse, "rmse: ", np.sqrt(mse))
# plt.plot(range(len(df_merge_clean)), yHat, "r-")
# plt.plot(range(len(df_merge_clean)), labels, "b--")
# plt.show()

# svr =SVR()
# real_predict_curve(svr, df_merge_clean, labels)

# gbr = GradientBoostingRegressor()
# # real_predict_curve(gbr, df_merge_clean, labels)
# reg = gbr.fit(X_train, y_train)
# yHat = reg.predict(X)
# print("yHat value is: ")
# for val in yHat:
#     print(val)
# mse = mean_squared_error(yHat[-12:], y[-12:])
# print("mse: ", mse, "rmse: ", np.sqrt(mse))
# plt.plot(range(len(df_merge_clean)), yHat, "r-")
# plt.plot(range(len(df_merge_clean)), labels, "b--")
# plt.show()

# def toOne(ar):
#     min_val = np.min(ar)
#     max_val = np.max(ar)
#     one_value = (ar - min_val)/(max_val - min_val)

#     return one_value

# print("OK")
# toOne(feature_import)
# for corr in df_merge.corr()["QLI"]:
#     print(corr)

