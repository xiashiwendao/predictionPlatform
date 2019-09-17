# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
#import pymysql
import os, sys

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import time
import warnings

warnings.filterwarnings('ignore')


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.show()

#plot_learning_curves(rf, X_train, y_train)

def model_train(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    yHat = model.predict(X_test)
    #plt.figure(figsize=(15, 12))
    plt.plot(range(len(yHat)), yHat, "r-")
    plt.plot(range(len(y_test)), y_test, "b--")
    plt.show()
    mse = mean_squared_error(y_test, yHat)
    print("mse: ", mse, "rmse: ", np.sqrt(mse))
    #plot_learning_curves(rf, X_train, y_train)

def real_predict_curve(model, X, y):
    model.fit(X, y)
    yHat = model.predict(X)
    #plt.figure(figsize=(15, 12))
    plt.plot(range(len(X)), yHat, "r-")
    plt.plot(range(len(X)), y, "b--")
    plt.show()
    mse = mean_squared_error(yHat, y)
    print("mse: ", mse, "rmse: ", np.sqrt(mse))
    #plot_learning_curves(rf, X_train, y_train)

# 通过聚合获取daily数据
basePath = "dataset"
st = time.time()
filePath = os.path.join(basePath, "banner_carr.csv")
df_banner=pd.read_csv(filePath)
et = time.time()
print("cost time: ", et -st)
df_banner_filted = df_banner[['REPORT_DATE', 'MONTH_OF_YEAR',
       'YEAR_OF_WEEK', 'IS_VALENTINE', 'IS_TEACHER', 'IS_C_VALENTINE',
       'IS_CHILDREN', 'IS_NEWYEAR', 'IS_CHRISTMAS', 'IS_12', 'IS_11', 'IS_618',
       'IS_SUMMER', 'IS_WINTER', 'STORE_COUNT', 'CITY_COUNT']]
# df_banner_filted.head()
#df_banner_agg["REPORT_DATE"] = df_banner_agg["REPORT_DATE"].apply(lambda x: x.replace("/", "-"))
df_banner_filted['MONTH_OF_YEAR'] = df_banner_filted['REPORT_DATE'].apply(lambda x:x[5:7])
df_banner_filted['YEAR_OF_WEEK'] = df_banner_filted['REPORT_DATE'].apply(lambda x:x[0:4])
df_banner_agg = df_banner_filted.groupby(['REPORT_DATE']).mean().reset_index()#.sort_values(by=[""])

df_banner_agg.head(10)
df_banner_agg.to_csv(os.path.join(basePath, "banner_carr_daily.csv"))


# 获取趋势聚合信息
filePath = os.path.join(basePath, "SalesTrend.csv")
df_trend = pd.read_csv(filePath)
df_trend_carre = df_trend[df_trend.BANNER_NAME == 'Carrefour']
df_trend_carre_filterd = df_trend_carre.drop(['BANNER_NAME','PRODUCT_BKEY'], axis=1, inplace=False)
df_trend_carre_filterd.columns
df_trend_carre_agg = df_trend_carre_filterd.groupby(['REPORT_DATE']).sum().reset_index()
df_trend_carre_agg.head()
df_merge = df_banner_agg.merge(df_trend_carre_agg, on=['REPORT_DATE'], how="inner")
# len(df_merge)
df_merge.to_csv(os.path.join(basePath, "banner_trends_carr_daily.csv"))
labels = df_merge["QLI"].values
df_merge_clean = df_merge.copy()
df_merge_clean = df_merge_clean.drop("QLI", 1)
# df_merge_clean.head(1)
df_merge_clean.columns
# len(df_merge_clean)
# X_train = df_merge_clean[0:-6]
# y_train = labels[0:-6]
# X_test = df_merge_clean[-6:]
# y_test = labels[-6:]
# X_train, X_test, y_train, y_test = train_test_split(df_merge_clean.values, labels, test_size=0.1, random_state=42)
# X = df_merge_clean.values
# y = labels
# X = X_train
# y = y_train
# print("X train size: ", len(X))
# print("X Test size: ", len(X_test))
lr = LinearRegression()
# real_predict_curve(lr, df_merge_clean, labels)
reg = lr.fit(df_merge_clean, labels)
yHat = reg.predict(df_merge_clean)
print(labels)
print(yHat)
print("yHat value is: ")
for val in yHat:
    print(val)
#plt.figure(figsize=(15, 12))
plt.plot(range(len(df_merge_clean)), yHat, "c-")
plt.plot(range(len(df_merge_clean)), labels, "b--")
plt.show()
mse = mean_squared_error(yHat, y)
print("mse: ", mse, "rmse: ", np.sqrt(mse))

# rf = RandomForestRegressor(random_state=42,min_samples_leaf=150)
# real_predict_curve(rf, df_merge_clean, labels)

# svr =SVR()
# real_predict_curve(svr, df_merge_clean, labels)

# gbr = GradientBoostingRegressor()
# real_predict_curve(gbr, df_merge_clean, labels)


