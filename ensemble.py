import numpy as np
import pandas as pd
import pymysql
import os, sys

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import time
import warnings

warnings.filterwarnings('ignore')

basePath = "dataset"
st = time.time()
filePath = os.path.join(basePath, "banner_carr.csv")
df_banner=pd.read_csv(filePath)
et = time.time()
print("cost time: ", et -st)
#df_banner2.head()

df_banner_agg = df_banner.drop("id", axis=1).groupby("REPORT_DATE").mean().reset_index()

#df_banner_agg["REPORT_DATE"] = df_banner_agg["REPORT_DATE"].apply(lambda x: x.replace("/", "-"))
df_banner_agg.head(1)

import datetime
def getDayNo(arr):
    count = len(arr)
    for i in range(count):
        dd = arr[i]
        dd = datetime.datetime.strptime(dd,"%Y-%m-%d")

        dayNo = dd.timetuple().tm_yday
        arr[i] = dayNo
        #print("date:", dd, "; dayNo: ", dayNo)

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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

#plot_learning_curves(rf, X_train, y_train)

def model_train(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    yHat = model.predict(X_test)
    #plt.figure(figsize=(15, 12))
    plt.plot(range(len(yHat)), yHat, "r-")
    plt.plot(range(len(y_test)), y_test, "b--")
    mse = mean_squared_error(y_test, yHat)
    print("mse: ", mse, "rmse: ", np.sqrt(mse))
    #plot_learning_curves(rf, X_train, y_train)

# 获取趋势聚合信息
filePath = os.path.join(basePath, "SalesTrend.csv")
df_trend = pd.read_csv(filePath)
df_trend_carre = df_trend[df_trend.BANNER_NAME == 'Carrefour']
df_trend_carre_agg = df_trend_carre.groupby('REPORT_DATE').mean().reset_index()
df_trend_carre_agg.head(3)

df_merge = df_banner_agg.merge(df_trend_carre_agg, on=["REPORT_DATE"], how="inner")
len(df_merge)

labels = df_merge["QLI"].values
df_merge_clean = df_merge.copy()
getDayNo(df_merge_clean["REPORT_DATE"].values)
df_merge_clean = df_merge_clean.drop("QLI", 1)
df_merge_clean.head(1)

len(df_merge_clean)

X_train, X_test, y_train, y_test = train_test_split(df_merge_clean.values, labels, test_size=0.1, random_state=42)
# X = df_merge_clean.values
# y = labels
X = X_train
y = y_train
print("X train size: ", len(X))
print("X Test size: ", len(X_test))
lr = LinearRegression()
lr.fit(X, y)
lr_hat = lr.predict(X_train)
lr_hat_test = lr.predict(X_test)

rf = RandomForestRegressor(random_state=42,min_samples_leaf=150)
rf.fit(X, y)
rf_hat = rf.predict(X_train)
rf_hat_test = rf.predict(X_test)

svr =SVR()
svr.fit(X, y)
svr_hat = svr.predict(X_train)
svr_hat_test = svr.predict(X_test)

gbr = GradientBoostingRegressor()
gbr.fit(X, y)
gbr_hat = gbr.predict(X_train)
gbr_hat_test = gbr.predict(X_test)

X_ensemble_train = []
for i,j,k,l in zip(lr_hat, rf_hat, svr_hat, gbr_hat):
    X_ensemble_train.append([i, j, k, l])
X_ensemble_test = []
for i,j,k,l in zip(lr_hat, rf_hat_test, svr_hat_test, gbr_hat_test):
    X_ensemble_test.append([i, j, k, l])
# print("len(X_ensemble): ", len(X_ensemble))
# print("len(y_train)", len(y_train))
rf_ensemble = RandomForestRegressor()
model_train(rf_ensemble, X_ensemble_train, y_train, X_ensemble_test, y_test)
