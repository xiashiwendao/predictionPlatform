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
filePath = os.path.join(basePath, "banner_carr_month.csv")
df_banner=pd.read_csv(filePath)
et = time.time()
print("cost time: ", et -st)
#df_banner2.head()
y = df_banner["QLI"]
X = df_banner.drop("QLI", axis=1)
df_banner.columns
lr = LinearRegression()
reg = lr.fit(X[0:-6], y[0:-6])
yHat = reg.predict(X)

plt.plot(range(len(df_banner)), yHat, "c-")
plt.plot(range(len(df_banner)), labels, "b--")
plt.show()

