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


# 通过聚合获取daily数据
basePath = "dataset"
dailyFile = os.path.join(basePath, "banner_trends_carr_daily.csv")
df_merge = pd.DataFrame()
# Daily文件不存在，则需要重新构建
if os.path.exists(dailyFile) == False:
    print("Do Daily File, need rebuild...")
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
    import time
    df_merge["REPORT_DATE"].head()
    df_merge["YEAR"] = df_merge["REPORT_DATE"].apply(lambda x: time.strptime(x, '%Y/%m/%d').tm_year)
    df_merge["DAY_NO"] = df_merge["REPORT_DATE"].apply(lambda x: time.strptime(x, '%Y/%m/%d').tm_yday)
    df_merge.to_csv(os.path.join(basePath, "banner_trends_carr_daily.csv"))
# 如果Daily文件已经存在则直接读取
else:
    print("Daily has exists...")
    df_merge = pd.read_csv(dailyFile)
    df_merge["YEAR"] = df_merge["REPORT_DATE"].apply(lambda x: time.strptime(x, '%Y-%m-%d').tm_year)
    df_merge["DAY_NO"] = df_merge["REPORT_DATE"].apply(lambda x: time.strptime(x, '%Y-%m-%d').tm_yday)
    #df_merge.to_csv(os.path.join(basePath, "banner_trends_carr_daily.csv"))

df_merge.columns
df_merge = df_merge.drop('REPORT_DATE', 1)
labels = df_merge["QLI"].values
df_merge_clean = df_merge.copy()
df_merge_clean = df_merge_clean.drop("QLI", 1)
# df_merge_clean.head(1)
df_merge_clean.columns
X = df_merge_clean.copy()
len(X)
y = labels.copy()
X_train = X[0:654]
y_train = y[0:654]
X_test = X[654:]
y_test = y[654:]
len(X_test)
from optimizerModel.RandomforestOptimizer import RandomforestOptimizer

rfo = RandomforestOptimizer(X_train, y_train)
rf = rfo.getOptimizedModel(X_train, y_train)
# reg = rf.fit(X_train, y_train)
# joblib.dump(reg, filename="rf.m")
# feature_import = rf.feature_importances_
yHat = rf.predict(X_test)
mse = mean_squared_error(y_test, yHat)
import numpy as np
print("mse is: ", mse, "rmse is: ", np.sqrt(mse))
print("len(y_test): ",len(y_test), "; len(yHat): ", len(yHat))
month_9 = np.sum(y_test[0:24])
print(month_9)
month_9_hat = np.sum(yHat[0:24])
print(month_9_hat)
mse = mean_squared_error([month_9], [month_9_hat])
print("month_9 mse is: ", mse, "month_9 rmse is: ", np.sqrt(mse))
month_10 = np.sum(y_test[25:49])
print(month_10)
from matplotlib import pyplot as plt
plt.plot(range(len(y_test)), y_test, "--")
plt.plot(range(len(yHat)), yHat, "-")
plt.show()
# svr =SVR()
# real_predict_curve(svr, df_merge_clean, labels)

# gbr = GradientBoostingRegressor()
# real_predict_curve(gbr, df_merge_clean, labels)


