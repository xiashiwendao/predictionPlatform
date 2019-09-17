from optimizerModel.SVROptimizer import SVROptimizer
from matplotlib import pyplot as plt
import os, sys
import pandas as pd
import warnings
import numpy as np

df_carre_daily = pd.read_csv("..\\dataset\\banner_trends_carr_daily.csv")
print(len(df_carre_daily.values))

# insert day of year for date feature
df_carre_daily.insert(0, 'DAY_OF_YEAR', pd.to_datetime(df_carre_daily['REPORT_DATE']).dt.dayofyear)
df_carre_daily.insert(1, 'YEAR', pd.to_datetime(df_carre_daily['REPORT_DATE']).dt.year)

# select data without year of 2019 for train.
train_dataset = df_carre_daily[~df_carre_daily['YEAR'].isin([2019])]
# train_data of y
train_y = train_dataset["QLI"].values
# predict of X
preidct_X = df_carre_daily[df_carre_daily['YEAR'].isin([2019])].drop(["Unnamed: 0", 'REPORT_DATE', 'YEAR', "QLI"],
                                                                     axis=1)
# train_data of X
train_X = train_dataset.drop(["Unnamed: 0", 'REPORT_DATE', 'YEAR', "QLI"], axis=1)

svr_opt = SVROptimizer(train_X, train_y)
svr = svr_opt.get_optimized_model(train_X, train_y)

predict_y = svr.predict(preidct_X)
print(predict_y,len(predict_y))
del df_carre_daily, train_dataset
#
# plt.figure()
# plt.subplot(1, 2, 1)
#

plt.subplot(1, 2, 2)
