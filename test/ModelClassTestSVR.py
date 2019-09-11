from optimizerModel.SVROptimizer import SVROptimizer
from matplotlib import pyplot as plt
import os, sys
import pandas as pd
import warnings
import numpy as np

df_carre_month = pd.read_csv("..\\dataset\\banner_carr_month(1).csv")
print(len(df_carre_month.values))

# select data without year of 2019 for train.
train_dataset = df_carre_month[~df_carre_month['YEAR_OF_WEEK'].isin([2019])]
# train_data of y
train_y = train_dataset["QLI"].values
# predict of X
X = df_carre_month[df_carre_month['YEAR_OF_WEEK'].isin([2019])].drop(["Unnamed: 0", "QLI"], axis=1)
# train_data of X
train_X = train_dataset.drop(["Unnamed: 0", "QLI"], axis=1)

svr_opt = SVROptimizer(train_X, train_y)
svr = svr_opt.get_optimized_model(train_X, train_y)

predict = svr.predict(X)
print(predict)
del df_carre_month, train_dataset
#
# plt.figure()
# plt.subplot(1, 2, 1)
#

plt.subplot(1, 2, 2)
