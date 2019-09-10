from optimizerModel.SVROptimizer import SVROptimizer
from matplotlib import pyplot as plt
import os, sys
import pandas as pd
import warnings

df_carre_month = pd.read_csv("..\\dataset\\banner_carr_month(1).csv")
print(len(df_carre_month.values))

train_y = df_carre_month["QLI"].values
X = df_carre_month[df_carre_month['YEAR_OF_WEEK'].isin([2019])]
train_X = df_carre_month[~df_carre_month['YEAR_OF_WEEK'].isin([2019])].drop(["Unnamed: 0","QLI"],axis=1)


svr_opt = SVROptimizer(train_X,train_y)
svr = svr_opt.getOptimizedModel(train_X, train_y)

predict = svr.predict(X)
del df_carre_month["QLI"]

plt.figure()
plt.subplot(1, 2, 1)







plt.subplot(1, 2, 2)
