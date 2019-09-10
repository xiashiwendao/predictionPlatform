from optimizerModel import SVROptimizer
from matplotlib import pyplot as plt
import os, sys
import pandas as pd
import warnings
from optimizerModel import SVROptimizer

df_carre_month = pd.read_csv(".\\dataset\\banner_carr_month(2).csv")
print(len(df_carre_month.values))

train_y = df_carre_month["QLI"].values
train_X = df_carre_month.drop(["Unnamed: 0","QLI"],axis=1)
del df_carre_month["QLI"]


svr = SVROptimizer.SVROptimizer(train_X,train_y)
predict = svr.predict(X)

plt.figure()
plt.subplot(1, 2, 1)







plt.subplot(1, 2, 2)
