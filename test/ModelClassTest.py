import os, sys
sys.path.append(os.path.abspath("."))
import pandas as pd
from data import Test
# from data.DataExtractor import DataExtractor
from optimizerModel.RandomforestOptimizer import RandomforestOptimizer
from sklearn.ensemble import RandomForestRegressor
Test.hello_print()
# de = DataExtractor()
# df_carre_month = de.getDataByMonth(False)

df_carre_month = pd.read_csv(os.path.abspath(".\\dataset\\banner_carr_month.csv"))
len(df_carre_month.values)

y = df_carre_month["QLI"].values
# df_carre_month.columns
del df_carre_month["QLI"]

# df_carre_month.columns
# df_carre_month.head()
x = df_carre_month.values
#x
from pandas import DataFrame
df_carre_month.columns
df_date = df_carre_month.apply(lambda x: str(int(x["YEAR_OF_WEEK"])) + '-' + str(int(x['MONTH_OF_YEAR'])),axis=1)
# train_x = x[0:-7]
# print(train_x)
# train_y = y[0:-7]
# test_x = x[-7:]
# test_y = y[-7:]
train_x = x
train_y = y
# from sklearn.metrics import SCORERS
# sorted(SCORERS.keys())

rfo = RandomforestOptimizer(train_x, train_y)
#rf = rfo.getOptimizedModel(train_x, train_y)
rf = RandomForestRegressor()
rf.fit(train_x, train_y)
train_y_predict = rf.predict(train_x)

from matplotlib import pyplot as plt
X = df_date.values
# plt.plot(X, train_y_predict,color="red")
len(X)
colors = []
for i in range(21):
    colors.append('blue')
colors.append('red')
colors.append('red')
colors.append('red')
plt.plot(X[:-6], train_y[:-6], color='blue')
plt.plot(X[-7:], train_y[-7:], color='red')
x_name=[]
for i in range(len(X)):
    x_name.append('a')
plt.xticks(range(len(X)), x_name)

plt.show()

# from matplotlib.collections import LineCollection
# import numpy as np
# X_range =range(len(X))
# len(X)
# len(y)
# print(y)
# x_array = []
# for i in range(len(X)):
#     x_array.append(i+1)
# print(x_array)
# segments = np.array([x_array, y]).T.reshape(-1,1,2)
# lineCol = LineCollection(segments, color=colors)
# ax = plt.axes()
# ax.add_collection(lineCol)
# x_name = []
# for i in range(len(X)):
#     x_name.append('a')
# plt.xticks(x_name)

# plt.show()

