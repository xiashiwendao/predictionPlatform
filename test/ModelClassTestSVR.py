from optimizerModel.SVROptimizer import SVROptimizer
from matplotlib import pyplot as plt
import os, sys
import pandas as pd
import warnings
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import preprocessing

df_carre_daily = pd.read_csv("..\\dataset\\banner_trends_carr_daily.csv")
print(len(df_carre_daily.values))

# insert day of year for date feature
df_carre_daily.insert(0, 'DAY_OF_YEAR', pd.to_datetime(df_carre_daily['REPORT_DATE']).dt.dayofyear)
df_carre_daily.insert(1, 'YEAR', pd.to_datetime(df_carre_daily['REPORT_DATE']).dt.year)

# select data without year of 2019 for train.
train_dataset = df_carre_daily[~df_carre_daily['YEAR'].isin([2019])]
# train_data of y
train_y = train_dataset["QLI"].values
y_scaled = preprocessing.scale(train_y)

# predict of X
preidct_X = df_carre_daily[df_carre_daily['YEAR'].isin([2019])].drop(["Unnamed: 0", 'REPORT_DATE', 'YEAR', "QLI"],
                                                                axis=1)
preidct_X_scaled = preprocessing.scale(preidct_X)
# train_data of X
train_X = train_dataset.drop(["Unnamed: 0", 'REPORT_DATE', 'YEAR', "QLI"], axis=1)
X_scaled = preprocessing.scale(train_X)

# SVR training
svr_opt = SVROptimizer(X_scaled, train_y)
svr = svr_opt.get_optimized_model(X_scaled, train_y)

#predict
preidct_Y_scaled = svr.predict(preidct_X_scaled)

print(preidct_Y_scaled,len(preidct_Y_scaled))
del df_carre_daily, train_dataset

# dimensionality reduction and draw the resultl
pca = PCA(n_components=1)
new_x = pd.DataFrame(pca.fit_transform(preidct_X))
plt.scatter(new_x, preidct_Y_scaled, c='k', label='data', zorder=1)
plt.show()