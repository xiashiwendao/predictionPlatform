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

# insert year for date feature
df_carre_daily.insert(1, 'YEAR', pd.to_datetime(df_carre_daily['REPORT_DATE']).dt.year)
# insert day of year for date feature
df_carre_daily.insert(1, 'DAY_OF_YEAR', pd.to_datetime(df_carre_daily['REPORT_DATE']).dt.dayofyear)
# insert weekend flag for feature,1 is weekend ,0 is not.
df_carre_daily.insert(1, 'IS_WEEKEND', pd.to_datetime(df_carre_daily['REPORT_DATE']).dt.weekday)
df_carre_daily['IS_WEEKEND'][df_carre_daily['IS_WEEKEND'] <= 4], df_carre_daily['IS_WEEKEND'][
    df_carre_daily['IS_WEEKEND'] > 4] = 0, 1

# select data without year of 2019 for train.
train_dataset = df_carre_daily[~df_carre_daily['YEAR'].isin([2019])]
# train_data of y
train_y = train_dataset["QLI"].values
y_scaled = preprocessing.scale(train_y)

# predict of X
preidct_X = df_carre_daily[df_carre_daily['YEAR'].isin([2019])].drop(["Unnamed: 0", 'REPORT_DATE', "QLI"],
                                                                     axis=1)
preidct_X_scaled = preprocessing.scale(preidct_X)
# train_data of X
train_X = train_dataset.drop(["Unnamed: 0", 'REPORT_DATE', "QLI"], axis=1)
X_scaled = preprocessing.scale(train_X)

# SVR training-
svr_opt = SVROptimizer(X_scaled, train_y)
svr = svr_opt.get_optimized_model(X_scaled, train_y)

# predict
preidct_Y_scaled = svr.predict(preidct_X_scaled)

# del df_carre_daily, train_dataset

# dimensionality reduction and draw the resultl
pca = PCA(n_components=1)
train_X_PCA = pd.DataFrame(pca.fit_transform(train_X))
preidct_X_PCA = pd.DataFrame(pca.fit_transform(preidct_X))

plt.scatter(train_X_PCA, train_y, c='b', label='Actual')
plt.scatter(preidct_X_PCA, preidct_Y_scaled, c='r', label='predict')

plt.show()
