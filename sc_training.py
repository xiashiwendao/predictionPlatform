# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
import pymysql
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

connection = pymysql.connect(user='root',password='123lug',database='sales',host='localhost',charset='utf8')
query = """select sum(Volume) as v from 
(select ReportDate, week(ReportDate) as w, Volume from fact_exd where banner_name = 'MT-Others-East' order by ReportDate)t 
group by w order by w"""
df = pd.read_sql(query, con=connection)

fig, ax = plt.subplots(1, 1, figsize=(12,8))
plt.plot(df.values)
plt.title('By Week Result')
plt.xticks(np.arange(1, len(df), 7))
plt.xlabel('Weeks')
plt.ylabel('Volumes')
plt.show()

# This function is used to get volume dataframe
use_week = False


def get_volume_df(banner_list, use_week=use_week, return_df=False):
    res = list()
    for i in range(len(banner_list)):
        if use_week:
            query_weeek = """select sum(Volume) as v from 
    (select ReportDate, week(ReportDate) as w, Volume from fact_exd where banner_name = "%s" order by ReportDate)t 
    group by w order by w""" % (str(banner_list[i]))
            df = pd.read_sql(query_weeek, con=connection)
        else:
            query_d = """select sum(Volume) as v from  fact_exd where 
banner_name = "%s" group by ReportDate order by ReportDate""" % (str(banner_list[i]))
            df = pd.read_sql(query_d, con=connection)
        if return_df:
            res.append(df)
        else:
            res.append(df.values)
    return res


# this function is based on volume dataframe to plot volume curve, can be in one or multi curve
def plot_volume_curve(data_list, use_week=use_week, one_plot=True):
    if one_plot:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    else:
        fig, ax = plt.subplots(len(data_list), 1, figsize=(14, 12))

    for i in range(len(data_list)):
        if one_plot:
            ax.plot(data_list[i], label=str(banner_list[i]))
            ax.set_ylabel('Volumes')
        else:
            ax[i].plot(data_list[i])
            ax[i].set_ylabel('Volumes')
            ax[i].set_title('%s Volume curve' % (banner_list[i]))

        if i == len(data_list):
            num_samples = len(data_list)
            _xlabel_set(ax, num_samples)

        def _xlabel_set(ax, num_samples, use_week=use_week, one_plot=one_plot):
            num = len(data_list)
            if not one_plot:
                ax = ax[num - 1]

            if use_week:
                ax.set_xlabel('Different weeks')
                ax.set_xticks(np.arange(1, num_samples + 1, 7))
            else:
                ax.set_xlabel('Different Days')
                ax.set_xticks(np.arange(1, num_samples + 1, 10))

    plt.legend()
    plt.show()

## this is just for  'Carrefour'
banner_list = ['Carrefour', 'Vanguard', 'RT-mart', 'Metro','BJ HL']
banner_list = ['Family mart']
banner_df = get_volume_df(banner_list)
plot_volume_curve(banner_df, one_plot=True)


path = "C:/Users/guangqiiang.lu/Documents/lugq/workings/201810/BANNER.csv"
df = pd.read_csv(path)

### This is Carrefour data
query_carre = "select ReportDate as REPORT_DATE,banner_name as BANNER_NAME, volume as day1 from lawson_carre where banner_name = 'Carrefour'"
label_carre = pd.read_sql(query_carre, con=connection)

label_carre['day2'] = label_carre.day1.shift(-1)
label_carre['day3'] = label_carre.day1.shift(-2)
label_carre['day4'] = label_carre.day1.shift(-3)
label_carre['day5'] = label_carre.day1.shift(-4)
label_carre['day6'] = label_carre.day1.shift(-5)
label_carre['day7'] = label_carre.day1.shift(-6)

path_trend = "C:/Users/guangqiiang.lu/Documents/lugq/workings/201810"

df_trend = pd.read_csv(path_trend + '/SalesTrend.csv')

df_trend_lawson = df_trend[df_trend.BANNER_NAME == 'Lawson']
df_trend_carre = df_trend[df_trend.BANNER_NAME == 'Carrefour']


### Because of time limit, Here I will first convert the carre data. For this data is based on product, so I have to convert
### data by date.
def aggregate_df(df, agg_func='mean'):
    if agg_func == 'mean':
        return df.groupby('REPORT_DATE').mean().reset_index()
    else:
        return df.groupby('REPORT_DATE').sum().reset_index()


df_trend_carre_agg = aggregate_df(df_trend_carre)
df_trend_lawson_agg = aggregate_df(df_trend_lawson)

#### Because I have to do the feature engineering works many times, Here I construct a class to process data
from sklearn.preprocessing import StandardScaler, Normalizer
import numpy as np


class Proce_train(object):
    def __init__(self, proc_data=False, proc_label=False, use_norm=False, use_stand=False, ):
        self.proc_data = proc_data
        self.proc_label = proc_label
        self.use_norm = use_norm
        self.use_stand = use_stand
        self._check_param()

    def split_data_label(self, df, label, new_features, ratio=.9):
        if not self._check_param():
            data, label = self.process_banner(df, label, new_features)
        else:
            data, label, processor = self.process_banner(df, label, new_features)
        s = int(len(data) * ratio)
        xtrain, ytrain = data[:s, :], label[:s]
        xtest, ytest = data[s:, :], label[s:]

        if not self._check_param():
            return xtrain, xtest, ytrain, ytest
        else:
            return xtrain, xtest, ytrain, ytest, processor

    def _check_param(self):
        if self.use_norm and self.use_stand:
            raise AttributeError('Can not use standard and norm both same time!')
        if not self.use_norm and not self.use_stand:
            return False

    def process_banner(self, df, label, new_features, return_df=False, re_tmp=False):
        df_avg = df.groupby(['REPORT_DATE']).mean().reset_index()
        df_new = df_avg.merge(new_features, on=['REPORT_DATE'], how='inner')
        df_new = df_new.merge(label, on=['REPORT_DATE'], how='inner')

        if re_tmp:
            return df_new, label
        df_new.drop(['REPORT_DATE', 'BANNER_NAME'], axis=1, inplace=True)
        df_new.dropna(inplace=True)

        data = df_new.iloc[:, :-7].astype(float)
        label = df_new.iloc[:, -7:].astype(float)
        if return_df:
            return data, label

        data_np = np.array(data).astype(np.float32)
        label_np = np.array(label).astype(np.float32)

        if self.proc_data:
            if self.use_stand:
                processor = StandardScaler().fit(data_np)
                data_np = processor.transform(data_np)
            if self.use_norm:
                processor = Normalizer().fit(data_np)
                data_np = processor.transform(data_np)
        if self.proc_label:
            if self.use_stand:
                processor = StandardScaler().fit(label_np)
                label_np = processor.transform(label_np)
            if self.use_norm:
                processor = Normalizer().fit(label_np)
                label_np = processor.transform(label_np)

        if not self._check_param():
            return data_np, label_np
        else:
            return data_np, label_np, processor


### split the carrefour data to be train and test
xtrain_c, xtest_c, ytrain_c, ytest_c = Proce_train(proc_data=False, use_norm=False).split_data_label(df_Carrefour, label_carre, df_trend_carre_agg, ratio=.9)

# Here I use random forest regresion for multi-output regression
rfr = RandomForestRegressor(random_state=1234)
mor_rfr = MultiOutputRegressor(rfr, n_jobs=-1)

mor_rfr.fit(xtrain_c, ytrain_c)

pred_rfr = mor_rfr.predict(xtest_c)

from sklearn.externals import joblib
joblib.dump(path+'/rfr.cpkt')

### Because I want to use machine learning first to build models, so here I write a function to evaluate the result and plot
def eval_rmse(ytest, pred, base_loss):
    loss = np.sqrt(metrics.mean_squared_error(ytest, pred))
    print('Model Testing RMSE = {:.6f}'.format(loss))
    if base_loss is not None:
        print("Model promote About %.2f %s above BaseLine"%((base_loss-loss)/loss*100, '%'))
    return loss

eval_rmse(ytest_c, pred_rfr)


