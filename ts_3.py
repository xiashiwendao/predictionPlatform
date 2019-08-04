#-*- coding: utf-8 -*-

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

#读取Excel数据
discfile = 'dataset/data_test.xls'
data = pd.read_excel(discfile,index_col=0)
data=data['number']
data.head()

data.plot(figsize=(12,8))
print(data)

#使用一阶差分，12步差分处理时间序列
diff_1 = data.diff(1)
diff1 = diff_1.dropna()
diff1_144_1 = diff_1-diff_1.shift(144)
diff1_144 = diff1_144_1.dropna()
#print(diff1_144_1)
#判断序列是否平稳，计算ACF，PACF
fig1 = plt.figure(figsize=(12,8))
ax1=fig1.add_subplot(111)
sm.graphics.tsa.plot_acf(diff1_144,lags=40,ax=ax1)
fig2 = plt.figure(figsize=(12,8))
ax2=fig2.add_subplot(111)
sm.graphics.tsa.plot_pacf(diff1_144,lags=40, ax=ax2)

#模型定阶，根据aic,bic,hqic,三者都是越小越好
# arma_mod01 = sm.tsa.ARMA(diff1_144,(0,1)).fit()
# print(arma_mod01.aic,arma_mod01.bic,arma_mod01.hqic)
# arma_mod10 = sm.tsa.ARMA(diff1_144,(1,0)).fit()
# print(arma_mod10.aic,arma_mod10.bic,arma_mod10.hqic)
# arma_mod60 = sm.tsa.ARMA(diff1_144,(6,0)).fit()
# print(arma_mod60.aic,arma_mod60.bic,arma_mod60.hqic)
arma_mod61 = sm.tsa.ARMA(diff1_144,(6,1)).fit()
print(arma_mod61.aic,arma_mod61.bic,arma_mod61.hqic)
#计算残差
resid = arma_mod61.resid
#看残差的acf和pacf,残差自相关图断尾，所以残差序列为白噪声
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)

print(sm.stats.durbin_watson(arma_mod61.resid.values))
# 残差DW检验，DW的值越接近2，表示越不相关
r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
d = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(d, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))

# 用模型预测
predict_data=arma_mod61.predict('2017/4/4 23:50','2017/4/6 00:00',dynamic=False)
# print(predict_data)
# print(diff_1)
# 由于是用差分后的值做的预测，因此需要把结果还原
# 144步差分还原
diff1_144_shift=diff_1.shift(144)
# print('print diff1_144_shift')
print(diff1_144_shift)
diff_recover_144=predict_data.add(diff1_144_shift)
# 一阶差分还原
diff1_shift=data.shift(1)
diff_recover_1=diff_recover_144.add(diff1_shift)
diff_recover_1=diff_recover_1.dropna() # 最终还原的预测值
print('预测值')
print(diff_recover_1)

# 实际值、预测值、差分预测值作图
fig, ax = plt.subplots(figsize=(12, 8))
ax = data.ix['2017-04-01':].plot(ax=ax)
ax = diff_recover_1.plot(ax=ax)
fig = arma_mod61.plot_predict('2017/4/2 23:50', '2017/4/6 00:00', dynamic=False, ax=ax, plot_insample=False)
plt.show()

