from sklearn.svm import SVR
import numpy as np


class SVROptimizer(object):
    def __init__(self, X_train, y_train):
        self.X = X_train
        self.y = y_train

    def getOptimizedModel(self, X, y):
        print("start SVR getOptimizedModel()")

        # C :Penalty Coefficient ,1e3 is more suitable.
        # gamma :param of kernel ,1e-2 is more suitable.

        # details:https://www.cnblogs.com/pinard/p/6117515.html 刘建平blog，第5点有SVR回归用法
        svr = SVR(kernel='rbf', C=1e3, gamma=0.01)
        svr.fit(X, y)
        return svr
