from sklearn.svm import SVR
import numpy as np


class SVROptimizer(object):
    def __init__(self, X_train, y_train):
        self.X = X_train
        self.y = y_train

    def get_optimized_model(self, X, y):
        # simply print shape of dataset
        print("start SVR getOptimizedModel()")
        for i in ["X", X, "y", y]:
            if len(i) == 1:
                print("train_%s shape:" % i, end="")
            else:
                print(str((np.shape(i))))

        # C :Penalty Coefficient ,1e3 is more suitable.
        # gamma :param of kernel ,1e-2 is more suitable.
        # details: https://www.cnblogs.com/pinard/p/6117515.html 刘建平blog，第5点有SVR回归用法
        svr = SVR(kernel='rbf', C=1e3, gamma=0.01)
        svr.fit(X, y)
        return svr
