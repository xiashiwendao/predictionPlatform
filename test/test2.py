from matplotlib.collections import LineCollection
import numpy as np
import math
import matplotlib.pyplot as plt

pi = 3.1415

x = np.linspace(0, 4*pi, 100)
y = [math.cos(xx) for xx in x]
lwidths = abs(x)
color = []
for i in range(len(y)):
    if i < 5:
        color.append('#FF0000')
    else:
        color.append('#000000')

print(x)
print(y)
print('--------------------------------------')
points = np.array([x, y]).T.reshape(-1, 1, 2)
print(points)
print('--------------------------------------')
segments = np.concatenate([points[:-1], points[1:]], axis=1)
print(segments)
lc = LineCollection(segments, linewidths=lwidths, color=color)

ax = plt.axes()
ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))
ax.add_collection(lc)
plt.show()

y = []
X = np.asarray(range(10))
# for x in X:
#     y.append( X^0.95)
y = X**0.55
# print(y)
plt.plot(X, y)
plt.show()

markvio = [[0.5,0.2,0.1,0],[0.5, 0, 0, 0], [0, 0.8, 0, 0], [0, 0, 0.9, 1]]
m = np.matrix(markvio)
m.T
m.T * m.T



import sklearn.datasets
a=sklearn.datasets.load_digits()
print(a)

import utils
print(utils.Logger.__name__)

import json
json_str = '[{"value":"LAST_2_MONTH_MID25","key":"fForecast"},{"value":"IS_12","key":"fChannel"},{"value":"IS_CHRISTMAS","key":"fTime"}]'
json_obj = json.loads(json_str)
type(json_obj)
json_obj['key'=='fForecast']['value']

def test():
    return 1, 2, 3

a, b, c = test()