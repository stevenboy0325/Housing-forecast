import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('./house_price1.csv')
x1 = data.iloc[:, 0:1]
x2 = data[['area']]
y = data.iloc[:, -1:]

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
axes = Axes3D(fig)
axes.scatter3D(x1, x2, y)
x3 = data.iloc[:, :-1]

from sklearn import linear_model
mode1 = linear_model.LinearRegression().fit(x3, y)
print(mode1.coef_)
print(mode1.intercept_)

# y=a*x1+b*x2+c
x11 = np.arange(0, 4.0, 0.1)
x22 = np.arange(10, 70, 1.5)
X1, X2 = np.meshgrid(x11, x22)
print(x11.size)
print(x22.size)
y11 = 0.695186*X1+0.01225383*X2+0.15032823
axes.plot_surface(X1, X2, y11, alpha=0.3)  # alpha表示透明度
plt.show()
