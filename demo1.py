import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('./house_price1.csv')
print(data)

x = data[['area']]
y = data.iloc[:, -1:]
fig = plt.figure()
plt.scatter(x, y)
# plt.plot([0, 70], [0.5, 3.5])
import numpy as np
from sklearn import linear_model
mode1 = linear_model.LinearRegression().fit(x, y)
print(mode1.coef_)  # 模块系数
print(mode1.intercept_)  # 模块截距
x1 = np.arange(10, 70, 1)
y1 = 0.04396146*x1+0.58426124
plt.plot(x1, y1)
plt.show()