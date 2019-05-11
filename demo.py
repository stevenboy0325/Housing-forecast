import numpy as np
import pandas as pd
data = pd.read_csv('./house_price.csv')
data1 = data.dropna()
data2 = pd.get_dummies(data1[['dist', 'floor']])
pd.set_option('display.max_columns', None)
data3 = data2.drop(['dist_shijingshan', 'floor_high'], axis=1)
data4 = pd.concat([data3, data1[['roomnum', 'halls', 'AREA', 'subway', 'school', 'price']]], axis=1)
# # print(data.info())
# print(data2)
# print(data3)
# print(data4)
x = data4.iloc[:, :-1]
y = data4.iloc[:, -1:]
# print(y)

from sklearn import linear_model
from sklearn.model_selection import train_test_split
x_train, x_text, y_train, y_text = train_test_split(x, y, test_size=0.3, random_state=42)
mode1 = linear_model.LinearRegression().fit(x_train, y_train)
result = mode1.predict(np.array([[0, 0, 0, 0, 0, 0, 0, 2, 1, 60, 1, 1]]))
print(result)
print(mode1.coef_)  # 模块系数
print(mode1.intercept_)  # 模块截距
print(mode1.score(x_text, y_text))