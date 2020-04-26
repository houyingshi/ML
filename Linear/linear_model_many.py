import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn import  metrics
import numpy as np

data = pd.read_csv('Advertising.csv')


sns.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=7, aspect=0.8, kind='reg')
plt.show()

feature_cols = ['TV', 'radio','newspaper']
X=data[feature_cols]
Y = data['sales']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1)
line = LinearRegression()
line.fit(X_train, Y_train)
for item in zip(feature_cols, line.coef_):
    print(item)
Y_pred = line.predict(X_test)
print(Y_pred)

sum_mean = 0
for i in range(len(Y_pred)):
    sum_mean+= (Y_pred[i] - Y_test.values[i])**2
sum_error = np.sqrt(sum_mean/50)
print('RMSE  is  ', sum_error)

plt.figure()
plt.plot(range(len(Y_pred)), Y_pred, 'b', label = 'predict')
plt.plot(range(len(Y_pred)), Y_test, 'r', label = 'test')
plt.legend(loc='upper right')
plt.xlabel('the number of sales')
plt.ylabel('value of sales')
plt.show()

