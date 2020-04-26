import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model


def get_data(file_name):
    """
    Args:
        file_name:data file name

    Returns:
        truple with x and y
    """
    data = pd.read_csv(file_name)
    X_parameter = []
    Y_parameter = []
    for single_square_feet, single_price_value in zip(data['square_feet'], data['price']):
        X_parameter.append([float(single_square_feet)])
        Y_parameter.append(float(single_price_value));

    return X_parameter,Y_parameter

def linear_model_main(x, y, in_x):
    """
    Args:
        x: train data
        y: train data
        in_x: test data

    Returns:
        prediction
    """
    line = linear_model.LinearRegression()
    line.fit(x, y)
    out = line.predict([[in_x]])
    predictions = {}
    predictions["intercept"] = line.intercept_
    predictions["coefficient"] = line.coef_
    predictions['predicted_value'] = out
    return predictions

def show_linear_line(xs, ys):
    """

    Args:
        xs: train data
        ys: train data

    """
    line = linear_model.LinearRegression()
    line.fit(xs, ys)
    plt.scatter(xs, ys, color='blue')
    plt.plot(xs, line.predict(xs), color='red', linewidth =4)
    plt.xticks(())
    plt.yticks(())
    plt.show()

X,Y = get_data('input_data.csv')
print((X,Y))
result = linear_model_main(X, Y, 120)
print("Intercept value ", result['intercept'])
print("coefficient", result['coefficient'])
print("predicted value ", result["predicted_value"])
show_linear_line(X, Y)

