Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. Select appropriate data set for your experiment and draw graphs.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Let's predict the 'sepal length' based on 'sepal width' for simplicity
X = data[['sepal width (cm)']]  # Independent variable (sepal width)
y = data['sepal length (cm)']   # Dependent variable (sepal length)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Locally Weighted Regression function
def locally_weighted_regression(X_train, y_train, X_test, tau=1.0):
    m = X_train.shape[0]
    y_pred = np.zeros(X_test.shape[0])

    for i in range(X_test.shape[0]):
        # Compute the weight matrix
        weights = np.exp(-np.sum((X_train - X_test.iloc[i]) ** 2, axis=1) / (2 * tau ** 2))

        # Create weighted X matrix and y vector
        X_weighted = X_train * weights[:, np.newaxis]
        y_weighted = y_train * weights

        # Calculate the regression coefficients using weighted least squares
        X_weighted_transpose = X_weighted.T
        theta = np.linalg.inv(X_weighted_transpose @ X_weighted) @ X_weighted_transpose @ y_weighted

        # Predict for the i-th test point
        y_pred[i] = np.dot(X_test.iloc[i], theta)

    return y_pred

# Perform Locally Weighted Regression
y_pred = locally_weighted_regression(X_train, y_train, X_test, tau=1.0)

# Visualize the results
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='red', label='Test data')
plt.plot(X_test, y_pred, color='green', label='Locally Weighted Regression')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.title('Locally Weighted Regression on Iris Dataset')
plt.show()

