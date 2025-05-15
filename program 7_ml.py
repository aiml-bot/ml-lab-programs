Program 7:

Develop a program to demonstrate the working of Linear Regression and Polynomial Regression. Use Boston Housing Dataset for Linear Regression and Auto MPG Dataset (for vehicle fuel efficiency prediction) for Polynomial Regression.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------
# Load Boston Housing Dataset Manually (Fix for load_boston())
# ------------------------------
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
boston_df = pd.read_csv(url)

# Selecting feature (RM = Average number of rooms) and target (MEDV = House Price)
X_boston_feature = boston_df[['rm']]
y_boston = boston_df['medv']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_boston_feature, y_boston, test_size=0.2, random_state=42)

# Linear Regression Model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Predictions
y_pred = linear_reg.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Results (Boston Housing):")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('House Price (MEDV)')
plt.title('Linear Regression on Boston Housing Dataset')
plt.legend()
plt.show()
