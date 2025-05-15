Program 2:

Develop a program to Load a dataset with at least two numerical columns (e.g., Iris, Titanic). Plot a scatter plot of two variables and calculate their Pearson correlation coefficient. Write a program to compute the covariance and correlation matrix for a dataset. Visualize the correlation matrix using a heatmap to know which variables have strong positive/negative correlations.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (Replace 'titanic.csv' with your actual dataset)
data = pd.read_csv('titanic.csv')

# Select numerical columns for correlation analysis
numerical_columns = ['Age', 'Fare']  # Modify as per dataset
df = data[numerical_columns].dropna()  # Drop missing values

# Compute Correlation Matrix
corr_matrix = df.corr()
print("Correlation Matrix:\n", corr_matrix)

# Compute Covariance Matrix
cov_matrix = df.cov()
print("\nCovariance Matrix:\n", cov_matrix)

# Scatterplot to visualize relationships
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df['Age'], y=df['Fare'])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Scatter Plot: Age vs Fare')
plt.show()

# Correlation Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()


