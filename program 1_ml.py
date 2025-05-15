
Program 1:

Develop a program to Load a dataset and select one numerical column. Compute mean, median, mode, standard deviation, variance, and range for a given numerical column in a dataset. Generate a histogram and boxplot to understand the distribution of the data. Identify any outliers in the data using IQR. Select a categorical variable from a dataset. Compute the frequency of each category and display it as a bar chart or pie chart.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset 
data = pd.read_csv('titanic.csv')

# Select a numerical column
column = 'Survived'  
stats = {
    'Mean': data[column].mean(),
    'Median': data[column].median(),
    'Mode': data[column].mode()[0],
    'Std Dev': data[column].std(),
    'Variance': data[column].var(),
    'Range': data[column].max() - data[column].min()
}
print(stats)

# Histogram and Boxplot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
data[column].hist()
plt.title('Histogram')

plt.subplot(1, 2, 2)
sns.boxplot(x=data[column])
plt.title('Boxplot')
plt.show()

# Outlier Detection using IQR
Q1 = data[column].quantile(0.25)
Q3 = data[column].quantile(0.75)

IQR = Q3 - Q1
outliers = data[(data[column] < Q1 - 1.5 * IQR) | (data[column] > Q3 + 1.5 * IQR)]
print("Outliers:\n", outliers)

# Categorical Analysis
cat_col = 'Embarked'  # actual categorical column name
category_counts = data[cat_col].value_counts()
category_counts.plot(kind='bar', title='Bar Chart of Categories')
plt.show()

