Program 8:

Develop a program to load the Titanic dataset. Split the data into training and test sets. Train a decision tree classifier. Visualize the tree structure. Evaluate accuracy, precision, recall, and F1-score.

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------------------
# Load the Titanic Dataset
# ------------------------------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Display first few rows
print("Dataset Preview:\n", data.head())

# ------------------------------
# Data Preprocessing
# ------------------------------
# Select relevant features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
data = data[features + ['Survived']]

# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Convert categorical variables to numeric
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})  # Male = 0, Female = 1
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})  # Encoding Embarked

# ------------------------------
# Splitting Data into Train & Test Sets
# ------------------------------
X = data.drop(columns=['Survived'])
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Train Decision Tree Classifier
# ------------------------------
dtree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
dtree.fit(X_train, y_train)

# ------------------------------
# Visualizing the Decision Tree
# ------------------------------
plt.figure(figsize=(15, 8))
plot_tree(dtree, feature_names=X.columns.tolist(), class_names=['Not Survived', 'Survived'], filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

# ------------------------------
# Model Evaluation
# ------------------------------
y_pred = dtree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display Metrics
print("\nDecision Tree Model Evaluation:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
