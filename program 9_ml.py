Program 9:
Develop a program to implement the Naive Bayesian classifier considering Iris dataset for training. Compute the accuracy of the classifier, considering the test data.

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------
# Load the Iris Dataset
# ------------------------------
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels (0 = Setosa, 1 = Versicolor, 2 = Virginica)

# Convert dataset to DataFrame for better readability
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y

# Display dataset info
print("Dataset Preview:\n", df.head())

# ------------------------------
# Splitting Data into Train & Test Sets
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Train Naïve Bayes Classifier
# ------------------------------
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# ------------------------------
# Model Predictions
# ------------------------------
y_pred = nb_classifier.predict(X_test)

# ------------------------------
# Model Evaluation
# ------------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\nNaïve Bayes Classifier Accuracy: {accuracy:.4f}")

# Display Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Display Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Naïve Bayes Classifier")
plt.show()
