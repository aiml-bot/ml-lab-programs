Program 4:

Develop a program to load the Iris dataset. Implement the k-Nearest Neighbors (k-NN) algorithm for classifying flowers based on their features. Split the dataset into training and testing sets and evaluate the model using metrics like accuracy and F1-score. Test it for different values of 𝑘 (e.g., k=1,3,5) and evaluate the accuracy. Extend the k-NN algorithm to assign weights based on the distance of neighbors (e.g., 𝑤𝑒𝑖𝑔ℎ𝑡=1/𝑑2 ). Compare the performance of weighted k-NN and regular k-NN on a synthetic or real-world dataset.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Euclidean distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# k-NN algorithm (regular and weighted)
def knn(X_train, y_train, X_test, k=3, weighted=False):
    y_pred = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        
        if weighted:
            # Compute weights (1/d^2)
            weights = [1 / (distances[i]**2) if distances[i] != 0 else 1e-10 for i in k_indices]
            # Weighted majority voting
            class_votes = {}
            for label, weight in zip(k_nearest_labels, weights):
                if label not in class_votes:
                    class_votes[label] = 0
                class_votes[label] += weight
            y_pred.append(max(class_votes, key=class_votes.get))
        else:
            # Regular majority voting
            y_pred.append(Counter(k_nearest_labels).most_common(1)[0][0])
    return np.array(y_pred)

# Test the models for different k values and compare
k_values = [1, 3, 5]
results = {'k': [], 'Accuracy (Regular k-NN)': [], 'F1-Score (Regular k-NN)': [], 
           'Accuracy (Weighted k-NN)': [], 'F1-Score (Weighted k-NN)': []}

for k in k_values:
    # Regular k-NN
    y_pred_knn = knn(X_train, y_train, X_test, k=k, weighted=False)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    f1_knn = f1_score(y_test, y_pred_knn, average='macro')
    
    # Weighted k-NN
    y_pred_wknn = knn(X_train, y_train, X_test, k=k, weighted=True)
    accuracy_wknn = accuracy_score(y_test, y_pred_wknn)
    f1_wknn = f1_score(y_test, y_pred_wknn, average='macro')
    
    # Store results
    results['k'].append(k)
    results['Accuracy (Regular k-NN)'].append(accuracy_knn)
    results['F1-Score (Regular k-NN)'].append(f1_knn)
    results['Accuracy (Weighted k-NN)'].append(accuracy_wknn)
    results['F1-Score (Weighted k-NN)'].append(f1_wknn)

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Plot comparison of accuracy and F1-score for different k values
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy comparison
axes[0].plot(results_df['k'], results_df['Accuracy (Regular k-NN)'], label='Regular k-NN', marker='o')
axes[0].plot(results_df['k'], results_df['Accuracy (Weighted k-NN)'], label='Weighted k-NN', marker='o')
axes[0].set_title('Accuracy Comparison')
axes[0].set_xlabel('k')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

# F1-Score comparison
axes[1].plot(results_df['k'], results_df['F1-Score (Regular k-NN)'], label='Regular k-NN', marker='o')
axes[1].plot(results_df['k'], results_df['F1-Score (Weighted k-NN)'], label='Weighted k-NN', marker='o')
axes[1].set_title('F1-Score Comparison')
axes[1].set_xlabel('k')
axes[1].set_ylabel('F1-Score (Macro)')
axes[1].legend()

plt.tight_layout()
plt.show()
