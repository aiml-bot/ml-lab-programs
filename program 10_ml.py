Program 10:

Develop a program to implement k-means clustering using Wisconsin Breast Cancer data set and visualize the clustering result.

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------
# 1. Load the Wisconsin Breast Cancer Dataset from "data.csv"
# ------------------------------------------------------------------------
data = pd.read_csv("data.csv")  # Ensure "data.csv" is in the same directory
print("Original Data Shape:", data.shape)
print(data.head())

# ---------------------------------
# 2. Data Preprocessing
# ---------------------------------
# Drop ID column if it exists
if "id" in data.columns:
    data.drop(columns=["id"], inplace=True)

# Convert diagnosis column to numeric (Malignant: 1, Benign: 0)
if "diagnosis" in data.columns:
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

# Separate features and labels (if available)
if "diagnosis" in data.columns:
    X = data.drop(columns=["diagnosis"])  # Features
    true_labels = data["diagnosis"].values  # Actual class labels
else:
    X = data.copy()
    true_labels = None

# ------------------------------
# 3. Feature Scaling
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------
# 4. Apply K-Means Clustering
# -------------------------------------

# Use 2 clusters (Benign & Malignant) and set n_init='auto' for compatibility
kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X_scaled)
data["Cluster"] = clusters  # Store cluster assignments in DataFrame


# -----------------------------------------------------------------
# 5. (Optional) Compare Clusters with Actual Diagnosis
# -----------------------------------------------------------------

if true_labels is not None:
    # Since K-Means assigns labels arbitrarily, check both alignments
    acc1 = np.mean(clusters == true_labels)
    acc2 = np.mean((1 - clusters) == true_labels)
    best_accuracy = max(acc1, acc2)
    print(f"\nBest Clustering Accuracy: {best_accuracy:.4f}")

# ----------------------------------------------------
# 6. Visualization using PCA (2D Projection)
# ----------------------------------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="coolwarm", alpha=0.8)
plt.title("K-Means Clustering on Wisconsin Breast Cancer Data (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.show()

 




