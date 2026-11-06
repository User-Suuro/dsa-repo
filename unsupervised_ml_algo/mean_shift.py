import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, estimate_bandwidth

# Load the Digits dataset
digits = load_digits()
X = digits.data    # Data (features of digit images)
y = digits.target  # True labels (actual digit values)

print("Data shape:", X.shape)  # Should be (1797, 64)
print("Unique labels:", np.unique(y))

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2 components using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the reduced data to get an initial sense of clustering
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.6)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Digits Dataset Reduced with PCA")
plt.colorbar(label="True Digit Labels")
plt.show()

# Estimate the bandwidth for Mean-Shift
bandwidth = estimate_bandwidth(X_pca, quantile=0.2, n_samples=500)
print("Estimated bandwidth:", bandwidth)

# Apply Mean-Shift clustering
mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
mean_shift.fit(X_pca)

# Retrieve cluster labels and cluster centers
labels = mean_shift.labels_
cluster_centers = mean_shift.cluster_centers_
print("Number of clusters found:", len(np.unique(labels)))

# Plot the clusters
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='plasma', s=50, alpha=0.6)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='red',
            marker='X', label='Cluster Centers')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Mean-Shift Clustering on Digits Dataset")
plt.legend()
plt.colorbar(label="Cluster Labels")
plt.show()