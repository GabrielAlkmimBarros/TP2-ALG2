import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from src.kmeans import KMeans

# Gera dados sintéticos
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Testa diferentes métricas
metricas = [
    ('euclidean', {}),
    ('manhattan', {}),
    ('minkowski', {'p': 3}),
    ('mahalanobis', {}),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (metrica, params) in enumerate(metricas):
    kmeans = KMeans(
        n_clusters=4, 
        random_state=42, 
        distance_metric=metrica,
        **params
    )
    labels = kmeans.fit_predict(X)
    
    print(f"\n{metrica.upper()}:")
    print(f"  Iterações: {kmeans.n_iter_}")
    print(f"  Inércia: {kmeans.inertia_:.2f}")
    
    axes[idx].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    axes[idx].scatter(
        kmeans.centroids_[:, 0], kmeans.centroids_[:, 1],
        c='red', marker='X', s=200, edgecolors='black', linewidths=2
    )
    axes[idx].set_title(f"K-Means - {metrica}")

plt.tight_layout()
plt.show()