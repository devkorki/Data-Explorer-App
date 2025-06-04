import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def find_optimal_k(df, numeric_cols, k_range=range(2, 11), plot=True):
    """
    Finds the optimal k for KMeans using silhouette score.
    """
    df_clean = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
    X = StandardScaler().fit_transform(df_clean)

    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        print(f"k={k} → silhouette score={score:.4f}")

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(list(k_range), silhouette_scores, marker='o')
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score vs. K")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("output/kmeans_silhouette_plot.png")
        plt.show()

    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"\n✅ Best k: {best_k} with silhouette score: {max(silhouette_scores):.4f}")
    return best_k
