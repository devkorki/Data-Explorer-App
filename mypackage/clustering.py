import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
# import umap
import numpy as np


def run_umap(df, columns):
    raise NotImplementedError("UMAP functionality is currently disabled.")
# def run_tsne(df, columns):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(df[columns])
#     tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#     X_embedded = tsne.fit_transform(X_scaled)
#     df['tsne_1'] = X_embedded[:, 0]
#     df['tsne_2'] = X_embedded[:, 1]
#     return df

def run_dbscan(df, columns, eps=0.5, min_samples=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[columns])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    result = df[["client_id"]].copy()
    result["dbscan_cluster"] = db.labels_
    return result
def run_tsne(df, columns, perplexity=30):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[columns])
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_embedded = tsne.fit_transform(X_scaled)
    proj = df[["client_id"]].copy()
    proj["tsne_1"] = X_embedded[:, 0]
    proj["tsne_2"] = X_embedded[:, 1]
    return proj


def run_pca(df, columns):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[columns])
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]
    return df

def run_umap(df, columns):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[columns])
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    df['umap_1'] = X_umap[:, 0]
    df['umap_2'] = X_umap[:, 1]
    return df

# def run_umap(df, feature_cols):
#     reducer = UMAP(n_components=2, random_state=42)
#     embedding = reducer.fit_transform(df[feature_cols])
#     df['umap_1'] = embedding[:, 0]
#     df['umap_2'] = embedding[:, 1]
#     return df


def plot_2d_projection(df, x_col, y_col, label_col, output_file, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=label_col, palette='Set2', alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"✅ Saved: {output_file}")
