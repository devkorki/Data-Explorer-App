import pandas as pd

def profile_kmeans_clusters(kmeans_result_path: str, output_path: str):
    """
    Generates descriptive statistics for each KMeans cluster.

    Parameters:
    - kmeans_result_path: Path to the CSV containing KMeans clustering results and features.
    - output_path: Path to save the cluster profile summaries.
    """
    df = pd.read_csv(kmeans_result_path)

    if "kmeans_cluster" not in df.columns:
        raise ValueError("'kmeans_cluster' column not found in the KMeans results.")

    # Drop non-numeric columns except cluster
    ignore_cols = ["client_id", "persona", "pca_1", "pca_2"]
    numeric_cols = [col for col in df.columns if col not in ignore_cols and pd.api.types.is_numeric_dtype(df[col])]

    # Group by cluster and compute mean and std
    cluster_profiles = df.groupby("kmeans_cluster")[numeric_cols].agg(["mean", "std"])

    # Flatten column MultiIndex
    cluster_profiles.columns = [f"{col}_{stat}" for col, stat in cluster_profiles.columns]

    # Save output
    cluster_profiles.to_csv(output_path)
    print(f"✅ KMeans cluster profiles saved to: {output_path}")
