import pandas as pd
import streamlit as st
import os

def profile_clusters(feature_matrix_path="output/user_feature_matrix.csv",
                     clustering_result_path="output/dbscan_results.csv",
                     output_path="output/cluster_profiles.csv"):
    """Generate summary statistics for each DBSCAN cluster."""
    if not os.path.exists(feature_matrix_path):
        st.error(f"Feature matrix not found at: {feature_matrix_path}")
        return

    if not os.path.exists(clustering_result_path):
        st.error(f"DBSCAN result file not found at: {clustering_result_path}")
        return

    # Load data
    features = pd.read_csv(feature_matrix_path)
    clusters = pd.read_csv(clustering_result_path)

    # Merge to get cluster per user
    df = features.merge(clusters[['client_id', 'dbscan_cluster']], on="client_id", how="inner")

    # Keep only numeric columns for profiling
    exclude_cols = {"client_id", "favorite_category", "preferred_price_tier", "persona"}
    numeric_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

    # Group by cluster and calculate means and counts
    profile = df.groupby("dbscan_cluster")[numeric_cols].mean().round(2)
    profile["num_users"] = df.groupby("dbscan_cluster")["client_id"].count()

    # Optional: Add most common persona in each cluster
    if "persona" in df.columns:
        top_personas = df.groupby("dbscan_cluster")["persona"] \
            .agg(lambda x: x.value_counts().idxmax())
        profile["dominant_persona"] = top_personas

    # Save output
    profile.to_csv(output_path)
    st.success(f"Cluster profiles saved to: {output_path}")
    st.dataframe(profile)
    return profile
