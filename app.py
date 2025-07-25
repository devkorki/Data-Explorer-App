import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from mypackage.clustering import run_dbscan
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go


from mypackage.profiling import create_user_segments, build_feature_matrix
from mypackage.clustering import run_tsne, run_pca, run_umap, plot_2d_projection
from mypackage.feature_engineering import run_feature_engineering
from mypackage.similarity import get_most_similar_users

from mypackage.cluster_profiling import profile_clusters
from mypackage.deep_learning import get_autoencoder_embedding

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score




# Load user ID lists
relevant_clients = np.load("input/relevant_clients.npy", allow_pickle=True).tolist()
active_clients = np.load("input/active_clients.npy", allow_pickle=True).tolist()


# ====== CONFIG ======
INPUT_DIR = "input"
OUTPUT_DIR = "output"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)



# ====== REQUIRED INPUT FILES CHECK ======
REQUIRED_FILES = [
    "relevant_clients.npy",
    "active_clients.npy",
    "add_to_cart.parquet",
    "remove_from_cart.parquet",
    "product_buy.parquet",
    "search_query.parquet",
    "page_visit.parquet",
    "product_properties.parquet"
]

missing_files = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(INPUT_DIR, f))]

if missing_files:
    st.error("🚫 The following required files are missing from the 'input' folder:")
    for f in missing_files:
        st.markdown(f"- `{f}`")
    st.error("Please add them and refresh the page.")
    st.stop()

# ====== STREAMLIT UI ======
#st.set_page_config(page_title="Main Tool", layout="wide")

st.set_page_config(page_title="Data Explorer", layout="wide")
st.title("Interactive Data Pipeline & Clustering UI")


# st.sidebar.title("User Selection")
# user_choice = st.sidebar.radio(
#     "Which user list should be used?",
#     ("Relevant Clients", "Active Clients")
# )

# Sidebar toggle
st.sidebar.title("User Filter Options")
user_choice = st.sidebar.radio("Select user set:", ("Relevant Clients", "Active Clients"))

# Final list to be used
selected_clients = active_clients if user_choice == "Active Clients" else relevant_clients
# ====== UTILS ======
@st.cache_data
def load_clients():
    return np.load(os.path.join(INPUT_DIR, "relevant_clients.npy"), allow_pickle=True)

def load_parquet_filtered(file, client_ids):
    df = pd.read_parquet(os.path.join(INPUT_DIR, file))
    return df[df['client_id'].isin(client_ids)]


# ====== HISTOGRAM AND HEATMAP ======

def plot_histogram(df, column, filename, title, bins):
    fig, ax = plt.subplots(figsize=(5, 3))
    df[column].value_counts().hist(bins=bins, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Interaction Count")
    ax.set_ylabel("User Count")
    st.pyplot(fig, use_container_width=True)
    fig.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

def plot_heatmap(df, file_label):
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.dayofweek
    heatmap_data = df.groupby(['day', 'hour']).size().unstack().fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='YlGnBu', linewidths=.5, ax=ax)
    ax.set_title(f"{file_label} Heatmap")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Day (0=Mon)")
    st.pyplot(fig, use_container_width=True)
    fig.figure.savefig(os.path.join(PLOT_DIR, f"{file_label}_heatmap.png"))
    plt.close()


# ====== TABS ======

# tab1, tab2, tab3= st.tabs(["Data Statistics","Main Pipeline", "Similar Users Comparison"])
tab1, tab2, tab3, tab4 = st.tabs(["Data Statistics","Main Pipeline", "Similar Users", "KMeans Clustering"])

with tab2:
    # --- Step 1: Load Clients ---
    st.header("Sample Relevant Clients")
    #clients_all = load_clients()
    clients_all = selected_clients

    st.write(f"Total clients available: {len(clients_all)}")

    chunk_size = st.number_input("Chunk size:", min_value=1000, max_value=len(clients_all), value=10000, step=1000)
    randomize = st.checkbox("Randomize", value=True)
    seed = st.number_input("Seed:", value=42, step=1)

    if st.button("Sample Clients"):
        with st.spinner("Sampling clients..."):
            np.random.seed(seed)
            sample = np.random.choice(clients_all, chunk_size, replace=False) if randomize else clients_all[:chunk_size]
            st.session_state.client_ids = sample
            st.session_state.seed = seed

            st.success(f"Sampled {chunk_size} clients")

    # --- Step 1.5: Filter and Cache All Files ---
    st.header("Filter Files for Sampled Clients")
    files = {
        "Add to Cart": "add_to_cart.parquet",
        "Remove from Cart": "remove_from_cart.parquet",
        "Product Buy": "product_buy.parquet",
        "Search Query": "search_query.parquet",
        "Page Visit": "page_visit.parquet"
    }

    # if st.button("Filter and Save All Files"):
    #     if 'client_ids' not in st.session_state:
    #         st.warning("Please sample clients first.")
    #     else:
    #         with st.spinner("Filtering and saving files..."):
    #             for label, file in files.items():
    #                 file_label = label.replace(" ", "_").lower()
    #                 output_path = os.path.join(OUTPUT_DIR, f"{file_label}_filtered.csv")
    #                 if os.path.exists(output_path):
    #                     st.info(f"✅ {label}: already cached.")
    #                 else:
    #                     df = load_parquet_filtered(file, st.session_state.client_ids)
    #                     df.to_csv(output_path, index=False)
    #                     st.success(f"📁 {label}: saved {len(df)} rows to `{output_path}`")
    
    if st.button("Filter and Save All Files"):
        if 'client_ids' not in st.session_state:
            st.warning("Please sample clients first.")
        else:
            seed_str = f"_seed_{seed}"
            with st.spinner("Filtering and saving files..."):
                for label, file in files.items():
                    file_label = label.replace(" ", "_").lower()
                    output_path = os.path.join(OUTPUT_DIR, f"{file_label}{seed_str}.csv")
                    if os.path.exists(output_path):
                        st.info(f"✅ {label}: using cached file for seed {seed}")
                    else:
                        df = load_parquet_filtered(file, st.session_state.client_ids)
                        # if 'sku' in df.columns:
                        #     props = pd.read_parquet(os.path.join(INPUT_DIR, "product_properties.parquet"))
                        #     df = df.merge(props[["sku", "category", "price"]], on="sku", how="left")
                        df.to_csv(output_path, index=False)
                        st.success(f"📁 {label}: saved {len(df)} rows to `{output_path}`")


    # --- Step 2: Choose File + EDA ---
    st.header("Select File and EDA Options")
    file_choice = st.selectbox("Choose interaction file:", list(files.keys()))
    eda_options = st.multiselect("Choose EDA steps:", ["Histogram", "Heatmap"], label_visibility="collapsed")
    if "Histogram" in eda_options:
        bins = st.number_input("Histogram Bins", 5, 100, 50)

    if st.button("Run EDA"):
        if 'client_ids' not in st.session_state:
            st.warning("Please sample clients first.")
        else:
            file_path = files[file_choice]
            file_label = file_choice.replace(" ", "_").lower()
            output_path = os.path.join(OUTPUT_DIR, f"{file_label}_filtered.csv")

            with st.spinner("Running EDA..."):
                if os.path.exists(output_path):
                    st.info(f"✅ Using cached filtered file: `{output_path}`")
                    df = pd.read_csv(output_path)
                else:
                    df = load_parquet_filtered(file_path, st.session_state.client_ids)
                    df.to_csv(output_path, index=False)
                    st.success(f"Saved filtered data to `{output_path}`")

                st.write(f"Filtered rows: {len(df)}")

                if "Histogram" in eda_options:
                    plot_histogram(df, 'client_id', f"{file_label}_hist.png", f"{file_choice} Histogram", bins)
                if "Heatmap" in eda_options and 'timestamp' in df.columns:
                    plot_heatmap(df, file_label)

    # --- Step 2.5: Generate Personas ---
    # st.header("Generate Personas")
    # if st.button("Create Personas"):
    #     with st.spinner("Assigning personas and saving file..."):
    #         visits = pd.read_csv("output/page_visit_filtered.csv")
    #         carts = pd.read_csv("output/add_to_cart_filtered.csv")
    #         removes = pd.read_csv("output/remove_from_cart_filtered.csv")
    #         buys = pd.read_csv("output/product_buy_filtered.csv")
    #         create_user_segments(visits, carts, removes, buys)
    #         st.success("✅ Personas saved to `output/user_personas.csv`")
    
    st.header("Generate Personas")
    if st.button("Create Personas"):
        if 'client_ids' not in st.session_state:
            st.warning("Please sample clients first.")
        else:
            seed_str = f"_seed_{seed}"
            with st.spinner("Assigning personas and saving file..."):
                try:
                    # visits = pd.read_csv(f"output/page_visit_seed_{seed_str}.csv")
                    # carts = pd.read_csv(f"output/add_to_cart_seed_{seed_str}.csv")
                    # removes = pd.read_csv(f"output/remove_from_cart_seed_{seed_str}.csv")
                    # buys = pd.read_csv(f"output/product_buy_seed_{seed_str}.csv")
                    
                    visits = pd.read_csv(f"output/page_visit{seed_str}.csv")
                    carts = pd.read_csv(f"output/add_to_cart{seed_str}.csv")
                    removes = pd.read_csv(f"output/remove_from_cart{seed_str}.csv")
                    buys = pd.read_csv(f"output/product_buy{seed_str}.csv")


                    create_user_segments(visits, carts, removes, buys)
                    st.success(f"✅ Personas saved to `output/user_personas.csv` (based on seed {seed})")
                except FileNotFoundError as e:
                    st.error(f" Missing one or more filtered files for seed {seed}. Please run filtering first.")
                    st.stop()


    # --- Step 3: Feature Engineering ---
    st.header("Run Feature Engineering")
    # if st.button("Run Feature Engineering"):
    #     with st.spinner("Generating user feature matrix..."):
    #         output_path = run_feature_engineering()
    #         st.success(f"✅ Feature matrix saved at: {output_path}")


    # if st.button("Run Feature Engineering"):
    #     with st.spinner("Generating user feature matrix..."):
    #         output_path = run_feature_engineering()
    #         df_feat = pd.read_csv(output_path)

    #         # Try to load personas as well (optional)
    #         try:
    #             personas = pd.read_csv("output/user_personas.csv")
    #             df_feat = df_feat.merge(personas, on="client_id", how="left")
    #             df_feat["persona"] = df_feat["persona"].fillna("unknown")
    #         except Exception:
    #             df_feat["persona"] = "unknown"

    #         st.session_state.features = df_feat
    #         st.session_state.profiles = df_feat[["client_id", "persona"]]
    #         st.success(f"✅ Feature matrix loaded and saved at: {output_path}")
    @st.cache_data(show_spinner="Running feature engineering...")
    def cached_feature_engineering():
        # output_path = run_feature_engineering()
        output_path = run_feature_engineering(seed=seed)

        df = pd.read_csv(output_path)

        # Merge personas if available
        try:
            personas = pd.read_csv("output/user_personas.csv")
            df = df.merge(personas, on="client_id", how="left")
            df["persona"] = df["persona"].fillna("unknown")
        except Exception:
            df["persona"] = "unknown"

        return df   

    if st.button("Run Feature Engineering"):
        with st.spinner("Loading from cache or computing..."):
            df_feat = cached_feature_engineering()
            st.session_state.features = df_feat
            st.session_state.profiles = df_feat[["client_id", "persona"]]
            st.success("✅ Feature matrix cached or recomputed successfully.")


    # --- Step 4: Clustering & Projection ---
    st.header("Clustering & Projection")
    #projection_methods = st.multiselect("Choose dimensionality reduction methods:", ["t-SNE", "PCA"])
    projection_methods = st.multiselect("Choose clustering / projection methods:", ["t-SNE", "PCA", "DBSCAN"])

    #perplexity = st.number_input("t-SNE perplexity:", min_value=5, max_value=100, value=30, step=1)
    if "t-SNE" in projection_methods:
        perplexity = st.number_input("t-SNE Perplexity", 5, 100, 30)



    if "DBSCAN" in projection_methods:
        eps = st.number_input("DBSCAN eps:", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
        min_samples = st.number_input("Min samples:", min_value=1, max_value=100, value=10, step=1)

    # if st.button("Run Projection"):
    #     if 'features' not in st.session_state:
    #         try:
    #             df_feat = pd.read_csv(os.path.join("output", "user_feature_matrix.csv"))
    #             personas = pd.read_csv("output/user_personas.csv")
    #             df_feat = df_feat.merge(personas, on="client_id", how="left")
    #             df_feat["persona"] = df_feat["persona"].fillna("unknown")
    #             st.session_state.features = df_feat
    #             st.session_state.profiles = df_feat[["client_id", "persona"]]
    #             st.success("Loaded features and personas.")
    #         except Exception as e:
    #             st.warning("Please generate the feature matrix and personas first.")
    #             st.stop()

    #     with st.spinner("Running projections..."):
    #         df = st.session_state.features.copy()
    #         cols = [c for c in df.columns if c != "client_id" and np.issubdtype(df[c].dtype, np.number)]
    #         df = df.dropna(subset=cols)

    #         for method in projection_methods:
    #             if method == "t-SNE":
    #                 proj = run_tsne(df.copy(), cols, perplexity=perplexity)
    #                 x_col, y_col = "tsne_1", "tsne_2"
    #             elif method == "PCA":
    #                 proj = run_pca(df.copy(), cols)
    #                 x_col, y_col = "pca_1", "pca_2"

    #             # df_merged = proj.merge(st.session_state.profiles, on="client_id")
    #             df_merged = proj.merge(st.session_state.profiles, on="client_id", how="left")
    #             if "persona" not in df_merged.columns:
    #                 df_merged["persona"] = "unknown"

    #             out_path = os.path.join(PLOT_DIR, f"{method.lower()}_projection.png")
    #             plot_2d_projection(df_merged, x_col, y_col, "persona", out_path, f"{method} Projection")
    #             st.image(out_path)
    #             st.success(f"{method} projection completed and saved")


    #             if "DBSCAN" in projection_methods:
    #                 X = StandardScaler().fit_transform(df[cols])
    #                 pca = PCA(n_components=2)
    #                 X_pca = pca.fit_transform(X)
    #                 df["pca_1"], df["pca_2"] = X_pca[:, 0], X_pca[:, 1]

    #                 db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    #                 df["dbscan_cluster"] = db.labels_

    #                 fig, ax = plt.subplots(figsize=(7, 4))
    #                 sns.scatterplot(data=df, x="pca_1", y="pca_2", hue="dbscan_cluster", palette="tab20", alpha=0.6, ax=ax)
    #                 ax.set_title("DBSCAN Clusters visualized with PCA")
    #                 st.pyplot(fig, use_container_width=True)
    #                 df.to_csv(os.path.join(OUTPUT_DIR, "dbscan_results.csv"), index=False)

    if st.button("Run Projection"):
        if 'features' not in st.session_state:
            try:
                df_feat = pd.read_csv(os.path.join("output", "user_feature_matrix.csv"))
                personas = pd.read_csv("output/user_personas.csv")
                df_feat = df_feat.merge(personas, on="client_id", how="left")
                df_feat["persona"] = df_feat["persona"].fillna("unknown")
                st.session_state.features = df_feat
                st.session_state.profiles = df_feat[["client_id", "persona"]]
                st.success("Loaded features and personas.")
            except Exception as e:
                st.warning("Please generate the feature matrix and personas first.")
                st.stop()

        with st.spinner("Running projections..."):
            df = st.session_state.features.copy()
            cols = [c for c in df.columns if c not in ["client_id", "persona"] and np.issubdtype(df[c].dtype, np.number)]
            df = df.dropna(subset=cols)

            for method in projection_methods:
                if method == "t-SNE":
                    proj = run_tsne(df.copy(), cols, perplexity=perplexity)
                    x_col, y_col = "tsne_1", "tsne_2"
                    df_merged = proj.merge(st.session_state.profiles, on="client_id", how="left")
                    df_merged["persona"] = df_merged.get("persona", "unknown")
                    out_path = os.path.join(PLOT_DIR, "tsne_projection.png")
                    plot_2d_projection(df_merged, x_col, y_col, "persona", out_path, "t-SNE Projection")
                    st.image(out_path)

                elif method == "PCA":
                    proj = run_pca(df.copy(), cols)
                    x_col, y_col = "pca_1", "pca_2"
                    df_merged = proj.merge(st.session_state.profiles, on="client_id", how="left")
                    df_merged["persona"] = df_merged.get("persona", "unknown")
                    out_path = os.path.join(PLOT_DIR, "pca_projection.png")
                    plot_2d_projection(df_merged, x_col, y_col, "persona", out_path, "PCA Projection")
                    st.image(out_path)

            if "DBSCAN" in projection_methods:
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.dropna(subset=cols)  # 'cols' is your selected numeric columns  
                X = StandardScaler().fit_transform(df[cols])
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                df["pca_1"], df["pca_2"] = X_pca[:, 0], X_pca[:, 1]

                db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
                df["dbscan_cluster"] = db.labels_

                fig, ax = plt.subplots(figsize=(7, 4))
                sns.scatterplot(data=df, x="pca_1", y="pca_2", hue="dbscan_cluster", palette="tab20", alpha=0.6, ax=ax)
                ax.set_title("DBSCAN Clusters visualized with PCA")
                st.pyplot(fig, use_container_width=True)

                df.to_csv(os.path.join(OUTPUT_DIR, "dbscan_results.csv"), index=False)

    # --- Cluster Profiling ---
    st.header("Cluster Profiling")

    if st.button("Generate Cluster Profiles"):
        with st.spinner("Profiling clusters..."):
            from mypackage.cluster_profiling import profile_clusters
            profile_clusters(
                feature_matrix_path=os.path.join(OUTPUT_DIR, "user_feature_matrix.csv"),
                clustering_result_path=os.path.join(OUTPUT_DIR, "dbscan_results.csv"),
                output_path=os.path.join(OUTPUT_DIR, "cluster_profiles.csv")
            )
            
            
            
    ##--------------DEEP LEARNING---------------
    st.header("Deep Learning")

    if st.button("Run Deep Learning Embedding"):
        df = st.session_state.features.copy()
        cols = [c for c in df.columns if c not in ["client_id", "persona"] and np.issubdtype(df[c].dtype, np.number)]

        embed_df = get_autoencoder_embedding(df, cols)
        st.session_state.deep_embedding = embed_df
        st.dataframe(embed_df.head())
        
        
        
# # --- Step 5: DBSCAN Clustering ---
# st.header("DBSCAN Clustering")
# eps = st.number_input("DBSCAN eps:", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
# min_samples = st.number_input("Min samples:", min_value=1, max_value=50, value=5, step=1)

# if st.button("Run DBSCAN"):
#     if 'features' not in st.session_state:
#         st.warning("Please run feature engineering first.")
#         st.stop()

#     with st.spinner("Running DBSCAN..."):
#         df = st.session_state.features.copy()
#         cols = [c for c in df.columns if c != "client_id" and np.issubdtype(df[c].dtype, np.number)]
#         df = df.dropna(subset=cols)
#         clusters = run_dbscan(df.copy(), cols, eps=eps, min_samples=min_samples)
#         merged = clusters.merge(st.session_state.profiles, on="client_id", how="left")

#         out_path = os.path.join(PLOT_DIR, "dbscan_projection.png")
#         plot_2d_projection(merged, "client_id", "dbscan_cluster", "persona", out_path, "DBSCAN Clustering")
#         st.image(out_path)
#         st.success("DBSCAN clustering completed and saved")

# # --- Step 5: DBSCAN Clustering ---
# st.header("DBSCAN Clustering")

# eps = st.number_input("DBSCAN eps:", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
# min_samples = st.number_input("Min samples:", min_value=1, max_value=100, value=10, step=1)

# if st.button("Run DBSCAN"):
#     try:
#         df = st.session_state.features.copy()
#         numeric_cols = [c for c in df.columns if c not in ['client_id', 'persona'] and np.issubdtype(df[c].dtype, np.number)]
#         X = StandardScaler().fit_transform(df[numeric_cols])

#         # PCA for 2D visualization
#         pca = PCA(n_components=2)
#         X_pca = pca.fit_transform(X)
#         df["pca_1"], df["pca_2"] = X_pca[:, 0], X_pca[:, 1]

#         # DBSCAN clustering
#         db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
#         df["dbscan_cluster"] = db.labels_

#         # Plot
#         fig, ax = plt.subplots(figsize=(10, 6))
#         sns.scatterplot(data=df, x="pca_1", y="pca_2", hue="dbscan_cluster", palette="tab20", alpha=0.6, ax=ax)
#         ax.set_title("DBSCAN Clusters visualized with PCA")
#         st.pyplot(fig)

#         # Optional: Save plot or data
#         df.to_csv(os.path.join(OUTPUT_DIR, "dbscan_results.csv"), index=False)

#     except Exception as e:
#         st.error(f"Error running DBSCAN: {e}")

# COSINE SIMILARITY COMPARISON

with tab3:
    st.header("Find Most Similar Clients")

 
    if st.button("Compare Most Similar Clients"):
        if "features" not in st.session_state:
            st.warning("Please run Feature Engineering first.")
            st.stop()
        else:
            df = st.session_state.features.copy()
            feature_cols = [c for c in df.columns if c not in ["client_id", "persona"] and np.issubdtype(df[c].dtype, np.number)]

            # Run similarity check
            result = get_most_similar_users(df, feature_cols, top_n=1)
            pair = result.iloc[0]

            st.subheader("Most Similar Pair")
            st.code(f"Client A: {int(pair.client_1)}\nClient B: {int(pair.client_2)}\nCosine similarity: {pair.similarity:.4f}")

            client_a = df[df["client_id"] == pair.client_1].iloc[0]
            client_b = df[df["client_id"] == pair.client_2].iloc[0]

            # Radar chart
            top_features = feature_cols[:10]  # Optional: top N features or all
            values_a = client_a[top_features].values
            values_b = client_b[top_features].values

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=values_a, theta=top_features, fill='toself', name=f"Client {int(pair.client_1)}"))
            fig.add_trace(go.Scatterpolar(r=values_b, theta=top_features, fill='toself', name=f"Client {int(pair.client_2)}"))
            fig.update_layout(title="Top Feature Comparison", showlegend=True, polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(fig, use_container_width=True)
            
  
# ## COSINE SIMILARITY BETWEEN CLIENTS
# if st.button("Find 2 Most Similar Clients"):
#     if 'features' not in st.session_state:
#         st.warning("⚠️ Please run Feature Engineering first.")
#     else:
#         df = st.session_state.features.copy()
#         feature_cols = [c for c in df.columns if c not in ['client_id', 'persona'] and np.issubdtype(df[c].dtype, np.number)]
        
#         result = get_most_similar_users(df, feature_cols, top_n=1)
#         row = result.iloc[0]

#         st.success("Most similar clients:")
#         st.write(f"Client A: `{row.client_1}`")
#         st.write(f"Client B: `{row.client_2}`")
#         st.write(f"Cosine similarity: `{row.similarity:.4f}`")
with tab1:
    # st.header("Basic Data Statistics for Filtered Files")

    # # Get list of available filtered CSVs
    # available_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith("_filtered.csv")]

    # if not available_files:
    #     st.warning("No filtered CSV files found in the output directory.")
    # else:
    #     selected_file = st.selectbox("Select a filtered file:", available_files)
    #     file_path = os.path.join(OUTPUT_DIR, selected_file)

    #     try:
    #         df = pd.read_csv(file_path)
    #         st.subheader("Preview of the Data")
    #         st.dataframe(df.head(100), use_container_width=True)

    #         st.subheader("Summary Statistics")
    #         st.write(df.describe(include='all'))

    #         st.subheader("Missing Values per Column")
    #         st.write(df.isnull().sum())

    #         st.subheader("Data Types")
    #         buffer = df.dtypes.reset_index()
    #         buffer.columns = ["Column", "Type"]
    #         st.dataframe(buffer)

    #     except Exception as e:
    #         st.error(f"Failed to load the file: {e}")
    
    st.header("Basic Data Statistics for Filtered Files")

    # Check that seed exists in session_state
    if 'seed' not in st.session_state:
        st.warning("Please define a seed by sampling clients first.")
        st.stop()

    # Use the seed from session
    seed = st.session_state.seed
    seed_str = f"_seed_{seed}.csv"

    # Filter files with the correct seed suffix
    available_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(seed_str)]

    if not available_files:
        st.warning(f"No filtered files found for seed {seed} in the output directory.")
    else:
        # Optional: create nicer display labels
        file_labels = [f.replace(seed_str, "") for f in available_files]
        selected_label = st.selectbox("Select a filtered file:", file_labels)
        selected_file = f"{selected_label}{seed_str}"
        file_path = os.path.join(OUTPUT_DIR, selected_file)

        try:
            df = pd.read_csv(file_path)
            st.subheader("Preview of the Data")
            st.dataframe(df.head(100), use_container_width=True)

            st.subheader("Summary Statistics")
            st.write(df.describe(include='all'))

            st.subheader("Missing Values per Column")
            st.write(df.isnull().sum())

            st.subheader("Data Types")
            buffer = df.dtypes.reset_index()
            buffer.columns = ["Column", "Type"]
            st.dataframe(buffer)

        except Exception as e:
            st.error(f"Failed to load the file: {e}")
        
    # try:
    #     df = pd.read_csv(file_path)
    #     st.subheader("Preview of the Data")
    #     st.dataframe(df.head(100), use_container_width=True)

    #     st.subheader("Summary Statistics")
    #     st.write(df.describe(include='all'))

    #     st.subheader("Missing Values per Column")
    #     st.write(df.isnull().sum())

    #     st.subheader("Data Types")
    #     buffer = df.dtypes.reset_index()
    #     buffer.columns = ["Column", "Type"]
    #     st.dataframe(buffer)

    #     # === Additional Analysis ===
    #     if "category" in df.columns:
    #         st.subheader("Top 10 Categories by Count")
    #         category_counts = df["category"].value_counts().head(10)
    #         st.bar_chart(category_counts)

    #         st.subheader("Average Price per Category (Top 10)")
    #         avg_price = df.groupby("category")["price"].mean().sort_values(ascending=False).head(10)
    #         st.bar_chart(avg_price)

    #     if "price" in df.columns:
    #         st.subheader("Price Distribution")
    #         fig, ax = plt.subplots()
    #         df["price"].dropna().hist(bins=30, ax=ax)
    #         ax.set_title("Price Distribution")
    #         ax.set_xlabel("Price")
    #         ax.set_ylabel("Frequency")
    #         st.pyplot(fig)

    #     if "sku" in df.columns:
    #         st.subheader("Top 10 SKUs by Frequency")
    #         top_skus = df["sku"].value_counts().head(10)
    #         st.bar_chart(top_skus)

    # except Exception as e:
    #         st.error(f"Failed to load the file: {e}")


            
with tab4:
    st.header("KMeans Clustering & Optimal K")

    if "features" not in st.session_state:
        st.warning("Please run feature engineering first.")
        st.stop()

    df = st.session_state.features.copy()
    numeric_cols = [c for c in df.columns if c not in ["client_id", "persona"] and np.issubdtype(df[c].dtype, np.number)]

    if st.button("Run Silhouette Score Analysis"):
        df_clean = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
        X = StandardScaler().fit_transform(df_clean)

        silhouette_scores = []
        k_range = range(2, 11)

        for k in k_range:
            model = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
            st.write(f"k = {k} → Silhouette Score: {score:.4f}")

        best_k = k_range[np.argmax(silhouette_scores)]
        st.session_state.best_k = best_k
        st.success(f"✅ Optimal number of clusters: {best_k}")

        # Plot silhouette score vs k
        fig, ax = plt.subplots()
        ax.plot(list(k_range), silhouette_scores, marker='o')
        ax.set_title("Silhouette Score vs. k")
        ax.set_xlabel("k")
        ax.set_ylabel("Score")
        st.pyplot(fig)

    if st.button("Run KMeans with Optimal k"):
        if "best_k" not in st.session_state:
            st.warning("Please run silhouette score analysis first.")
            st.stop()

        best_k = st.session_state.best_k
        df_clean = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
        X = StandardScaler().fit_transform(df_clean)

        with st.spinner(f"Running KMeans with k = {best_k}"):
            model = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
            labels = model.fit_predict(X)

            # Assign to df_clean to avoid index mismatch
            df_clean = df_clean.copy()
            df_clean["kmeans_cluster"] = labels

            # PCA for plotting
            pca = PCA(n_components=2)
            proj = pca.fit_transform(X)
            df_clean["pca_1"] = proj[:, 0]
            df_clean["pca_2"] = proj[:, 1]

            # Plot clusters
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_clean, x="pca_1", y="pca_2", hue="kmeans_cluster", palette="tab10", ax=ax)
            ax.set_title("KMeans Clusters (PCA)")
            st.pyplot(fig)

            # Save results
            df_clean.to_csv(os.path.join(OUTPUT_DIR, "kmeans_results.csv"), index=False)
            st.success("✅ KMeans clustering completed and saved.")
    # --- KMeans Cluster Profiling ---
    st.subheader("KMeans Cluster Profiling")

    if st.button("Generate KMeans Profiles"):
        with st.spinner("Profiling KMeans clusters..."):
            from mypackage.profile_kmeans_clusters import profile_kmeans_clusters
            profile_kmeans_clusters(
                kmeans_result_path=os.path.join(OUTPUT_DIR, "kmeans_results.csv"),
                output_path=os.path.join(OUTPUT_DIR, "kmeans_cluster_profiles.csv")
            )
            st.success("✅ KMeans cluster profiles saved to 'output/kmeans_cluster_profiles.csv'")
            df_profiles = pd.read_csv(os.path.join(OUTPUT_DIR, "kmeans_cluster_profiles.csv"))
            st.dataframe(df_profiles)
# with tab4:
#     st.header("KMeans Clustering & Optimal K")

#     if "features" not in st.session_state:
#         st.warning("Please run feature engineering first.")
#         st.stop()

#     df = st.session_state.features.copy()
#     numeric_cols = [c for c in df.columns if c not in ["client_id", "persona"] and np.issubdtype(df[c].dtype, np.number)]

#     # if st.button("Run Silhouette Score Analysis"):
#     #     from sklearn.preprocessing import StandardScaler

#     #     df_clean = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
#     #     X = StandardScaler().fit_transform(df_clean)

#     #     silhouette_scores = []
#     #     k_range = range(2, 11)

#     #     for k in k_range:
#     #         model = KMeans(n_clusters=k, random_state=42, n_init='auto')
#     #         labels = model.fit_predict(X)
#     #         score = silhouette_score(X, labels)
#     #         silhouette_scores.append(score)
#     #         st.write(f"k = {k} → Silhouette Score: {score:.4f}")

#     #     best_k = k_range[np.argmax(silhouette_scores)]
#     #     st.success(f"✅ Optimal number of clusters: {best_k}")

#     #     fig, ax = plt.subplots()
#     #     ax.plot(list(k_range), silhouette_scores, marker='o')
#     #     ax.set_title("Silhouette Score vs. k")
#     #     ax.set_xlabel("k")
#     #     ax.set_ylabel("Score")
#     #     st.pyplot(fig)
    
#     if st.button("Run Silhouette Score Analysis"):
#         df_clean = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
#         X = StandardScaler().fit_transform(df_clean)

#         silhouette_scores = []
#         k_range = range(2, 11)

#         for k in k_range:
#             model = KMeans(n_clusters=k, random_state=42, n_init='auto')
#             labels = model.fit_predict(X)
#             score = silhouette_score(X, labels)
#             silhouette_scores.append(score)
#             st.write(f"k = {k} → Silhouette Score: {score:.4f}")

#         best_k = k_range[np.argmax(silhouette_scores)]
#         st.session_state.best_k = best_k  # ✅ Save in session state
#         st.success(f"✅ Optimal number of clusters: {best_k}")

#         # Plot
#         fig, ax = plt.subplots()
#         ax.plot(list(k_range), silhouette_scores, marker='o')
#         ax.set_title("Silhouette Score vs. k")
#         ax.set_xlabel("k")
#         ax.set_ylabel("Score")
#         st.pyplot(fig)
        
#     if st.button("Run KMeans with Optimal k"):
#         if "best_k" not in st.session_state:
#             st.warning("Please run silhouette score analysis first.")
#             st.stop()

#         best_k = st.session_state.best_k
#         df_clean = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
#         X = StandardScaler().fit_transform(df_clean)

#         with st.spinner(f"Running KMeans with k = {best_k}"):
#             model = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
#             labels = model.fit_predict(X)
#             df["kmeans_cluster"] = labels

#             # PCA projection
#             pca = PCA(n_components=2)
#             proj = pca.fit_transform(X)
#             df["pca_1"] = proj[:, 0]
#             df["pca_2"] = proj[:, 1]

#             # Plot
#             fig, ax = plt.subplots()
#             sns.scatterplot(data=df, x="pca_1", y="pca_2", hue="kmeans_cluster", palette="tab10", ax=ax)
#             ax.set_title("KMeans Clusters (PCA)")
#             st.pyplot(fig)

#             df.to_csv(os.path.join(OUTPUT_DIR, "kmeans_results.csv"), index=False)
#             st.success("KMeans clustering completed and saved.")

