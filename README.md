#  Interactive Data Pipeline & Clustering App

  This Streamlit application allows you to interactively explore, clean, and analyze user interaction data through visual EDA, persona creation, feature engineering, and clustering with dimensionality reduction.

## Folder Structure
```bash

├── app.py                      # Main Streamlit application
├── mypackage/                 # Modular pipeline logic
│   ├── chunking.py
│   ├── clustering.py
│   ├── feature_engineering.py
│   ├── profiling.py
│   ├── similarity.py
│   ├── deep_learning.py       # Autoencoder embeddings
│   └── ...
├── input/                     # Input parquet and .npy files
│   ├── relevant_clients.npy
│   ├── add_to_cart.parquet
│   ├── remove_from_cart.parquet
│   ├── product_buy.parquet
│   ├── search_query.parquet
│   └── page_visit.parquet
├── output/                    # Auto-generated outputs
│   ├── user_feature_matrix.csv
│   ├── user_personas.csv
│   ├── kmeans_results.csv
│   ├── dbscan_results.csv
│   ├── cluster_profiles.csv
│   ├── plots/
│   └── *_seed_*.csv           # Seed-filtered event files
├── requirements.txt           # Python dependencies
└── README.md
```

## Features

- **Client Sampling:** Choose a subset of users to focus the analysis on (with seed support).
- **EDA:** Visualize interaction data via histograms and heatmaps, clustering visualizations.
- **Persona Creation:** Categorize users based on behavior (e.g., browsers, buyers).
- **Feature Engineering:** Calculates behavioral, monetary, categorical, and composite features.
- **Clustering & Projection:**
  - Dimensionality Reduction: t-SNE and PCA
  - Clustering: DBSCAN
  - Means with silhouette score analysis and profiling
- **Exportable Results:** Filtered datasets, feature matrix, cluster plots.
- **Deep Learning Embeddings:** 
  - Autoencoder-based user embeddings for dimensionality reduction.

- **Similarity Analysis:** Compare client behaviors via radar plots and metrics.

- **Profiling by Cluster:** Generate cluster-level summaries.

- **Caching and Reproducibility:** Uses consistent seeds and cached outputs for reproducibility.

## Dataset Download

https://www.recsyschallenge.com/2025/#dataset


##  Getting Started

 **Clone the repo**
git clone https://github.com/your-username/data-pipeline-clustering-app.git
cd data-pipeline-clustering-app

## Install dependencies

pip install -r requirements.txt

## Add input files

Place the following files inside the input/ directory:

- relevant_clients.npy

- active_clients.npy

- add_to_cart.parquet

- remove_from_cart.parquet

- product_buy.parquet

- search_query.parquet

- page_visit.parquet


## Extra Tips
TensorFlow 2.x works best with Python 3.10.

## Run the app
python -m streamlit run app.py

