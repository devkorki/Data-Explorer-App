#  Interactive Data Pipeline & Clustering App

This Streamlit application allows you to interactively explore, clean, and analyze user interaction data through visual EDA, persona creation, feature engineering, and clustering with dimensionality reduction.

## Folder Structure
```bash

├── app.py # Main Streamlit application
├── mypackage/ # Contains modular pipeline logic
│ ├── clustering.py
│ ├── feature_engineering.py
│ ├── profiling.py
│ └── ...
├── input/ # Place your input parquet and .npy files here
│ ├── relevant_clients.npy
│ ├── add_to_cart.parquet
│ ├── remove_from_cart.parquet
│ ├── product_buy.parquet
│ ├── search_query.parquet
│ └── page_visit.parquet
├── output/ # Auto-created to store results
│ ├── user_feature_matrix.csv
│ ├── user_personas.csv
│ └── plots/
└── requirements.txt # Required Python packages

```

## Features

- **Client Sampling:** Choose a subset of users to focus the analysis on.
- **EDA:** Visualize interaction data via histograms and heatmaps.
- **Persona Creation:** Categorize users based on behavior (e.g., browsers, buyers).
- **Feature Engineering:** Generate numerical vectors for each user.
- **Clustering & Projection:**
  - Dimensionality Reduction: t-SNE and PCA
  - Clustering: DBSCAN
- **Exportable Results:** Filtered datasets, feature matrix, cluster plots.

##  Getting Started

 **Clone the repo**
git clone https://github.com/your-username/data-pipeline-clustering-app.git
cd data-pipeline-clustering-app

## Install dependencies

pip install -r requirements.txt

## Add input files

Place the following files inside the input/ directory:

- relevant_clients.npy

- add_to_cart.parquet

- remove_from_cart.parquet

- product_buy.parquet

- search_query.parquet

- page_visit.parquet

## Run the app
streamlit run app.py

