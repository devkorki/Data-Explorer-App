import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

def get_most_similar_users(df, feature_cols, top_n=1):
    
    X = StandardScaler().fit_transform(df[feature_cols])
    sim_matrix = cosine_similarity(X)

    # Remove self-similarity
    np.fill_diagonal(sim_matrix, -1)

    pairs = []
    for _ in range(top_n):
        idx1, idx2 = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
        score = sim_matrix[idx1, idx2]
        pairs.append({
            "client_1": df.iloc[idx1]["client_id"],
            "client_2": df.iloc[idx2]["client_id"],
            "similarity": score
        })
        sim_matrix[idx1, idx2] = -1  # prevent duplicate

    return pd.DataFrame(pairs)
