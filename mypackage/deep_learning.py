import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

def get_autoencoder_embedding(df, cols, encoding_dim=16, epochs=30, batch_size=128):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    X = df[cols].values
    X = StandardScaler().fit_transform(X)

    input_dim = X.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    bottleneck = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(bottleneck)
    decoded = Dense(64, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    encoder = Model(inputs=input_layer, outputs=bottleneck)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)

    X_encoded = encoder.predict(X)
    embedding_df = pd.DataFrame(X_encoded, columns=[f"embed_{i+1}" for i in range(encoding_dim)])
    embedding_df["client_id"] = df["client_id"].values
    return embedding_df
