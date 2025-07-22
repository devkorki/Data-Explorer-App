# pages/02_SupervisedLearning.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Supervised Learning", layout="wide")
st.title("Supervised Learning: Predicting User Personas")

FEATURE_FILE = "output/user_feature_matrix.csv"
PERSONA_FILE = "output/user_personas.csv"

# Load feature matrix
if not os.path.exists(FEATURE_FILE):
    st.error("Missing 'user_feature_matrix.csv'. Please run feature engineering first.")
    st.stop()

df = pd.read_csv(FEATURE_FILE)

# Merge persona labels
if not os.path.exists(PERSONA_FILE):
    st.error("Missing 'user_personas.csv'. Please generate personas first.")
    st.stop()

personas = pd.read_csv(PERSONA_FILE)
df = df.merge(personas[["client_id", "persona"]], on="client_id", how="left")

# Parse datetime for time-based split
if "first_seen" not in df.columns:
    st.error("Column 'first_seen' not found. Please ensure it is included during feature engineering.")
    st.stop()

df["first_seen"] = pd.to_datetime(df["first_seen"], errors="coerce")

# Filter valid rows
df = df.dropna(subset=["persona", "first_seen"])

# Select numeric features
X = df.drop(columns=["client_id", "persona", "first_seen"], errors="ignore")
X = X.select_dtypes(include=[np.number])
y = df["persona"]

# Sort by time and split
df_sorted = df.sort_values(by="first_seen")
split_ratio = st.slider("Train/Test Split Ratio", 0.1, 0.9, 0.8, 0.05)
split_index = int(len(df_sorted) * split_ratio)

# X_train = X.loc[df_sorted.index[:split_index]]
# X_test = X.loc[df_sorted.index[split_index:]]
# y_train = y.loc[df_sorted.index[:split_index]]
# y_test = y.loc[df_sorted.index[split_index:]]

X_train = X.loc[df_sorted.index[:split_index]]
X_test = X.loc[df_sorted.index[split_index:]]
y_train = y.loc[df_sorted.index[:split_index]]
y_test = y.loc[df_sorted.index[split_index:]]


# Remove infinite or invalid values
X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()

# Align labels with cleaned features
y_train = y_train.loc[X_train.index]
y_test = y_test.loc[X_test.index]

# Clean invalid values
X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()
y_train = y_train.loc[X_train.index]
y_test = y_test.loc[X_test.index]


st.write(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Train model
if st.button("Train Random Forest Classifier"):
    with st.spinner("Training..."):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Classification report
        st.subheader("Classification Report")
        report = classification_report(y_test, preds, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # Confusion matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix(y_test, preds), display_labels=model.classes_)
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
        st.pyplot(fig)
