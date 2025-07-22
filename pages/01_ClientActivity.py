# pages/01_ClientActivity.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Page config
st.set_page_config(page_title="Client Activity", layout="wide")
st.title("📊 Client Activity Duration & Persona Histograms")

# Constants
OUTPUT_DIR = "output"
SEED = 42
SEED_SUFFIX = f"_seed_{SEED}.csv"
PLOT_HIST = os.path.join(OUTPUT_DIR, "persona_activity_histogram.png")
PLOT_BOX = os.path.join(OUTPUT_DIR, "persona_activity_boxplot.png")

# Required interaction files
interaction_files = [
    "page_visit",
    "add_to_cart",
    "remove_from_cart",
    "product_buy",
    "search_query"
]

# === Function: Load all interactions with timestamps ===
@st.cache_data
def load_and_stack_timestamps():
    dfs = []
    for file_prefix in interaction_files:
        file_path = os.path.join(OUTPUT_DIR, f"{file_prefix}{SEED_SUFFIX}")
        if not os.path.exists(file_path):
            st.error(f"❌ Missing file: {file_path}")
            st.stop()
        df = pd.read_csv(file_path, usecols=["client_id", "timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
        df = df.dropna(subset=["timestamp"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# === Function: Compute activity duration ===
def compute_user_history(df_all):
    history = df_all.groupby("client_id")["timestamp"].agg(["min", "max"]).reset_index()
    history.rename(columns={"min": "first_seen", "max": "last_seen"}, inplace=True)
    history["activity_days"] = (history["last_seen"] - history["first_seen"]).dt.days
    history["activity_months"] = (history["activity_days"] / 30).round(1)
    return history

# === Function: Load personas ===
def load_personas():
    path = os.path.join(OUTPUT_DIR, "user_personas.csv")
    if not os.path.exists(path):
        st.error("❌ Missing `user_personas.csv`. Please generate personas from the main pipeline.")
        st.stop()
    return pd.read_csv(path)

# === Function: Plot charts ===
def generate_plots(df):
    df = df[df["activity_months"] <= 12]

    st.subheader("Stacked Histogram: Activity Duration by Persona")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x="activity_months", hue="persona", multiple="stack", palette="Set2", bins=20, ax=ax1)
    ax1.set_title("User Activity Duration (in Months) by Persona")
    ax1.set_xlabel("Activity Duration (months)")
    ax1.set_ylabel("Number of Users")
    st.pyplot(fig1)
    fig1.savefig(PLOT_HIST)
    plt.close(fig1)

    st.subheader("Boxplot: Activity Duration by Persona")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="persona", y="activity_months", palette="Set2", ax=ax2)
    ax2.set_title("Distribution of Activity Duration by Persona")
    ax2.set_xlabel("Persona")
    ax2.set_ylabel("Activity Duration (months)")
    st.pyplot(fig2)
    fig2.savefig(PLOT_BOX)
    plt.close(fig2)

# === Pipeline Execution ===
with st.spinner("Loading and processing data..."):
    all_ts = load_and_stack_timestamps()
    history_df = compute_user_history(all_ts)
    personas_df = load_personas()
    merged = personas_df.merge(history_df, on="client_id", how="left")

    if merged["activity_months"].isna().all():
        st.error("⚠️ No timestamp history matched the personas. Are you using the same clients?")
        st.stop()

st.success("✅ Data loaded and merged successfully.")
generate_plots(merged)

# Optional preview
with st.expander("🔍 Preview Merged Data"):
    st.dataframe(merged[["client_id", "persona", "activity_months"]].head(100), use_container_width=True)
