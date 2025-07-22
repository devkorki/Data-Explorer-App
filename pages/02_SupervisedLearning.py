import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    precision_recall_curve
)
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("Supervised Learning: Churn Prediction")

# === Load data
@st.cache_data
def load_data():
    df = pd.read_csv("output/user_feature_matrix.csv")
    df["churned"] = (df["days_since_last_buy"] > 90).astype(int)
    df = df.dropna(subset=["churned"])

    X = df.drop(columns=["client_id", "churned", "churn_risk_score"], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    y = df["churned"]
    X = X.drop(columns=["min", "max", "preferred_price_tier"], errors="ignore")

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    # Remove leakage
    leakage_cols = ["days_since_last_buy", "buy_decay_score"]
    X = X.drop(columns=[col for col in leakage_cols if col in X.columns])

    df_sorted = df.sort_values("days_since_last_visit").reset_index(drop=True)
    X_sorted = X.iloc[df_sorted.index].reset_index(drop=True)
    y_sorted = y.iloc[df_sorted.index].reset_index(drop=True)

    cutoff = int(0.7 * len(df_sorted))
    X_train = X_sorted.iloc[:cutoff]
    X_test = X_sorted.iloc[cutoff:]
    y_train = y_sorted.iloc[:cutoff]
    y_test = y_sorted.iloc[cutoff:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return df_sorted, X_train_scaled, X_test_scaled, y_train, y_test, X.columns, cutoff

df_sorted, X_train, X_test, y_train, y_test, feature_names, cutoff = load_data()

# === Model selection
model_choice = st.selectbox("Choose model", ["Random Forest", "Logistic Regression"])

if model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
else:
    model = LogisticRegression(max_iter=1000, class_weight='balanced')

model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]

# === Threshold selection
st.sidebar.header("Threshold Optimization")
threshold = st.sidebar.slider("Classification threshold", 0.1, 0.9, 0.5, 0.01)

final_preds = (probs >= threshold).astype(int)
f1 = f1_score(y_test, final_preds)
accuracy = accuracy_score(y_test, final_preds)

precision = precision_score(y_test, final_preds, zero_division=0)
recall = recall_score(y_test, final_preds)
auc = roc_auc_score(y_test, probs)

# === Show KPIs
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("F1 Score", f"{f1:.3f}")
col3.metric("Precision", f"{precision:.3f}")
col4.metric("Recall", f"{recall:.3f}")
col5.metric("ROC AUC", f"{auc:.3f}")

# === Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, final_preds)
fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax_cm)
st.pyplot(fig_cm)

# === ROC Curve
fpr, tpr, _ = roc_curve(y_test, probs)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
ax_roc.plot([0, 1], [0, 1], "k--")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend()
st.pyplot(fig_roc)

# === Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, probs)
fig_pr, ax_pr = plt.subplots()
ax_pr.plot(rec, prec)
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision-Recall Curve")
st.pyplot(fig_pr)

# === Feature Importance (RF only)
if model_choice == "Random Forest":
    st.subheader("Top 15 Features by Importance")
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]
    fig_feat, ax_feat = plt.subplots()
    ax_feat.barh(range(len(indices)), importances[indices])
    ax_feat.set_yticks(range(len(indices)))
    ax_feat.set_yticklabels(feature_names[indices])
    ax_feat.set_title("Top 15 Predictors")
    st.pyplot(fig_feat)

# === Export predictions
df_test = df_sorted.iloc[cutoff:].copy()
df_test["y_true"] = y_test.values
df_test["y_pred"] = final_preds
df_test["churn_probability"] = probs

top_cols = [
    "client_id", "y_true", "y_pred", "churn_probability", "days_since_last_buy",
    "buy_decay_score", "cart_conversion_rate", "total_spend", "visit_count", "category_loyalty_score"
]

for feat in feature_names[:10]:
    if feat not in top_cols:
        top_cols.append(feat)

export_df = df_test[top_cols].copy()
export_df.to_csv("output/churn_predictions.csv", index=False)

st.success("✅ Predictions exported to `output/churn_predictions.csv`")

with open("output/churn_predictions.csv", "rb") as f:
    st.download_button("📥 Download Predictions CSV", f, file_name="churn_predictions.csv", mime="text/csv")
