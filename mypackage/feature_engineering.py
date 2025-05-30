# mypackage/feature_engineering.py

import pandas as pd
import numpy as np
import os
from functools import reduce

CUTOFF_DATE = pd.to_datetime("2025-05-01")

def run_feature_engineering(input_dir="input", output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    def load_csv(name):
        path = os.path.join(output_dir, f"{name}_filtered.csv")
        return pd.read_csv(path)

    def aggregate_counts(df, name):
        return df.groupby("client_id").agg(
            **{f"{name}_count": ("sku", "count"), f"unique_{name}s": ("sku", "nunique")}
        ).reset_index()

    def aggregate_visit(df):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        span = df.groupby("client_id")["timestamp"].agg(min="min", max="max")
        span["visit_days"] = (span["max"] - span["min"]).dt.days
        counts = df.groupby("client_id").agg(
            visit_count=("url", "count"),
            unique_pages=("url", "nunique")
        )
        return span.join(counts).reset_index()

    def recency_feature(df, name):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        last = df.groupby("client_id")["timestamp"].max().reset_index()
        last[f"days_since_last_{name}"] = (CUTOFF_DATE - last["timestamp"]).dt.days
        return last[["client_id", f"days_since_last_{name}"]]

    def decay_score(df, name):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["days_ago"] = (CUTOFF_DATE - df["timestamp"]).dt.days
        score = df.groupby("client_id")["days_ago"] \
                  .apply(lambda d: (1 / (d + 1)).sum()) \
                  .rename(f"{name}_decay_score").reset_index()
        return score

    def monetary_feature(df):
        props = pd.read_parquet(os.path.join(input_dir, "product_properties.parquet"))
        df = df.merge(props[["sku", "price"]], on="sku", how="left")
        return df.groupby("client_id")["price"].agg(
            total_spend="sum",
            avg_order_value="mean"
        ).reset_index()

    # Load datasets
    add = load_csv("add_to_cart")
    buy = load_csv("product_buy")
    rem = load_csv("remove_from_cart")
    visit = load_csv("page_visit")
    search = load_csv("search_query")

    # Generate features
    cart = aggregate_counts(add, "cart")
    buy_ = aggregate_counts(buy, "buy")
    remove = aggregate_counts(rem, "remove")
    visit_ = aggregate_visit(visit)
    search_ = search.groupby("client_id").agg(search_count=("timestamp", "count")).reset_index()

    rec_cart = recency_feature(add, "cart")
    rec_buy = recency_feature(buy, "buy")
    rec_visit = recency_feature(visit, "visit")
    rec_search = recency_feature(search, "search")

    decay_cart = decay_score(add, "cart")
    decay_buy = decay_score(buy, "buy")
    decay_visit = decay_score(visit, "visit")
    decay_search = decay_score(search, "search")

    monetary = monetary_feature(buy)

    # Merge all
    features = [cart, buy_, remove, visit_, search_,
                rec_cart, rec_buy, rec_visit, rec_search,
                decay_cart, decay_buy, decay_visit, decay_search,
                monetary]

    df = reduce(lambda l, r: pd.merge(l, r, on="client_id", how="outer"), features)
    df.fillna(0, inplace=True)

    # Derived
    df["cart_to_buy_rate"] = df["buy_count"] / (df["cart_count"] + 1)
    df["visit_to_cart_rate"] = df["cart_count"] / (df["visit_count"] + 1)
    df["pages_per_visit"] = df["unique_pages"] / (df["visit_count"] + 1)
   # df["items_per_cart"] = df["unique_cart_items"] / (df["cart_count"] + 1)
    df["items_per_cart"] = df["unique_carts"] / (df["cart_count"] + 1)


    # Save
    output_path = os.path.join(output_dir, "user_feature_matrix.csv")
    df.to_csv(output_path, index=False)
    return output_path
