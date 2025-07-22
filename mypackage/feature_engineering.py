import pandas as pd
import numpy as np
import os
from functools import reduce
from datetime import datetime

#CUTOFF_DATE = pd.to_datetime("2025-06-03")
CUTOFF_DATE = pd.to_datetime("2022-10-10")


def calculate_category_metrics(df, props_path):
    props = pd.read_parquet(props_path)
    df = df.merge(props[["sku", "category", "price"]], on="sku", how="left")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    def get_category_stats(group):
        category_counts = group["category"].value_counts()
        total_items = len(group)

        category_shares = category_counts / total_items
        concentration = (category_shares ** 2).sum()

        recent = group[group["timestamp"] >= (CUTOFF_DATE - pd.Timedelta(days=30))]
        recent_cats = recent["category"].value_counts()

        # group["price_tier"] = pd.qcut(group["price"], q=4, labels=["budget", "value", "premium", "luxury"])
        
        # Dynamically assign labels only if we have enough unique prices
        if group["price"].nunique() > 1:
            try:
                group["price_tier"] = pd.qcut(
                    group["price"],
                    q=4,
                    labels=["budget", "value", "premium", "luxury"],
                    duplicates='drop'
                )
            except ValueError:
                group["price_tier"] = "unknown"
        else:
            group["price_tier"] = "unknown"


        tier_counts = group["price_tier"].value_counts()

        return pd.Series({
            "total_categories": len(category_counts),
            "category_concentration": concentration,
            "favorite_category": category_counts.index[0] if len(category_counts) > 0 else None,
            "favorite_category_share": category_counts.iloc[0] / total_items if len(category_counts) > 0 else 0,
            "recent_categories": len(recent_cats),
            "preferred_price_tier": tier_counts.index[0] if len(tier_counts) > 0 else None,
            "luxury_purchase_ratio": (group["price_tier"] == "luxury").mean()
        })

    return df.groupby("client_id").apply(get_category_stats).reset_index()

def calculate_cart_patterns(cart_df, buy_df, remove_df):
    for df in [cart_df, buy_df, remove_df]:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    cart_events = pd.concat([
        cart_df.assign(event_type="add"),
        buy_df.assign(event_type="purchase"),
        remove_df.assign(event_type="remove")
    ]).sort_values(["client_id", "timestamp"])

    cart_events["time_diff"] = cart_events.groupby("client_id")["timestamp"].diff()
    cart_events["new_session"] = (
        (cart_events["time_diff"] > pd.Timedelta(hours=2)) |
        (cart_events["event_type"].shift(1) == "purchase")
    )
    cart_events["cart_session"] = cart_events.groupby("client_id")["new_session"].cumsum()

    def get_session_metrics(group):
        sessions = group.groupby("cart_session")

        def session_outcome(session):
            if "purchase" in session["event_type"].values:
                return "purchased"
            elif "remove" in session["event_type"].values:
                return "removed"
            elif (CUTOFF_DATE - session["timestamp"].max()) > pd.Timedelta(hours=24):
                return "abandoned"
            return "active"

        session_stats = sessions.apply(lambda x: pd.Series({
            "outcome": session_outcome(x),
            "items_count": len(x),
            "duration_minutes": (x["timestamp"].max() - x["timestamp"].min()).total_seconds() / 60
        }))

        recent_sessions = session_stats[
            group.groupby("cart_session")["timestamp"].first() >= 
            (CUTOFF_DATE - pd.Timedelta(days=30))
        ]

        return pd.Series({
            "total_cart_sessions": len(sessions),
            "cart_abandonment_rate": (session_stats["outcome"] == "abandoned").mean(),
            "recent_abandonment_rate": (recent_sessions["outcome"] == "abandoned").mean(),
            "avg_cart_duration": session_stats["duration_minutes"].mean(),
            "avg_items_per_cart": session_stats["items_count"].mean(),
            "cart_conversion_rate": (session_stats["outcome"] == "purchased").mean()
        })

    return cart_events.groupby("client_id").apply(get_session_metrics).reset_index()

def run_feature_engineering(input_dir="input", output_dir="output", seed=None):
    seed_str = f"_seed_{seed}" if seed is not None else ""

    def load_csv(name):
        path = os.path.join(output_dir, f"{name}{seed_str}.csv")
        return pd.read_csv(path)
    
    def aggregate_counts(df, name):
        return df.groupby("client_id").agg(
            **{f"{name}_count": ("sku", "count"), f"unique_{name}s": ("sku", "nunique")}
        ).reset_index()

    def aggregate_visit(df):
        # df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        # span = df.groupby("client_id")["timestamp"].agg(min="min", max="max")
        # span["visit_days"] = (span["max"] - span["min"]).dt.days
        # counts = df.groupby("client_id").agg(
        #     visit_count=("url", "count"),
        #     unique_pages=("url", "nunique")
        # )
        # return span.join(counts).reset_index()
        # span = df.groupby("client_id")["timestamp"].agg(first_seen="min", last_seen="max")
        # span["visit_days"] = (span["last_seen"] - span["first_seen"]).dt.days
        # counts = df.groupby("client_id").agg(
        # visit_count=("url", "count"),
        # unique_pages=("url", "nunique")
        # )
        # return span.join(counts).reset_index()
        
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        span = df.groupby("client_id")["timestamp"].agg(first_seen="min", last_seen="max")

        # Convert to datetime (if not already)
        span["first_seen"] = pd.to_datetime(span["first_seen"], errors="coerce")
        span["last_seen"] = pd.to_datetime(span["last_seen"], errors="coerce")

        span["visit_days"] = (span["last_seen"] - span["first_seen"]).dt.days

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

    # Basic features
    cart = aggregate_counts(add, "cart")
    buy_ = aggregate_counts(buy, "buy")
    remove = aggregate_counts(rem, "remove")
    visit_ = aggregate_visit(visit)
    search_ = search.groupby("client_id").agg(search_count=("timestamp", "count")).reset_index()

    # Recency
    rec_cart = recency_feature(add, "cart")
    rec_buy = recency_feature(buy, "buy")
    rec_visit = recency_feature(visit, "visit")
    rec_search = recency_feature(search, "search")

    # Decay
    decay_cart = decay_score(add, "cart")
    decay_buy = decay_score(buy, "buy")
    decay_visit = decay_score(visit, "visit")
    decay_search = decay_score(search, "search")

    # Monetary
    monetary = monetary_feature(buy)

    # Advanced features
    category_metrics = calculate_category_metrics(
        pd.concat([add, buy]), os.path.join(input_dir, "product_properties.parquet")
    )
    cart_patterns = calculate_cart_patterns(add, buy, rem)

    features = [cart, buy_, remove, visit_, search_,
                rec_cart, rec_buy, rec_visit, rec_search,
                decay_cart, decay_buy, decay_visit, decay_search,
                monetary, category_metrics, cart_patterns]

    df = reduce(lambda l, r: pd.merge(l, r, on="client_id", how="outer"), features)
    df.fillna(0, inplace=True)

    # Derived
    df["cart_to_buy_rate"] = df["buy_count"] / (df["cart_count"] + 1)
    df["visit_to_cart_rate"] = df["cart_count"] / (df["visit_count"] + 1)
    df["pages_per_visit"] = df["unique_pages"] / (df["visit_count"] + 1)
    df["items_per_cart"] = df["unique_carts"] / (df["cart_count"] + 1)

    df["category_loyalty_score"] = (
        (1 - df["category_concentration"]) * 
        df["cart_conversion_rate"] * 
        (1 - df["recent_abandonment_rate"])
    )

    df["churn_risk_score"] = (
        df["cart_abandonment_rate"] * 0.3 +
        (1 - df["cart_conversion_rate"]) * 0.3 +
        (df["days_since_last_visit"] / 30) * 0.2 +
        (1 - df["category_loyalty_score"]) * 0.2
    ).clip(0, 1)

    output_path = os.path.join(output_dir, "user_feature_matrix.csv")
    df.to_csv(output_path, index=False)
    return output_path
