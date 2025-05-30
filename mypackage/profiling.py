import pandas as pd
import numpy as np

def create_user_segments(visits, carts, removes, buys):
    """Assign personas to users based on their behavior across datasets."""
    all_users = set(visits['client_id']) | set(carts['client_id']) | set(buys['client_id'])
    viewers = set(visits['client_id'])
    cart_users = set(carts['client_id'])
    buyers = set(buys['client_id'])

    profiles = []

    for uid in all_users:
        is_viewer = uid in viewers
        is_cart_user = uid in cart_users
        is_buyer = uid in buyers

        if is_buyer:
            if is_cart_user:
                persona = "loyal_buyer"
            else:
                persona = "impulsive_buyer"
        elif is_cart_user:
            persona = "cart_abandoner"
        elif is_viewer:
            persona = "browser"
        else:
            persona = "inactive"

        profiles.append({
            'client_id': uid,
            'is_viewer': is_viewer,
            'is_cart_user': is_cart_user,
            'is_buyer': is_buyer,
            'persona': persona
        })

    #return pd.DataFrame(profiles)
    df = pd.DataFrame(profiles)
    df.to_csv("output/user_personas.csv", index=False)
    return df

def build_feature_matrix(visits, carts, removes, buys):
    """Constructs numerical feature matrix from interactions."""
    features = []
    users = pd.concat([visits, carts, removes, buys])['client_id'].unique()

    for uid in users:
        user_visits = visits[visits['client_id'] == uid]
        user_carts = carts[carts['client_id'] == uid]
        user_removes = removes[removes['client_id'] == uid]
        user_buys = buys[buys['client_id'] == uid]

        first_visit = pd.to_datetime(user_visits['timestamp']).min() if not user_visits.empty else pd.NaT
        first_buy = pd.to_datetime(user_buys['timestamp']).min() if not user_buys.empty else pd.NaT
        time_to_buy = (first_buy - first_visit).total_seconds() / 3600 if pd.notnull(first_buy) and pd.notnull(first_visit) else np.nan

        features.append({
            'client_id': uid,
            'num_page_visits': len(user_visits),
            'num_adds': len(user_carts),
            'num_removes': len(user_removes),
            'num_purchases': len(user_buys),
            'conversion_rate': len(user_buys) / len(user_carts) if len(user_carts) > 0 else 0,
            'time_to_buy_hr': time_to_buy
        })

    return pd.DataFrame(features)
