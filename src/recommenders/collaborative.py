import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from implicit.als import AlternatingLeastSquares
except Exception as e:
    AlternatingLeastSquares = None

def _build_mappings(ratings: pd.DataFrame):
    users = ratings['userId'].unique()
    items = ratings['movieId'].unique()
    user_map = {u:i for i,u in enumerate(sorted(users))}
    item_map = {m:i for i,m in enumerate(sorted(items))}
    return user_map, item_map

def _build_user_item_csr(ratings: pd.DataFrame, user_map, item_map):
    rows = ratings['userId'].map(user_map).values
    cols = ratings['movieId'].map(item_map).values
    vals = ratings['rating'].astype(float).values
    n_users = len(user_map); n_items = len(item_map)
    ui = scipy_sparse = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    return ui

def fit_als(ratings: pd.DataFrame, factors: int = 64, reg: float = 0.05, iterations: int = 15, random_state: int = 42):
    if AlternatingLeastSquares is None:
        raise ImportError("implicit library not available. Install with: pip install implicit")
    user_map, item_map = _build_mappings(ratings)
    ui = _build_user_item_csr(ratings, user_map, item_map)

    model = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iterations, random_state=random_state)
    model.fit(ui.T)

    inv_item_map = {i:m for m,i in item_map.items()}
    inv_user_map = {i:u for u,i in user_map.items()}
    return {"model": model, "ui": ui, "user_map": user_map, "item_map": item_map,
            "inv_user_map": inv_user_map, "inv_item_map": inv_item_map}

def recommend_als(user_id: int, model_bundle: dict, k: int = 10, exclude_ids=None):
    exclude_ids = set(exclude_ids or [])
    model = model_bundle["model"]
    ui = model_bundle["ui"]
    user_map = model_bundle["user_map"]
    inv_item_map = model_bundle["inv_item_map"]

    if user_id not in user_map:
        return []

    internal_uid = user_map[user_id]
    item_indices, scores = model.recommend(internal_uid, ui, N=k + len(exclude_ids))
    recs = []
    for idx, sc in zip(item_indices, scores):
        mid = inv_item_map.get(idx)
        if mid in exclude_ids:
            continue
        recs.append((mid, float(sc)))
        if len(recs) >= k:
            break
    return [mid for mid, _ in recs]
