import numpy as np
import pandas as pd

def blend_lists(ids1, scores1, ids2, scores2, w1=0.6, w2=0.4):
    score = {}
    for mid, sc in zip(ids1, scores1):
        score[mid] = score.get(mid, 0.0) + w1*sc
    for mid, sc in zip(ids2, scores2):
        score[mid] = score.get(mid, 0.0) + w2*sc
    ranked = sorted(score.items(), key=lambda x: -x[1])
    return [mid for mid,_ in ranked], [s for _,s in ranked]

def normalize_scores(scores):
    if not scores:
        return []
    arr = np.array(scores, dtype=float)
    if np.all(arr == arr[0]):
        return (arr / max(1.0, np.abs(arr).max())).tolist()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    return arr.tolist()

def hybrid_recommend(user_id: int, ratings: pd.DataFrame, movies: pd.DataFrame,
                     pop_table: pd.DataFrame,
                     content_model: dict, content_func,
                     als_bundle: dict, als_func,
                     k: int = 10, min_like: float = 4.0,
                     w_cf: float = 0.6, w_content: float = 0.4):
    seen = set(ratings.loc[ratings['userId']==user_id, 'movieId'].tolist())

    als_ids, als_scores = [], []
    if als_bundle is not None:
        try:
            tmp_ids = als_func(user_id, als_bundle, k=200, exclude_ids=seen)
            als_ids = tmp_ids
            als_scores = list(reversed(range(1, len(tmp_ids)+1)))
        except Exception:
            pass

    content_ids, content_scores = [], []
    try:
        tmp_ids = content_func(user_id, ratings, movies, content_model, k=200, min_like=min_like)
        content_ids = tmp_ids
        content_scores = list(reversed(range(1, len(tmp_ids)+1)))
    except Exception:
        pass

    if not als_ids and not content_ids:
        from .popularity import recommend as recommend_pop
        return recommend_pop(pop_table, k=k, exclude_ids=seen)

    als_scores = normalize_scores(als_scores)
    content_scores = normalize_scores(content_scores)

    merged_ids, _ = blend_lists(als_ids, als_scores, content_ids, content_scores, w1=w_cf, w2=w_content)

    out = []
    for mid in merged_ids:
        if mid in seen:
            continue
        out.append(mid)
        if len(out) >= k:
            break
    return out
