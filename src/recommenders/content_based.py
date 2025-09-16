# src/recommenders/content_based.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def fit_content_model(movies: pd.DataFrame):
    """
    Build TF-IDF item matrix from titles + genres.
    Returns dict with matrix X and fitted vectorizer.
    """
    text = (
        movies["title"].fillna("").astype(str) + " " +
        movies["genres"].fillna("").str.replace("|", " ", regex=False)
    )
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=30000)
    X = tfidf.fit_transform(text)
    return {"X": X, "tfidf": tfidf}

def _user_profile_from_ratings(user_movie_ids, movies, X):
    if not user_movie_ids:
        return None
    idx_map = {mid: i for i, mid in enumerate(movies["movieId"].tolist())}
    liked_idx = [idx_map[m] for m in user_movie_ids if m in idx_map]
    if not liked_idx:
        return None
    return X[liked_idx].mean(axis=0)

def recommend_content(user_id: int, ratings: pd.DataFrame, movies: pd.DataFrame,
                      model: dict, k: int = 10, min_like: float = 4.0):
    """
    Recommend by averaging vectors of movies the user liked (rating >= min_like).
    Excludes already seen movies.
    """
    X = model["X"]
    user_rows = ratings[ratings["userId"] == user_id]
    seen = set(user_rows["movieId"].tolist())
    liked = user_rows.loc[user_rows["rating"] >= min_like, "movieId"].tolist()

    profile = _user_profile_from_ratings(liked, movies, X)
    if profile is None:
        return []  # fallback to popularity in the app

    sims = cosine_similarity(profile, X).ravel()
    order = np.argsort(-sims)
    recs = []
    mids = movies["movieId"].tolist()
    for idx in order:
        mid = mids[idx]
        if mid in seen or mid in liked:
            continue
        recs.append(mid)
        if len(recs) >= k:
            break
    return recs
