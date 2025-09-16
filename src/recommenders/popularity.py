import pandas as pd

def _bayesian_average(mean, count, global_mean, m=50):
    return (count/(count+m))*mean + (m/(count+m))*global_mean

def fit_popularity(ratings: pd.DataFrame, m: int = 50):
    # Aggregate mean rating and count per movie
    agg = ratings.groupby('movieId')['rating'].agg(['mean','count']).reset_index()
    global_mean = agg['mean'].mean()
    agg['score'] = _bayesian_average(agg['mean'], agg['count'], global_mean, m=m)
    pop_table = agg.sort_values('score', ascending=False)[['movieId','score','count']]
    return pop_table

def recommend(pop_table: pd.DataFrame, k=10, exclude_ids=None):
    exclude_ids = set(exclude_ids or [])
    recs = pop_table[~pop_table['movieId'].isin(exclude_ids)].head(k)
    return recs['movieId'].tolist()
