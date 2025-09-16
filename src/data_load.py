import os
import pandas as pd

def load_movielens(raw_dir='data/raw'):
    ratings_path = os.path.join(raw_dir, 'ratings.csv')
    movies_path = os.path.join(raw_dir, 'movies.csv')
    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        raise FileNotFoundError(
            f"""Expected ratings.csv and movies.csv in {raw_dir}.
Download MovieLens (ml-latest-small) and place files there."""
        )
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    # Basic type checks
    needed_r = {'userId','movieId','rating'}
    needed_m = {'movieId','title','genres'}
    if not needed_r.issubset(ratings.columns):
        raise ValueError(f'ratings.csv missing columns: {needed_r - set(ratings.columns)}')
    if not needed_m.issubset(movies.columns):
        raise ValueError(f'movies.csv missing columns: {needed_m - set(movies.columns)}')
    return ratings, movies
