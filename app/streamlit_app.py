import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd

from src.data_load import load_movielens
from src.preprocess import filter_min_interactions
from src.recommenders.popularity import fit_popularity, recommend as recommend_pop
from src.recommenders.content_based import fit_content_model, recommend_content
from src.recommenders.collaborative import fit_als, recommend_als
from src.recommenders.hybrid import hybrid_recommend

# Optional: posters
import math
try:
    import requests
except Exception:
    requests = None

st.set_page_config(page_title='Movie Recommender', page_icon='🎬', layout='wide')
st.title('🎬 Movie Recommender — Multi-Algorithm')

# Sidebar controls
st.sidebar.header("⚙️ Settings")
k = st.sidebar.slider('How many recommendations?', 5, 30, 10, step=1)
algo = st.sidebar.selectbox("Algorithm:", ["Popularity", "Content-based", "ALS", "Hybrid"])
show_posters = st.sidebar.checkbox("Show posters (links.csv + TMDB_API_KEY required)", value=False)

if algo == "Hybrid":
    w_cf = st.sidebar.slider("Weight: Collaborative (ALS)", 0.0, 1.0, 0.6, 0.05)
    w_content = 1.0 - w_cf
    st.sidebar.caption(f"Content ağırlığı otomatik: {w_content:.2f}")
else:
    w_cf = 0.6
    w_content = 0.4

# Main controls
user_id = st.number_input('User ID (ratings.csv içinde var olmalı)', min_value=1, step=1, value=1)

@st.cache_data(show_spinner=False)
def _load():
    return load_movielens('data/raw')

@st.cache_data(show_spinner=False)
def _prep(ratings):
    return filter_min_interactions(ratings, min_user=3, min_item=10)

@st.cache_data(show_spinner=False)
def build_popularity(ratings):
    return fit_popularity(ratings, m=50)

@st.cache_resource(show_spinner=False)
def build_content(movies):
    return fit_content_model(movies)

@st.cache_resource(show_spinner=False)
def build_als(ratings):
    try:
        return fit_als(ratings, factors=48, reg=0.05, iterations=12)
    except Exception as e:
        st.warning(f"ALS could not be initialized (implicit may be missing): {e}")
        return None

# Posters helpers
@st.cache_data(show_spinner=False)
def load_links():
    links_path = os.path.join('data','raw','links.csv')
    if os.path.exists(links_path):
        df = pd.read_csv(links_path)
        return df[['movieId','tmdbId']]
    return None

def get_poster_url_tmdb(tmdb_id: int, api_key: str):
    if not (requests and api_key and not (pd.isna(tmdb_id) or tmdb_id==0)):
        return None
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}",
                         params={"api_key": api_key, "language":"en-US"}, timeout=5)
        if r.status_code == 200:
            poster_path = r.json().get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w342{poster_path}"
    except Exception:
        return None
    return None

def show_recs(rec_ids, movies_df):
    rec_df = movies_df[movies_df['movieId'].isin(rec_ids)].copy()
    rec_df['rank'] = rec_df['movieId'].apply(lambda x: rec_ids.index(x)+1)
    rec_df = rec_df.sort_values('rank')[['rank','movieId','title','genres']]
    if not show_posters:
        st.dataframe(rec_df)
        return

    # Poster flow
    tmdb_key = os.getenv("TMDB_API_KEY", "")
    links = load_links()
    if (links is None) or not tmdb_key or (requests is None):
        st.info("For the poster, data/raw/links.csv and TMDB_API_KEY are required (and ‘requests’ must be installed). I am showing the table.")
        st.dataframe(rec_df)
        return

    merged = rec_df.merge(links, on='movieId', how='left')
    cols = st.columns(5)
    for i, row in merged.iterrows():
        col = cols[(row['rank']-1) % 5]
        with col:
            url = get_poster_url_tmdb(row.get('tmdbId'), tmdb_key)
            if url:
                st.image(url, use_column_width=True)
            st.caption(f"{row['rank']}. {row['title']}")
    with st.expander("Detay tablo"):
        st.dataframe(rec_df)

# Load & prep
try:
    ratings, movies = _load()
    ratings = _prep(ratings)
except Exception as e:
    st.warning(f"Data not ready: {e}")
    st.stop()

valid_user_ids = sorted(ratings['userId'].unique().tolist())
if user_id not in valid_user_ids:
    st.info("You entered an invalid userId. Please select a valid user after filtering.")
    st.write("Örnek ilk 20 userId:", valid_user_ids[:20])

user_seen = set(ratings.loc[ratings['userId']==user_id, 'movieId'].tolist())
pop_table = build_popularity(ratings)

if st.button('Recommend'):
    if algo == "Popularity":
        rec_ids = recommend_pop(pop_table, k=k, exclude_ids=user_seen)
    elif algo == "Content-based":
        content_model = build_content(movies)
        rec_ids = recommend_content(user_id, ratings, movies, content_model, k=k, min_like=4.0)
        if not rec_ids:
            st.info("If the user does not have any movies rated 4+, a content profile could not be generated. You may try using popularity instead.")
    elif algo == "ALS":
        als_bundle = build_als(ratings)
        if als_bundle is None:
            st.stop()
        rec_ids = recommend_als(user_id, als_bundle, k=k, exclude_ids=user_seen)
        if not rec_ids:
            st.info("The ALS could not produce a result. The user’s interaction may be insufficient.")
    else:  # Hybrid
        content_model = build_content(movies)
        als_bundle = build_als(ratings)
        rec_ids = hybrid_recommend(
            user_id, ratings, movies,
            pop_table,
            content_model, recommend_content,
            als_bundle, recommend_als,
            k=k, min_like=4.0, w_cf=w_cf, w_content=w_content
        )
    if rec_ids:
        show_recs(rec_ids, movies)
    else:
        st.warning("No suggestion found. Please try another user ID or change the algorithm.”")
