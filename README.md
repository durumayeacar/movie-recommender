# 🎬 Movie Recommender (Portfolio Project)

A simple, extensible movie recommendation system using the MovieLens dataset.  
This starter includes:
- Popularity baseline (with Bayesian smoothing)
- Content-based + Collaborative Filtering stubs (ready for you to implement)
- A minimal Streamlit app to demo recommendations
- Clean repo structure & `.gitignore`

## 1) Setup (VS Code or terminal)

**Create a virtual environment & install deps**

- **Windows PowerShell**
  ```ps1
  py -3 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

- **macOS/Linux**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

## 2) Get the data

Download MovieLens **ml-latest-small** (or 100k) and place these files in `data/raw/`:

- `ratings.csv` (columns: userId, movieId, rating, timestamp)
- `movies.csv` (columns: movieId, title, genres)

Official site: https://grouplens.org/datasets/movielens/

## 3) Run the demo app

```bash
streamlit run app/streamlit_app.py
```

- Choose **Popularity** → enter a `userId` that exists in your ratings file.
- The app computes recommendations and lists titles (excluding items you've already rated).

## 4) Project structure

```
movie-recommender/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ raw/          # put MovieLens ratings.csv, movies.csv here
│  └─ processed/
├─ notebooks/
│  └─ README.md
├─ src/
│  ├─ __init__.py
│  ├─ data_load.py
│  ├─ preprocess.py
│  ├─ eval.py
│  └─ recommenders/
│     ├─ popularity.py
│     ├─ content_based.py   # stub
│     └─ collaborative.py   # stub
└─ app/
   └─ streamlit_app.py
```

## 5) Next steps (for your portfolio)

- Implement **content-based** similarity (genres/title TF-IDF) in `src/recommenders/content_based.py`
- Add **ALS** collaborative filtering in `src/recommenders/collaborative.py` (package `implicit`)
- Add offline metrics in `src/eval.py` (precision@K, recall@K, NDCG)
- Include screenshots of the Streamlit app in README

## 6) How to push to GitHub (CLI)

```bash
git init
git add .
git commit -m "Initial commit: movie recommender skeleton"
git branch -M main
# Replace USERNAME and (optionally) REPO name
git remote add origin https://github.com/USERNAME/movie-recommender.git
git push -u origin main
```

**VS Code GUI**
1. Open folder → left sidebar **Source Control** (branch icon) → **Initialize Repository**
2. Enter a commit message → **Commit**
3. Click **Publish Branch** (sign in to GitHub if prompted), choose repo name → **Publish**

---

> Tip: keep `data/` out of git (already handled by `.gitignore`). Commit code, not raw datasets.
