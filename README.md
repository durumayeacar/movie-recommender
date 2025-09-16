# 🎬 Movie Recommender (Portfolio Project)

A simple, extensible movie recommendation system using the MovieLens dataset.  
This starter includes:
- Popularity baseline (with Bayesian smoothing)
- Content-based + Collaborative Filtering stubs (ready for you to implement)
- A minimal Streamlit app to demo recommendations
- Clean repo structure & `.gitignore`



## Project structure

```
movie-recommender/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ raw/         
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


