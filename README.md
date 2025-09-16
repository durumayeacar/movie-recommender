# 🎬 Movie Recommender System

This project is a **content-based movie recommendation system** built in Python, featuring a **Streamlit web app** for an interactive experience.  

It recommends movies to users based on similarities in movie **titles and genres** using **TF-IDF vectorization** and **cosine similarity**.  

---

## ✨ Features
- **Content-based recommendations** using TF-IDF and cosine similarity  
- **User profile generation** from movies the user rated highly  
- **Interactive Streamlit interface** for exploring recommendations  
- **Easily extendable** to other algorithms or datasets  

---

## 🛠️ Tech Stack
- [Python 3.x](https://www.python.org/)  
- [Pandas](https://pandas.pydata.org/)  
- [NumPy](https://numpy.org/)  
- [scikit-learn](https://scikit-learn.org/stable/)  
- [Streamlit](https://streamlit.io/)  

---

## 📂 Project Structure
movie-recommender/
│
├── app/
│ └── streamlit_app.py # Streamlit app (entry point)
│
├── src/
│ └── recommenders/
│ ├── content_based.py # Content-based recommendation algorithm
│ └── init.py
│
├── data/ # Movie & ratings data (optional)
├── requirements.txt # Python dependencies
└── README.md


---

## ▶️ How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/movie-recommender.git
   cd movie-recommender
   
2. Create a virtual environment & install dependencies:

python -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows

pip install -r requirements.txt


3. Launch the app:

   streamlit run app/streamlit_app.py

4. Open the local URL shown in the terminal (usually http://localhost:8501) to use the app.

## 📊 Recommendation Logic
1. Combine **title + genres** into a text field.  
2. Apply **TF-IDF vectorization** to generate feature vectors.  
3. Create a **user profile vector** by averaging movies the user rated highly.  
4. Compute **cosine similarity** between the user profile and all movies.  
5. Recommend the top *k* most similar unseen movies.  

---

## 🚀 Future Improvements
- Add **collaborative filtering** or hybrid models  
- Integrate **IMDb/Rotten Tomatoes metadata** for richer recommendations  
- Deploy on **Streamlit Cloud** or **Heroku**  
- Add filtering by genre, release year, or rating  

---

## 📜 License
This project is licensed under the **MIT License**.  



