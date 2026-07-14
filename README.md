# 🎵 Intelligent Music Recommender System
> Discover songs that feel just like your favourite track — powered by acoustic feature analysis & cosine similarity

---

## About This Project
### Architecture
A content-based music recommendation engine using Spotify audio features and cosine similarity. The system is trained on a global multi-million-song dataset and computes similarity on-demand — no NxN matrix, making it scalable to 1M+ songs across any genre or language.

---

## How Recommendations Work
1. **Feature Extraction** — Each song becomes a 9-dimensional vector of Spotify audio features
2. **Normalisation** — Min-Max scaling to [0, 1] for fair comparison
3. **Cosine Similarity** — One seed vector vs all N songs via fast matrix multiplication
4. **Filtering** — Language, Genre, and Audio Feature bucket filters
5. **Ranking** — Top-N most similar songs returned with confidence scores

---

## Audio Features
| Feature | Range | What it measures |
|---|---|---|
| Danceability | 0–1 | Rhythmic suitability for dancing |
| Energy | 0–1 | Intensity & perceptual activity |
| Loudness | dB | Average loudness |
| Speechiness | 0–1 | Spoken word presence |
| Acousticness | 0–1 | Acoustic confidence |
| Instrumentalness | 0–1 | No-vocals prediction |
| Liveness | 0–1 | Live audience presence |
| Valence | 0–1 | Musical positiveness (happiness) |
| Tempo | BPM | Beats per minute |

---

## Dataset
The recommender is trained on a combined global dataset drawn from multiple Spotify data sources, covering millions of tracks spanning Pop, Rock, Hip-Hop, Jazz, Electronic, Classical, Folk, Anime, Country, R&B, Indie, and many more genres. Data preparation merges, deduplicates, and standardises all sources into a single optimised CSV.

📥 **Download the dataset:** [master_songs.csv (Google Drive)](https://drive.google.com/file/d/1dEJALD3GjBnUveAuGxofB5lbqteD2O5R/view?usp=sharing) — place it inside the `data/` folder before running the app.

> **Note:** `prepare_data.py` is only needed if you have your own raw Spotify CSV source files to merge — place them inside `data/raw/` (any `.csv` files, subfolders are fine too). If you just want to run the app, download the ready-made `master_songs.csv` above instead.

---

## Tech Stack
`Python` · `Streamlit` · `scikit-learn` · `pandas` · `NumPy` · `Plotly` · `SciPy`

---

## How to Run
### Option A: If You Cloned the Repository
1. Clone the repository: `git clone <repo-url>`
2. Navigate into the project folder: `cd music-recommender` (or whatever the folder is named)
3. Create a virtual environment (optional but recommended): `python -m venv venv`
4. Activate the virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
5. Install dependencies: `pip install -r requirements.txt`
6. Ensure `data/master_songs.csv` exists — [download the pre-built dataset](https://drive.google.com/file/d/1dEJALD3GjBnUveAuGxofB5lbqteD2O5R/view?usp=sharing) and place it in `data/` (recommended), **or**, only if you have your own raw Spotify CSVs in `data/raw/`, run `python prepare_data.py` to generate it yourself
7. Run the app: `python -m streamlit run app.py`
8. Open the URL shown in the terminal (usually `http://localhost:8501`) in your browser

### Option B: If You Downloaded/Have the Code Folder
1. Extract or place the project folder anywhere on your system
2. Open a terminal and navigate into the project folder: `cd path/to/music-recommender`
3. Create a virtual environment (optional but recommended): `python -m venv venv`
4. Activate the virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
5. Install dependencies: `pip install -r requirements.txt`
6. Ensure `data/master_songs.csv` exists — [download the pre-built dataset](https://drive.google.com/file/d/1dEJALD3GjBnUveAuGxofB5lbqteD2O5R/view?usp=sharing) and place it in `data/` (recommended), **or**, only if you have your own raw Spotify CSVs in `data/raw/`, run `python prepare_data.py` to generate it yourself
7. Run the app: `python -m streamlit run app.py`
8. Open the URL shown in the terminal (usually `http://localhost:8501`) in your browser

---

## Team
| Name | Roll Number |
|---|---|
| Abhay Pandit | 0002CD231001 |
| Aman Sen | 0002CD231007 |
| Anany Mishra | 0002CD231008 |
| Varun Gupta | 0002CD231075 |

**Under Guidance Of**
- Mr. Madhav Chaturvedi — Assistant Professor
- Mr. Gajendra Kumar Ahirwar — Assistant Professor

**Bachelor of Technology — Computer Science Engineering (Data Science)**
School of Information Technology, R.G.P.V, Bhopal
