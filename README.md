# 🎬 Movie Recommendation System

A content-based movie recommender: give it a movie you like, and it suggests similar ones — using TF-IDF over genres and plot text, plus cosine similarity. No user ratings required.

Built as Task 4 / Project 1 for the **Syntecxhub Machine Learning Internship Program**.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit–learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Flask](https://img.shields.io/badge/flask-optional%20API-black)
![Status](https://img.shields.io/badge/status-complete-success)

-----

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Quickstart](#quickstart)
- [Example Output](#example-output)
- [Optional Flask API](#optional-flask-api)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Diagrams](#diagrams)
- [Roadmap](#roadmap)
- [Full Technical Report](#full-technical-report)
- [Acknowledgments](#acknowledgments)

-----

## Overview

**Content-based filtering** recommends items based on their own attributes, not on what other users liked. That makes it a good fit here: a brand-new movie with zero ratings can still be recommended correctly on day one, since the recommendation is derived entirely from its own genre tags and plot summary.

**Highlights:**

- Two independently-weighted TF-IDF feature spaces (genre + overview), combined via `scipy.sparse.hstack` — not a naive “repeat the genre text” hack
- Full EDA + metadata cleaning pipeline (nulls, duplicates, genre normalization)
- Cosine similarity for ranking, with a quantitative proxy metric (Jaccard genre overlap@K) to sanity-check the model
- Optional Flask REST API with proper error handling (400s for bad input, not unhandled 500s)
- Dataset-agnostic — point it at the real MovieLens or TMDB data with one flag, no code changes

## How It Works

Each movie’s feature vector is built like this:

```
v_movie = [ w_g · v_genre  ‖  v_overview ]
```

where `w_g` is `genre_weight` (default `2.0`) and `‖` is column-wise concatenation. Concretely:

1. **Load & clean** — `load_data()` + `run_eda()` handle nulls, duplicate titles, and normalize genre tags (trimmed, deduplicated, sorted).
1. **Vectorize** — two separate `TfidfVectorizer`s: one for genre tags (hyphen-aware tokenizer, so `"Sci-Fi"` stays one token), one for the cleaned overview text (`sublinear_tf=True`, so long summaries don’t win purely on word count).
1. **Weight & combine** — the genre matrix is scaled by `genre_weight` and concatenated with the overview matrix via `hstack`. This weights genre agreement more heavily *without* distorting either vectorizer’s own TF/IDF statistics.
1. **Similarity** — `cosine_similarity()` computes an N×N similarity matrix once, at startup.
1. **Recommend** — `recommend(title, df, sim_matrix, top_n=5)` looks up a title (case-insensitive, whitespace-tolerant), and returns its top-N nearest neighbors by similarity score.

## Quickstart

```bash
# clone and enter the repo
git clone <your-repo-url>
cd movie_recommender

# set up a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run it
python3 recommender.py
```

That’s it — this prints the EDA summary, 5 sample recommendations, and a quantitative evaluation metric.

### Run your own query

```python
from recommender import load_data, run_eda, build_similarity_matrix, recommend

df = run_eda(load_data("data/movies.csv"))
sim_matrix, _ = build_similarity_matrix(df)
print(recommend("The Dark Knight", df, sim_matrix, top_n=5))
```

### Use your own dataset

```bash
python3 recommender.py --data path/to/your_movies.csv --genre_weight 2.0
```

Any CSV with `title`, `genres` (pipe-separated), and `overview` columns works — including the real [MovieLens](https://grouplens.org/datasets/movielens/) or [TMDB](https://www.themoviedb.org/) datasets.

## Example Output

```
Because you watched: Toy Story
       title                            genres  similarity
Finding Nemo Animation|Adventure|Comedy|Family       0.800
          Up Animation|Adventure|Comedy|Family       0.800
    Zootopia Animation|Adventure|Comedy|Family       0.800
  Inside Out Animation|Adventure|Comedy|Family       0.800
       Shrek Animation|Adventure|Comedy|Family       0.800

Because you watched: Get Out
         title                       genres  similarity
The Conjuring Horror|Mystery|Thriller       0.809
            It          Horror|Thriller       0.643
 A Quiet Place    Horror|Sci-Fi|Thriller       0.581
```

## Optional Flask API

```bash
python3 app.py
```

|Endpoint                                |Description                          |
|----------------------------------------|-------------------------------------|
|`GET /movies`                           |List every title in the dataset      |
|`GET /recommend?title=Inception&top_n=5`|Get top-N recommendations for a title|

Both endpoints return JSON, with proper `400`/`404` error responses (missing title, non-integer `top_n`, unknown title) instead of raw server errors.

```bash
curl "http://127.0.0.1:5000/recommend?title=Inception&top_n=3"
```

## Project Structure

```
movie_recommender/
├── data/
│   └── movies.csv          # dataset (title, genres, overview)
├── assets/                  # diagrams used in this README
├── make_dataset.py           # builds data/movies.csv
├── recommender.py             # EDA, cleaning, TF-IDF, cosine similarity, CLI
├── app.py                      # optional Flask API
├── requirements.txt
└── README.md
```

## Dataset

`data/movies.csv` ships with 50 curated movies (title, genres, overview) spanning 18 genres, used to validate the full pipeline end-to-end. Swap in the real MovieLens or TMDB dataset any time — see [Quickstart](#quickstart) above; no code changes needed, just matching column names.

## Evaluation

Besides qualitative sample queries, `mean_genre_overlap_at_k()` computes a proxy metric that needs no held-out ratings: the average Jaccard overlap between each movie’s genre set and its own top-K recommendations’ genre sets.

- **Mean Genre Overlap (Jaccard) @ 5 = 0.635** at the default `genre_weight=2.0`
- A sensitivity sweep (`genre_weight` from 0.5 to 5.0) confirms this default isn’t a fragile, arbitrarily-tuned choice — the metric stays essentially flat (0.634–0.635) across a wide range

Worth noting: this metric measures **genre-consistency**, not recommendation *quality* — since `genre_weight` already controls how much genre drives the similarity score, a high score here mostly confirms the model does what it was built to do. It’s best used as a regression check (“did a code change break genre alignment?”), not as evidence users would actually like the results. See the [full report](#full-technical-report) for the complete discussion.

## Diagrams

<p align="center">
  <img src="assets/pipeline_diagram.png" alt="End-to-end pipeline" width="700"><br>
  <em>End-to-end content-based recommendation pipeline</em>
</p>

<p align="center">
  <img src="assets/genre_distribution.png" alt="Genre distribution" width="500">
  <img src="assets/similarity_heatmap.png" alt="Cosine similarity heatmap" width="440">
</p>

<p align="center">
  <img src="assets/architecture_diagram.png" alt="Flask API architecture" width="420"><br>
  <em>Optional Flask API request flow</em>
</p>

## Roadmap

- [ ] Swap in the full MovieLens (25M ratings) or TMDB dataset at production scale
- [ ] Add collaborative filtering and combine into a hybrid recommender
- [ ] Incorporate cast/director/keyword metadata into the feature space
- [ ] Replace TF-IDF with dense sentence embeddings for semantic similarity
- [ ] Add precision@k / recall@k against real held-out user interactions
- [ ] Swap the dense similarity matrix for an approximate nearest-neighbour index (FAISS/Annoy) at scale

## Full Technical Report

For the complete methodology, mathematical formulation, architecture diagrams, full source code listing, and a two-round code-review changelog (bugs found, fixes verified, and points investigated and found not applicable), see **`Movie_Recommendation_System_Report.docx`** in this repo.

## Acknowledgments

- [Syntecxhub](https://www.syntecxhub.com) — Machine Learning Internship Program
- [MovieLens](https://grouplens.org/datasets/movielens/) / [TMDB](https://www.themoviedb.org/) — dataset schema reference
- Built with [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/), and [Flask](https://flask.palletsprojects.com/)

-----

*This project was built for educational purposes as part of an internship program. Feel free to fork and adapt.*