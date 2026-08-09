"""
Movie Recommendation System (Content-Based)
Syntecxhub Machine Learning Internship - Task 4 / Project 1

Pipeline:
    1. Load dataset            -> data/movies.csv (title, genres, overview)
    2. EDA + metadata cleaning -> nulls, duplicates, genre normalization
    3. Feature engineering     -> two separate TF-IDF spaces (genres,
                                   overview), genre space weighted, then
                                   concatenated with scipy.sparse.hstack
    4. Similarity              -> cosine similarity between all movies
    5. Recommend()              -> top-N most similar movies to a given title
    6. Qualitative evaluation  -> run a handful of sample queries and print
                                   the results, so the model's quality can
                                   be inspected by eye.
    7. Quantitative evaluation -> Mean Genre Overlap (Jaccard) @ K, a
                                   proxy metric that needs no held-out
                                   ratings, used only as an internal
                                   consistency check (see README).

Usage:
    python3 recommender.py                  # runs full pipeline + demo
    python3 recommender.py --data path.csv  # use your own dataset instead

Revision note (round 1): first code-review pass. Changes:
    - Genre and overview text are now vectorized separately and combined
      via scipy.sparse.hstack, instead of weighting genres by repeating
      them inside one shared TF-IDF string. Repeating text N times inside
      a single vectorizer distorts that vectorizer's own term-frequency
      and document-frequency statistics; scaling a separately-fitted
      genre matrix by a constant weight does not.
    - TfidfVectorizer(sublinear_tf=True) is used for the overview text,
      replacing raw term frequency with 1 + log(tf) so long plot
      summaries don't dominate purely by being wordier.
    - The genre vectorizer uses a hyphen-aware token pattern so tags like
      "Sci-Fi" are kept as one token instead of splitting into "sci"/"fi".
    - recommend()'s fuzzy-match fallback now passes regex=False, so movie
      titles containing regex metacharacters (parentheses, "?", "+", ...)
      no longer raise a regex error.

Revision note (round 2): a second code-review pass found and fixed:
    - DEFAULT_DATA_PATH is now resolved relative to this file, not the
      caller's working directory, so `python3 app.py` or
      `python3 recommender.py` both find the bundled dataset regardless
      of where they're run from. load_data() also now fails with a
      clear message instead of a raw traceback if the path is wrong.
    - build_similarity_matrix() and mean_genre_overlap_at_k() no longer
      depend on run_eda() having been called first to create
      genres_clean — both call the new _ensure_genres_clean() helper
      defensively, so a KeyError can't leak out from skipped call order.
    - recommend() now strips leading/trailing whitespace from `title`
      before matching — confirmed that "  Inception  " previously failed
      both the exact match AND the substring fallback, even though the
      movie exists.
    - app.py's top_n query parameter is now parsed with a try/except, so
      a non-integer value (?top_n=five) returns a 400 instead of an
      unhandled 500.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Resolved relative to THIS FILE, not the caller's working directory, so
# `python3 /some/other/path/recommender.py` and `python3 app.py` from a
# different cwd both still find the bundled dataset. (A prior version
# defaulted to the bare string "data/movies.csv", which only worked if
# you happened to run the command from inside movie_recommender/.)
DEFAULT_DATA_PATH = str(Path(__file__).resolve().parent / "data" / "movies.csv")


# ----------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"Dataset not found at '{path}'. Pass --data /path/to/movies.csv.")
    required = {"title", "genres", "overview"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Dataset is missing required column(s): {missing}")
    return df


# ----------------------------------------------------------------------
# 2. EDA + CLEANING
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/extra whitespace (used for free-text
    overview only — genre tags are vectorized separately, see
    build_similarity_matrix)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _ensure_genres_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantees a `genres_clean` column exists (trimmed, deduplicated,
    alphabetically sorted, pipe-joined genre tags). Idempotent — a no-op
    if the column is already present.

    This used to happen only inside run_eda(), which meant
    build_similarity_matrix() and mean_genre_overlap_at_k() would raise
    a confusing KeyError if a caller ever invoked them without first
    calling run_eda(). Both functions now call this defensively, so each
    one is independently correct regardless of call order.
    """
    if "genres_clean" in df.columns:
        return df
    df = df.copy()
    df["genres_clean"] = df["genres"].fillna("").apply(
        lambda g: "|".join(sorted({x.strip() for x in g.split("|") if x.strip()}))
    )
    return df


def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("EDA & METADATA CLEANING")
    print("=" * 60)
    print(f"Rows: {len(df)}   Columns: {list(df.columns)}")

    # Note: EDA here reports structural quality (nulls, duplicates, genre
    # frequency) rather than any statistic derived from cleaned overview
    # text, so there is no raw-vs-cleaned-text mismatch to reconcile —
    # clean_text() is only ever applied downstream, inside
    # build_similarity_matrix(), immediately before vectorization.
    nulls = df[["title", "genres", "overview"]].isna().sum()
    print("\nNull counts:\n", nulls.to_string())

    dupes = df.duplicated(subset="title").sum()
    print(f"\nDuplicate titles: {dupes}")

    # drop rows missing the fields we actually need, drop exact dup titles
    before = len(df)
    df = df.dropna(subset=["title", "genres", "overview"]).drop_duplicates(subset="title")
    print(f"Dropped {before - len(df)} row(s) during cleaning -> {len(df)} remain")

    # normalize genre strings, explode for a quick distribution check
    df = _ensure_genres_clean(df)
    genre_counts = (
        df["genres_clean"].str.split("|").explode().value_counts().head(10)
    )
    print("\nTop genres:\n", genre_counts.to_string())
    print()

    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# 3 & 4. FEATURE ENGINEERING + SIMILARITY
# ----------------------------------------------------------------------
def build_similarity_matrix(df: pd.DataFrame, genre_weight: float = 2.0):
    """
    Builds the combined cosine-similarity matrix from two independent
    TF-IDF feature spaces:

      - genre space:    fit only on genre tags, then scaled by
                         `genre_weight` so genre agreement counts for
                         more than incidental word overlap in the plot
                         text, WITHOUT distorting either vectorizer's own
                         internal TF/IDF statistics (unlike repeating the
                         genre string N times inside a shared vectorizer).
      - overview space: fit on cleaned plot-overview text, with
                         sublinear_tf=True (1 + log(tf)) so longer
                         overviews don't dominate purely on word count.

    The two sparse matrices are concatenated column-wise with
    scipy.sparse.hstack before computing cosine similarity, so a movie's
    final vector is simply [weighted genre vector | overview vector].
    """
    df = _ensure_genres_clean(df)  # safe even if run_eda() was skipped

    # token_pattern keeps hyphenated tags like "Sci-Fi" as ONE token
    # instead of the sklearn default splitting it into "sci" and "fi".
    genre_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b")
    genre_matrix = genre_vectorizer.fit_transform(
        df["genres_clean"].str.replace("|", " ", regex=False)
    ) * genre_weight

    overview_vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    overview_matrix = overview_vectorizer.fit_transform(df["overview"].apply(clean_text))

    combined = hstack([genre_matrix, overview_matrix]).tocsr()
    sim_matrix = cosine_similarity(combined, combined)

    vectorizers = {"genre": genre_vectorizer, "overview": overview_vectorizer}
    return sim_matrix, vectorizers


# ----------------------------------------------------------------------
# 5. RECOMMEND
# ----------------------------------------------------------------------
def recommend(title: str, df: pd.DataFrame, sim_matrix, top_n: int = 5) -> pd.DataFrame:
    title = title.strip()  # a leading/trailing space otherwise fails the
    # exact match AND the substring fallback, even for a movie that
    # genuinely exists in the dataset — confirmed via manual testing.
    matches = df.index[df["title"].str.lower() == title.lower()]
    if len(matches) == 0:
        # regex=False: titles containing regex metacharacters — "It (2017)",
        # "What If...?" — must not be interpreted as a regex pattern here.
        close = df[df["title"].str.contains(title, case=False, na=False, regex=False)]
        hint = f" Did you mean: {', '.join(close['title'].head(3))}?" if len(close) else ""
        raise ValueError(f"'{title}' not found in dataset.{hint}")

    idx = matches[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]

    result = df.iloc[[i for i, _ in scores]][["title", "genres"]].copy()
    result["similarity"] = [round(s, 3) for _, s in scores]
    return result.reset_index(drop=True)


# ----------------------------------------------------------------------
# 6. QUALITATIVE EVALUATION
# ----------------------------------------------------------------------
def demo(df: pd.DataFrame, sim_matrix, queries):
    print("=" * 60)
    print("SAMPLE RECOMMENDATIONS (qualitative evaluation)")
    print("=" * 60)
    for q in queries:
        print(f"\nBecause you watched: {q}")
        try:
            print(recommend(q, df, sim_matrix, top_n=5).to_string(index=False))
        except ValueError as e:
            print(f"  [skipped] {e}")


# ----------------------------------------------------------------------
# 7. QUANTITATIVE PROXY EVALUATION
# ----------------------------------------------------------------------
def _genre_set(genres_pipe_str: str) -> set:
    return {g for g in str(genres_pipe_str).split("|") if g}


def mean_genre_overlap_at_k(df: pd.DataFrame, sim_matrix, k: int = 5) -> float:
    """
    Proxy quantitative metric that needs no held-out user ratings: for
    every movie in the catalogue, take its top-K recommendations and
    compute the Jaccard overlap (|A ∩ B| / |A ∪ B|) between its genre set
    and each recommendation's genre set, then average across every movie
    and every one of its K recommendations.

    This is NOT a substitute for a real accuracy metric (precision@k /
    recall@k) against held-out user interactions — see Section 11 /
    "Future Work" in the report — but it gives a single, reproducible
    number to track whether changes to the model make recommendations
    more or less genre-consistent on average, complementing the
    qualitative sample queries in demo().
    """
    df = _ensure_genres_clean(df)  # safe even if run_eda() was skipped
    overlaps = []
    for idx in range(len(df)):
        row = list(enumerate(sim_matrix[idx]))
        row = sorted(row, key=lambda x: x[1], reverse=True)
        row = [r for r in row if r[0] != idx][:k]
        base = _genre_set(df.loc[idx, "genres_clean"])
        for j, _ in row:
            other = _genre_set(df.loc[j, "genres_clean"])
            union = base | other
            overlaps.append(len(base & other) / len(union) if union else 0.0)
    return float(np.mean(overlaps)) if overlaps else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="path to movies CSV")
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument(
        "--genre_weight", type=float, default=2.0,
        help="multiplier applied to the genre TF-IDF space before combining with overview"
    )
    args = parser.parse_args()

    df = load_data(args.data)
    df = run_eda(df)
    sim_matrix, _ = build_similarity_matrix(df, genre_weight=args.genre_weight)

    sample_queries = [
        "Inception",
        "Toy Story",
        "The Dark Knight",
        "La La Land",
        "Get Out",
    ]
    demo(df, sim_matrix, sample_queries)

    metric = mean_genre_overlap_at_k(df, sim_matrix, k=args.top_n)
    print("\n" + "=" * 60)
    print("QUANTITATIVE PROXY EVALUATION")
    print("=" * 60)
    print(f"Mean Genre Overlap (Jaccard) @ {args.top_n}: {metric:.3f}")
    print("(1.0 = every top-K recommendation shares all genres with the")
    print(" query movie; 0.0 = no genre overlap at all. See README.)")

    print("\n" + "=" * 60)
    print("Try your own: python3 recommender.py  then edit sample_queries,")
    print("or `from recommender import *` in a notebook and call")
    print("recommend('Movie Title', df, sim_matrix).")
    print("=" * 60)


if __name__ == "__main__":
    main()
