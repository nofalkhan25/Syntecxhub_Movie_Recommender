"""
Optional Flask endpoint for the Movie Recommendation System.

Run:
    python3 app.py
Then visit:
    http://127.0.0.1:5000/recommend?title=Inception
    http://127.0.0.1:5000/recommend?title=Toy%20Story&top_n=3
    http://127.0.0.1:5000/movies              (lists all available titles)

Revision note (round 2 code review): top_n is now parsed defensively —
a non-integer value like ?top_n=five previously raised an unhandled
ValueError (HTTP 500); it now returns a clean HTTP 400 with a message
explaining what was wrong. The dataset path also now resolves via
recommender.DEFAULT_DATA_PATH instead of the bare string
"data/movies.csv", so this app works regardless of the caller's cwd.
"""

from flask import Flask, jsonify, request

from recommender import DEFAULT_DATA_PATH, build_similarity_matrix, load_data, recommend, run_eda

app = Flask(__name__)

# Build the model once at startup
_df = run_eda(load_data(DEFAULT_DATA_PATH))
_sim_matrix, _ = build_similarity_matrix(_df)


@app.get("/movies")
def list_movies():
    return jsonify(sorted(_df["title"].tolist()))


@app.get("/recommend")
def get_recommendations():
    title = request.args.get("title", "")
    if not title:
        return jsonify({"error": "pass a ?title= query param"}), 400

    raw_top_n = request.args.get("top_n", "5")
    try:
        top_n = int(raw_top_n)
    except ValueError:
        return jsonify({"error": f"top_n must be an integer, got {raw_top_n!r}"}), 400

    try:
        result = recommend(title, _df, _sim_matrix, top_n=top_n)
        return jsonify(result.to_dict(orient="records"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


if __name__ == "__main__":
    app.run(debug=True)
