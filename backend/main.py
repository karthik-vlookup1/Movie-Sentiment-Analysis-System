"""FastAPI backend for the movie sentiment analysis app.

Endpoints
  GET  /api/movies?search=...   find movies by title
  GET  /api/movies/{imdb_id}    get the movie's reviews, classify each one, return the rating
  POST /api/analyze             same analysis for reviews typed in by the user

Run from the project root:  uvicorn backend.main:app --reload
"""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml.config import MODEL_PATH, MOVIES_PATH, REVIEWS_PATH
from ml.preprocess import clean_text
from ml.rating import MIN_REVIEWS, summarize_reviews

app = FastAPI(title="Movie Sentiment Analysis API")
app.add_middleware(
    CORSMiddleware,  # allow the React dev server to call this API
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model_and_data():
    """Load the trained model and the movies/reviews of the *test* split (never seen in training)."""
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Run 'python -m ml.train' first (see README).")
    model = joblib.load(MODEL_PATH)

    reviews = pd.read_csv(REVIEWS_PATH)
    reviews = reviews[reviews["split"] == "test"]
    reviews["text"] = reviews["text"].str.replace("<br />", "\n")  # IMDb stores line breaks as HTML

    review_counts = reviews.groupby("imdb_id").size().rename("review_count")
    movies = pd.read_csv(MOVIES_PATH, dtype=str).fillna("")
    movies = movies.join(review_counts, on="imdb_id", how="inner")
    movies = movies[movies["review_count"] >= MIN_REVIEWS].sort_values("review_count", ascending=False)
    return model, movies, reviews


model, movies, reviews = load_model_and_data()


def movie_info(row):
    return {"imdb_id": row["imdb_id"], "title": row["title"], "year": row["year"],
            "review_count": int(row["review_count"])}


def analyze_reviews(texts):
    """NLP preprocessing -> TF-IDF + Logistic Regression -> aggregate into a rating and verdict."""
    cleaned = [clean_text(text) for text in texts]
    probabilities = model.predict_proba(cleaned)[:, 1]  # the pipeline applies TF-IDF, then the model
    probabilities = [float(p) for p in probabilities]
    results = [
        {"text": text, "sentiment": "Positive" if p >= 0.5 else "Negative", "positive_probability": round(p, 3)}
        for text, p in zip(texts, probabilities)
    ]
    return summarize_reviews(probabilities), results


@app.get("/api/movies")
def search_movies(search: str = ""):
    """Movies whose title contains the search text, most-reviewed first (top 20)."""
    search = search.strip()
    found = movies[movies["title"].str.contains(search, case=False, regex=False)] if search else movies
    return [movie_info(row) for row in found.head(20).to_dict("records")]


@app.get("/api/movies/{imdb_id}")
def analyze_movie(imdb_id: str):
    match = movies[movies["imdb_id"] == imdb_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Movie not found.")

    movie_reviews = reviews[reviews["imdb_id"] == imdb_id]
    summary, results = analyze_reviews(movie_reviews["text"].tolist())

    # The reviewers' own star ratings are NOT used for prediction, only shown for comparison.
    for result, stars in zip(results, movie_reviews["stars"]):
        result["actual_stars"] = int(stars)
    summary["actual_average_stars"] = round(float(movie_reviews["stars"].mean()), 1)
    return {"movie": movie_info(match.iloc[0]), "summary": summary, "reviews": results}


class ReviewsRequest(BaseModel):
    reviews: list[str]


@app.post("/api/analyze")
def analyze_own_reviews(request: ReviewsRequest):
    texts = [text.strip() for text in request.reviews if clean_text(text)]
    if not texts:
        raise HTTPException(status_code=400, detail="Please enter at least one review containing words.")
    summary, results = analyze_reviews(texts)
    return {"summary": summary, "reviews": results}


@app.get("/api/health")
def health():
    return {"status": "ok", "movies": len(movies)}
