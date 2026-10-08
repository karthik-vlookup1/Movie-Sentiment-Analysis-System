"""Basic tests for preprocessing, rating and the API.

Run from the project root:  python -m pytest tests
The API tests need the trained model and prepared data (see README).
"""

import pytest

from ml.config import MODEL_PATH, REVIEWS_PATH
from ml.preprocess import clean_text
from ml.rating import estimate_rating, get_verdict, summarize_reviews


def test_clean_text_removes_html_punctuation_and_stopwords():
    assert clean_text("The movie was GREAT!<br /><br />Loved it.") == "movie great loved"


def test_clean_text_keeps_negations():
    assert clean_text("I didn't like it, it was not good") == "did not like not good"


def test_rating_formula_and_verdict():
    assert estimate_rating([0.0, 0.0]) == 1.0
    assert estimate_rating([1.0, 1.0]) == 10.0
    assert estimate_rating([0.9, 0.7]) == 8.2
    assert get_verdict(8.2) == "Good"
    assert get_verdict(6.0) == "Average"
    assert get_verdict(3.5) == "Bad"


def test_summarize_reviews_counts_positive_and_negative():
    summary = summarize_reviews([0.9, 0.8, 0.1, 0.6])
    assert summary["positive_count"] == 3
    assert summary["negative_count"] == 1
    assert summary["positive_percent"] == 75.0
    assert summary["rating"] == 6.4
    assert summary["verdict"] == "Average"


needs_model = pytest.mark.skipif(not (MODEL_PATH.exists() and REVIEWS_PATH.exists()),
                                 reason="train the model first")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from backend.main import app
    return TestClient(app)


@needs_model
def test_search_and_analyze_movie(client):
    movies = client.get("/api/movies", params={"search": "the"}).json()
    assert movies and all("the" in movie["title"].lower() for movie in movies)

    result = client.get(f"/api/movies/{movies[0]['imdb_id']}").json()
    summary = result["summary"]
    assert summary["review_count"] == len(result["reviews"]) >= 5
    assert 1 <= summary["rating"] <= 10
    assert summary["verdict"] in {"Good", "Average", "Bad"}


@needs_model
def test_unknown_movie_returns_404(client):
    assert client.get("/api/movies/tt0000000").status_code == 404


@needs_model
def test_analyze_own_reviews(client):
    response = client.post("/api/analyze", json={"reviews": [
        "A wonderful film with brilliant acting. I loved every minute.",
        "Boring, badly written and far too long. A complete waste of time.",
    ]})
    sentiments = [review["sentiment"] for review in response.json()["reviews"]]
    assert sentiments == ["Positive", "Negative"]
    assert client.post("/api/analyze", json={"reviews": ["!!!", "  "]}).status_code == 400
