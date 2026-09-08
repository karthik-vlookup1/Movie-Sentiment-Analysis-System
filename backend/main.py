"""FastAPI application serving the trained sentiment model."""

from pathlib import Path
import sys

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import PredictionResponse, ReviewRequest

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "backend" / "model"
sys.path.insert(0, str(ROOT_DIR / "ml"))
from preprocess import preprocess_text  # noqa: E402

app = FastAPI(title="Movie Sentiment Analysis API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_artifacts():
    try:
        model = joblib.load(MODEL_DIR / "sentiment_model.pkl")
        vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError as error:
        raise RuntimeError("Model files not found. Run 'python ml/train.py' first.") from error


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest):
    try:
        model, vectorizer = load_artifacts()
        cleaned_review = preprocess_text(request.review)
        if not cleaned_review:
            raise HTTPException(status_code=422, detail="Review has no usable words after preprocessing.")
        features = vectorizer.transform([cleaned_review])
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][prediction])
        return PredictionResponse(
            review=request.review,
            sentiment="Positive" if prediction == 1 else "Negative",
            confidence=round(probability, 4),
        )
    except HTTPException:
        raise
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
