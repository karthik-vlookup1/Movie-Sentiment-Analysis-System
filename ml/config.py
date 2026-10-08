"""File paths shared by the ML scripts and the backend."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ACLIMDB_DIR = DATA_DIR / "aclImdb"
REVIEWS_PATH = DATA_DIR / "processed" / "reviews.csv.gz"
MOVIES_PATH = DATA_DIR / "processed" / "movies.csv"

MODEL_PATH = ROOT_DIR / "backend" / "model" / "sentiment_model.joblib"
RESULTS_DIR = ROOT_DIR / "results"

RANDOM_STATE = 42
