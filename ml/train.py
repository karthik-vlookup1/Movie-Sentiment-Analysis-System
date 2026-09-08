"""Train and save a simple TF-IDF + Logistic Regression sentiment classifier."""

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import preprocess_text  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "sample_imdb_reviews.csv"
MODEL_DIR = ROOT_DIR / "backend" / "model"
RESULTS_DIR = ROOT_DIR / "results"
RANDOM_STATE = 42


def main() -> None:
    dataset = pd.read_csv(DATA_PATH)
    if not {"review", "sentiment"}.issubset(dataset.columns):
        raise ValueError("Dataset must contain review and sentiment columns.")

    dataset = dataset.dropna(subset=["review", "sentiment"]).copy()
    dataset["clean_review"] = dataset["review"].apply(preprocess_text)
    dataset = dataset[dataset["clean_review"].str.len() > 0]
    dataset["label"] = dataset["sentiment"].map({"positive": 1, "negative": 0})
    if dataset["label"].isna().any():
        raise ValueError("Sentiment values must be 'positive' or 'negative'.")

    x_train, x_test, y_train, y_test = train_test_split(
        dataset["clean_review"], dataset["label"], test_size=0.25,
        random_state=RANDOM_STATE, stratify=dataset["label"]
    )
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(x_train_tfidf, y_train)
    predictions = model.predict(x_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "sentiment_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")

    matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.title("Sentiment Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    metrics_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}\n"
        f"Test samples: {len(y_test)}\n"
    )
    (RESULTS_DIR / "model_metrics.txt").write_text(metrics_text, encoding="utf-8")
    print(metrics_text)


if __name__ == "__main__":
    main()
