"""Step 4: evaluate the trained model on the test split (reviews it has never seen).

1. Review level: accuracy, precision, recall, F1-score and a confusion matrix.
2. Movie level:  for every test movie with at least MIN_REVIEWS reviews, compare our
   estimated rating with the average star rating its reviewers actually gave.

Results are written to results/model_metrics.txt and results/confusion_matrix.png.

Run:  python -m ml.evaluate
"""

import joblib
import matplotlib

matplotlib.use("Agg")  # save plots to files without opening a window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ml.config import MODEL_PATH, RESULTS_DIR, REVIEWS_PATH  # noqa: E402
from ml.preprocess import clean_text  # noqa: E402
from ml.rating import MIN_REVIEWS, estimate_rating, get_verdict  # noqa: E402


def save_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.title("Confusion matrix (test reviews)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def movie_level_check(test):
    """Compare estimated movie ratings with the reviewers' real average star ratings."""
    movies = test.groupby("imdb_id").agg(
        reviews=("label", "size"),
        estimated=("probability", estimate_rating),
        positive_share=("predicted", "mean"),
        actual_stars=("stars", "mean"),
    )
    movies = movies[movies["reviews"] >= MIN_REVIEWS]
    error = (movies["estimated"] - movies["actual_stars"]).abs().mean()
    # Alternative formula, for comparison: rating from the share of positive reviews only.
    share_error = (1 + 9 * movies["positive_share"] - movies["actual_stars"]).abs().mean()
    verdict_match = (movies["estimated"].map(get_verdict) == movies["actual_stars"].map(get_verdict)).mean()
    return [
        f"Movie-level check ({len(movies)} test movies with at least {MIN_REVIEWS} reviews)",
        f"  Average error of estimated rating vs reviewers' real stars: {error:.2f} stars",
        f"  (using the share of positive reviews instead: {share_error:.2f} stars)",
        f"  Good/Average/Bad verdict matches the verdict from real stars: {verdict_match:.1%}",
    ]


def main():
    model = joblib.load(MODEL_PATH)
    reviews = pd.read_csv(REVIEWS_PATH)
    test = reviews[reviews["split"] == "test"].copy()

    test["probability"] = model.predict_proba(test["text"].apply(clean_text))[:, 1]
    test["predicted"] = (test["probability"] >= 0.5).astype(int)
    y_true, y_pred = test["label"], test["predicted"]

    lines = [
        f"Model: TF-IDF + Logistic Regression (C={model.named_steps['model'].C})",
        f"Test set: {len(test)} reviews of movies not used in training",
        "",
        f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}",
        f"Precision: {precision_score(y_true, y_pred):.4f}",
        f"Recall:    {recall_score(y_true, y_pred):.4f}",
        f"F1-score:  {f1_score(y_true, y_pred):.4f}",
        "",
        classification_report(y_true, y_pred, target_names=["negative", "positive"], digits=4),
        *movie_level_check(test),
    ]
    report = "\n".join(lines) + "\n"

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "model_metrics.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(y_true, y_pred)
    print(report)
    print(f"Saved {RESULTS_DIR / 'model_metrics.txt'} and {RESULTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
