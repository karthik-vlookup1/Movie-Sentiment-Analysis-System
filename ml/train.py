"""Step 3: train the sentiment model.

Model:  NLP preprocessing -> TF-IDF -> Logistic Regression

* Training data: the 25,000 reviews of the aclImdb *train* split (the test split is kept
  aside and only used by ml/evaluate.py).
* The regularisation strength C of Logistic Regression is chosen with 5-fold
  cross-validation on the training data (GridSearchCV).
* For comparison, Multinomial Naive Bayes is cross-validated on the same features.
* The best pipeline (TF-IDF vectorizer + Logistic Regression) is saved with joblib.

Run:  python -m ml.train
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from ml.config import MODEL_PATH, RANDOM_STATE, REVIEWS_PATH
from ml.preprocess import clean_text


def make_tfidf():
    return TfidfVectorizer(
        ngram_range=(1, 2),     # single words and word pairs, so "not good" is one feature
        min_df=2,               # ignore words that appear in only one review (typos, names)
        max_features=50_000,    # keep the 50,000 most frequent words/pairs
        sublinear_tf=True,      # use 1 + log(count), so repeating a word 10 times is not 10x stronger
    )


def main():
    reviews = pd.read_csv(REVIEWS_PATH)
    train = reviews[reviews["split"] == "train"]
    print(f"Training reviews: {len(train)} ({train['label'].sum()} positive)")

    print("NLP preprocessing ...")
    x_train = train["text"].apply(clean_text)
    y_train = train["label"]

    # A Pipeline keeps TF-IDF and the classifier together, so during cross-validation
    # the TF-IDF vocabulary is learned only from the training folds.
    pipeline = Pipeline([
        ("tfidf", make_tfidf()),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    print("Training Logistic Regression with 5-fold cross-validation (about 2 minutes) ...")
    search = GridSearchCV(pipeline, {"model__C": [0.5, 1, 2, 5, 10]}, cv=5, scoring="accuracy", n_jobs=-1)
    search.fit(x_train, y_train)
    for c, score in zip(search.cv_results_["param_model__C"], search.cv_results_["mean_test_score"]):
        print(f"  C={c:<4}  cross-validation accuracy = {score:.4f}")
    print(f"Best C = {search.best_params_['model__C']}")

    naive_bayes = Pipeline([("tfidf", make_tfidf()), ("model", MultinomialNB())])
    nb_score = cross_val_score(naive_bayes, x_train, y_train, cv=5, scoring="accuracy", n_jobs=-1).mean()
    print(f"\nCross-validation accuracy: Logistic Regression {search.best_score_:.4f}, "
          f"Naive Bayes {nb_score:.4f}")

    # GridSearchCV has already re-trained the best model on all 25,000 training reviews.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
