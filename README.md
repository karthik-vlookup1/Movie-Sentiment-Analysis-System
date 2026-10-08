# Movie Sentiment Analysis

A student NLP / machine-learning project. You pick a movie; the app takes that movie's IMDb
reviews, classifies every review as **positive** or **negative** with a **TF-IDF + Logistic
Regression** model, combines the predictions into an **estimated rating out of 10**, and shows
a **Good / Average / Bad** verdict.

```text
User selects a movie (React)
  -> backend gets that movie's reviews            (backend/main.py)
  -> NLP preprocessing of each review             (ml/preprocess.py)
  -> TF-IDF turns text into numbers               (saved model pipeline)
  -> Logistic Regression: P(positive) per review  (saved model pipeline, trained by ml/train.py)
  -> aggregate all reviews -> rating out of 10    (ml/rating.py)
  -> Good / Average / Bad                         (ml/rating.py)
  -> shown in the browser                         (frontend/src)
```

## Results

Measured on the **24,655 test reviews**. Those reviews come from movies the model never saw
during training (`results/model_metrics.txt`).

| Metric | Value |
| --- | --- |
| Accuracy | 89.2% |
| Precision | 89.3% |
| Recall | 89.3% |
| F1-score | 89.3% |

Confusion matrix (`results/confusion_matrix.png`):

|  | Predicted negative | Predicted positive |
| --- | --- | --- |
| **Actual negative** | 10,930 | 1,330 |
| **Actual positive** | 1,328 | 11,067 |

Model choice (5-fold cross-validation on the training set): Logistic Regression 87.4%,
Multinomial Naive Bayes 83.8%, so Logistic Regression was chosen.

Movie-level check on the 1,594 test movies with at least 5 reviews: the estimated rating is on
average **0.76 stars** away from the average star rating the reviewers actually gave. The
Good/Average/Bad verdict matches the verdict from the real stars for **87.5%** of movies.

## Dataset

[Stanford Large Movie Review Dataset (aclImdb)](https://ai.stanford.edu/~amaas/data/sentiment/)
(Maas et al., 2011): 50,000 IMDb reviews.

* Labels: reviews with 7-10 stars are **positive**, 1-4 stars are **negative**.
* 25,000 train and 25,000 test reviews. The two splits contain **different movies**.
* Every review records the IMDb id of its movie, so reviews can be grouped per movie.
  Movie titles come from the [IMDb title list](https://developer.imdb.com/non-commercial-datasets/).
* Cleaning (`ml/prepare_data.py`): empty and duplicate reviews are removed, plus the test
  reviews of the one movie that also appears in the train split.

The app shows the **test** movies with at least 5 reviews: 1,506 movies and 19,330 reviews.
Their reviews were never used for training.

## How it works

**1. NLP preprocessing** (`ml/preprocess.py`, function `clean_text`)
lowercase → remove HTML tags (`<br />`) → expand negations (`didn't` → `did not`) →
remove punctuation and numbers → split into words → remove stopwords, **but keep negation
words** like *not*, *no* and *never*, because removing them would turn "not good" into "good".

**2. TF-IDF** (`ml/train.py`). `TfidfVectorizer` turns each cleaned review into a vector of 50,000
numbers. Each number is the TF-IDF weight of one word or word pair (`ngram_range=(1, 2)`, so
"not good" is a feature of its own). Words that are frequent in a review but rare across all
reviews get high weights.

**3. Logistic Regression** (`ml/train.py`). Learns one weight per feature and outputs
P(positive) for a review. The regularisation strength `C` is chosen from 0.5, 1, 2, 5 and 10
with 5-fold cross-validation (`GridSearchCV`); the best was `C = 2`. TF-IDF and Logistic
Regression are saved together as one scikit-learn `Pipeline` in
`backend/model/sentiment_model.joblib`.

**4. Classify each review**. If P(positive) ≥ 0.5 the review is Positive, otherwise Negative.

**5. Aggregate and rate** (`ml/rating.py`):

```text
average = average P(positive) over all reviews of the movie     (0 to 1)
rating  = 1 + 9 × average                                       (1 to 10)
verdict = Good if rating ≥ 7,  Average if 5 ≤ rating < 7,  Bad if rating < 5
```

The rating averages the probabilities instead of only counting positive reviews, so a review
the model is unsure about (P = 0.55) counts less than a clear one (P = 0.99). On the test
movies this was closer to the real star ratings (0.76 vs 0.91 stars error).

## Project structure

```text
ml/
  config.py          file paths
  download_data.py   step 1: download the datasets
  prepare_data.py    step 2: read raw files -> data/processed/reviews.csv.gz, movies.csv
  preprocess.py      NLP preprocessing (clean_text), used in training and in the backend
  train.py           step 3: TF-IDF + Logistic Regression, cross-validation, save model
  evaluate.py        step 4: test-set metrics, confusion matrix, movie-level check
  rating.py          combine review predictions -> rating and Good/Average/Bad
backend/
  main.py            FastAPI app: search movies, analyse a movie, analyse your own reviews
  model/             the saved model (created by ml/train.py)
frontend/src/
  App.jsx            page layout and tabs
  api.js             fetch() calls to the backend
  components/        MovieSearch, MovieAnalysis, RatingCard, ReviewList, OwnReviews
notebooks/           walkthrough notebook of the whole pipeline
results/             model_metrics.txt and confusion_matrix.png
tests/               pytest tests for preprocessing, rating and the API
```

## How to run (Windows)

You need Python 3.11+ and Node.js 20+. Run everything from the project folder.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python -m ml.download_data   # ~300 MB download, only needed once
python -m ml.prepare_data    # ~1 minute
python -m ml.train           # ~2-3 minutes
python -m ml.evaluate        # writes results/

uvicorn backend.main:app --reload       # API on http://localhost:8000 (docs at /docs)
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev                  # open http://localhost:5173
```

Tests: `python -m pytest tests`

## API

The React app calls the backend with `fetch` (see `frontend/src/api.js`). All responses are JSON.

| Method | URL | What it does |
| --- | --- | --- |
| GET | `/api/movies?search=batman` | Movies whose title contains the text (top 20, most reviews first) |
| GET | `/api/movies/{imdb_id}` | The movie's reviews, each one classified, plus the rating and verdict |
| POST | `/api/analyze` | Same analysis for your own reviews: `{"reviews": ["...", "..."]}` |
| GET | `/api/health` | Check that the API is running |

## Limitations

* The reviews come from a fixed dataset (aclImdb, up to 30 reviews per movie), not live from
  IMDb. New movies can still be analysed by pasting their reviews into the
  "Analyse your own reviews" tab.
* The model only sees words. It can miss sarcasm and mixed reviews ("great actors, terrible
  film"). This causes most of the ~11% errors.
* aclImdb has no neutral (5-6 star) reviews, so every review is classified as either positive
  or negative.
