"""Step 2: prepare the dataset.

Reads the raw aclImdb text files and writes two CSV files:

* data/processed/reviews.csv.gz - one row per review: split, imdb_id, stars, label, text
* data/processed/movies.csv     - imdb_id, title, year for every movie in the dataset

aclImdb already comes with a train split and a test split of 25,000 reviews each, and the
two splits contain *different movies*. We keep that split: the model is trained on the
train movies, and the app only shows test movies, which the model has never seen.

Run:  python -m ml.prepare_data
"""

import re

import pandas as pd

from ml.config import ACLIMDB_DIR, MOVIES_PATH, RAW_DIR, REVIEWS_PATH


def load_split(split):
    """Read every review file of one split (train or test)."""
    rows = []
    for folder, label in [("pos", 1), ("neg", 0)]:
        # urls_pos.txt line N is the IMDb page of review N, e.g. .../title/tt0453418/usercomments
        urls = (ACLIMDB_DIR / split / f"urls_{folder}.txt").read_text(encoding="utf-8").splitlines()
        for path in (ACLIMDB_DIR / split / folder).glob("*.txt"):
            review_number, stars = path.stem.split("_")  # file name is "<review number>_<stars>.txt"
            rows.append({
                "split": split,
                "imdb_id": re.search(r"tt\d+", urls[int(review_number)]).group(),
                "stars": int(stars),  # the star rating (1-10) the reviewer gave
                "label": label,       # 1 = positive (7-10 stars), 0 = negative (1-4 stars)
                "text": path.read_text(encoding="utf-8").strip(),
            })
    return pd.DataFrame(rows)


def load_movie_titles(imdb_ids):
    """Look up title and year in the (large) IMDb title file, reading it in chunks."""
    chunks = pd.read_csv(RAW_DIR / "title.basics.tsv.gz", sep="\t", quoting=3, na_values="\\N",
                         usecols=["tconst", "primaryTitle", "startYear"], dtype=str, chunksize=500_000)
    movies = pd.concat(chunk[chunk["tconst"].isin(imdb_ids)] for chunk in chunks)
    return movies.rename(columns={"tconst": "imdb_id", "primaryTitle": "title", "startYear": "year"})


def main():
    reviews = pd.concat([load_split("train"), load_split("test")], ignore_index=True)
    print(f"Loaded {len(reviews)} reviews")

    # Data cleaning: remove empty and duplicate reviews. Train rows come first and
    # drop_duplicates keeps the first copy, so no test review is a copy of a train review.
    reviews = reviews[reviews["text"].str.len() > 0]
    reviews = reviews.drop_duplicates(subset="text")
    # The two splits use different movies except for one; drop that movie's test reviews
    # so that every test movie is truly unseen by the model.
    train_movies = set(reviews.loc[reviews["split"] == "train", "imdb_id"])
    reviews = reviews[~((reviews["split"] == "test") & reviews["imdb_id"].isin(train_movies))]
    print(f"After removing empty and duplicate reviews: {len(reviews)}")

    print(reviews.groupby(["split", "label"]).size().unstack().rename(columns={0: "negative", 1: "positive"}))

    print("Looking up movie titles ...")
    movies = load_movie_titles(set(reviews["imdb_id"]))
    print(f"Found titles for {len(movies)} movies")

    REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(REVIEWS_PATH, index=False)
    movies.to_csv(MOVIES_PATH, index=False)
    print(f"Saved {REVIEWS_PATH} and {MOVIES_PATH}")


if __name__ == "__main__":
    main()
