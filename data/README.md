# Dataset

`sample_imdb_reviews.csv` is a small, balanced IMDb-style review sample included only to let the project train and run immediately. It has `review` and `sentiment` columns and is not intended to represent full IMDb benchmark performance.

For a stronger portfolio result, download the [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), save it as `data/imdb_dataset.csv`, and update `DATA_PATH` in `ml/train.py` to point to that file. Do not commit the large downloaded CSV.
