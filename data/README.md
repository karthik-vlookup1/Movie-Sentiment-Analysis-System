# Data

Nothing large is committed. These files are created by the ML scripts:

| Path | Created by | Contents |
| --- | --- | --- |
| `raw/aclImdb_v1.tar.gz` | `python -m ml.download_data` | [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (Maas et al., 2011) |
| `raw/title.basics.tsv.gz` | `python -m ml.download_data` | [IMDb title list](https://developer.imdb.com/non-commercial-datasets/), used for movie names and years |
| `aclImdb/` | `python -m ml.download_data` | The extracted review text files |
| `processed/reviews.csv.gz` | `python -m ml.prepare_data` | One row per review: split, imdb_id, stars, label, text |
| `processed/movies.csv` | `python -m ml.prepare_data` | imdb_id, title, year |

aclImdb is distributed for research use (cite Maas et al., 2011). The IMDb title list is for
personal and non-commercial use only.
