"""Step 1: download the datasets.

* aclImdb (Stanford Large Movie Review Dataset): 50,000 IMDb reviews labelled positive or
  negative. Each review also stores the IMDb id of its movie, so reviews can be grouped
  by movie.
* IMDb title list (title.basics.tsv.gz): used only to look up each movie's name and year.

Run:  python -m ml.download_data
"""

import tarfile
import urllib.request

from ml.config import ACLIMDB_DIR, DATA_DIR, RAW_DIR

ACLIMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_TITLES_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"


def download(url, destination):
    if destination.exists():
        print(f"Already downloaded: {destination.name}")
        return
    print(f"Downloading {url} ...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def main():
    archive = RAW_DIR / "aclImdb_v1.tar.gz"
    download(ACLIMDB_URL, archive)
    download(IMDB_TITLES_URL, RAW_DIR / "title.basics.tsv.gz")

    if not ACLIMDB_DIR.exists():
        print("Extracting reviews (takes a minute) ...")
        with tarfile.open(archive) as tar:
            # Skip the 50,000 unlabelled reviews in train/unsup: we only need labelled ones.
            members = [m for m in tar.getmembers() if "/unsup/" not in m.name]
            tar.extractall(DATA_DIR, members=members, filter="data")
    print("Done.")


if __name__ == "__main__":
    main()
