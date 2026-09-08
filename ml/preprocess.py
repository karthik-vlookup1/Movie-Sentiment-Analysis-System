"""Small, reusable NLP preprocessing helpers."""

import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def ensure_nltk_resources() -> None:
    """Download the two small NLTK resources only when they are missing."""
    resources = [("corpora/stopwords", "stopwords"), ("tokenizers/punkt", "punkt")]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def preprocess_text(text: str) -> str:
    """Return lowercase review text with HTML, noise, punctuation and stopwords removed."""
    if not isinstance(text, str):
        return ""
    ensure_nltk_resources()
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    english_stopwords = set(stopwords.words("english"))
    return " ".join(token for token in tokens if token not in english_stopwords)
