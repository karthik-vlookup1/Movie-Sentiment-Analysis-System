"""NLP preprocessing: turn a raw review into clean, lowercase words.

The same clean_text() function is used when training the model (ml/train.py) and when
predicting in the backend (backend/main.py), so the model always sees text cleaned the
same way.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negation words are normally in stopword lists, but removing them would turn
# "not good" into "good" and flip the sentiment, so we keep them.
NEGATION_WORDS = {"no", "not", "nor", "never", "nothing", "none", "nobody", "cannot"}
STOP_WORDS = ENGLISH_STOP_WORDS - NEGATION_WORDS


def clean_text(text: str) -> str:
    """Apply the NLP preprocessing steps to one review and return the cleaned text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()                                            # 1. lowercase
    text = re.sub(r"<[^>]+>", " ", text)                           # 2. remove HTML tags such as <br />
    text = text.replace("won't", "will not").replace("can't", "cannot")
    text = re.sub(r"n't\b", " not", text)                          # 3. expand negations: didn't -> did not
    text = re.sub(r"[^a-z\s]", " ", text)                          # 4. remove punctuation and numbers
    words = text.split()                                           # 5. tokenize into words
    words = [word for word in words if word not in STOP_WORDS]     # 6. remove stopwords (keep negations)
    return " ".join(words)
