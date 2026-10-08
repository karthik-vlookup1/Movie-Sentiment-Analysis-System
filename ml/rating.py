"""Combine the predictions for many reviews into one movie rating and verdict.

For every review the model gives P(positive), a probability between 0 and 1.

1. Review label:  Positive if P(positive) >= 0.5, otherwise Negative.
2. Average:       average of P(positive) over all reviews of the movie (0 to 1).
3. Rating:        rating = 1 + 9 * average   -> maps 0..1 onto a 1..10 scale
                  (all reviews surely negative -> 1/10, all surely positive -> 10/10).
4. Verdict:       rating >= 7 -> Good,  5 <= rating < 7 -> Average,  rating < 5 -> Bad.

We average the probabilities rather than just counting positive reviews because a review
the model is only 55% sure about should not count the same as one it is 99% sure about.
On the test movies this was also closer to the reviewers' real star ratings
(see ml/evaluate.py).
"""

GOOD_RATING = 7.0
AVERAGE_RATING = 5.0
MIN_REVIEWS = 5  # movies with fewer reviews are not shown: too few for a meaningful average


def estimate_rating(positive_probabilities):
    average = sum(positive_probabilities) / len(positive_probabilities)
    return round(1 + 9 * average, 1)


def get_verdict(rating):
    if rating >= GOOD_RATING:
        return "Good"
    if rating >= AVERAGE_RATING:
        return "Average"
    return "Bad"


def summarize_reviews(positive_probabilities):
    """Return the movie-level summary for a list of per-review P(positive) values."""
    total = len(positive_probabilities)
    if total == 0:
        raise ValueError("At least one review is needed to rate a movie.")
    positive = sum(1 for p in positive_probabilities if p >= 0.5)
    rating = estimate_rating(positive_probabilities)
    return {
        "review_count": total,
        "positive_count": positive,
        "negative_count": total - positive,
        "positive_percent": round(100 * positive / total, 1),
        "average_positive_probability": round(sum(positive_probabilities) / total, 3),
        "rating": rating,
        "verdict": get_verdict(rating),
    }
