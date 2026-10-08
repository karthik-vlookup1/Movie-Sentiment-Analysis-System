import { useState } from 'react';

const PREVIEW_LENGTH = 300;

function ReviewItem({ review }) {
  const [expanded, setExpanded] = useState(false);
  const isLong = review.text.length > PREVIEW_LENGTH;
  const text = isLong && !expanded ? `${review.text.slice(0, PREVIEW_LENGTH)}…` : review.text;

  return (
    <li className="review">
      <div className="review-header">
        <span className={`label ${review.sentiment.toLowerCase()}`}>{review.sentiment}</span>
        <span className="muted small">P(positive) = {review.positive_probability}</span>
        {review.actual_stars != null && <span className="muted small">Reviewer gave {review.actual_stars}/10</span>}
      </div>
      <p className="review-text">{text}</p>
      {isLong && (
        <button type="button" className="link" onClick={() => setExpanded(!expanded)}>
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </li>
  );
}

function ReviewList({ reviews }) {
  return (
    <div className="card">
      <h3>Prediction for each review</h3>
      <ul className="review-list">
        {reviews.map((review, index) => <ReviewItem key={index} review={review} />)}
      </ul>
    </div>
  );
}

export default ReviewList;
