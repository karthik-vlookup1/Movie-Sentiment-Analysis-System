// Shows the movie-level result: estimated rating, Good/Average/Bad verdict and review counts.
function RatingCard({ title, summary }) {
  const verdictClass = summary.verdict.toLowerCase();

  return (
    <div className="card rating-card">
      {title && <h2>{title}</h2>}
      <div className="rating-row">
        <div className="rating">
          <span className="rating-number">{summary.rating.toFixed(1)}</span>
          <span className="muted">/ 10</span>
        </div>
        <span className={`verdict ${verdictClass}`}>{summary.verdict}</span>
      </div>

      <p>
        <strong>{summary.positive_count}</strong> positive and <strong>{summary.negative_count}</strong> negative
        out of {summary.review_count} reviews ({summary.positive_percent}% positive).
      </p>
      <div className="bar" aria-hidden="true">
        <div className="bar-positive" style={{ width: `${summary.positive_percent}%` }} />
      </div>

      <p className="muted small">
        Estimated rating = 1 + 9 × average positive probability ({summary.average_positive_probability}).
        Good ≥ 7, Average 5 – 6.9, Bad &lt; 5.
      </p>
      {summary.actual_average_stars != null && (
        <p className="muted small">
          For comparison, these reviewers actually gave an average of {summary.actual_average_stars} / 10 stars
          (not used by the model).
        </p>
      )}
    </div>
  );
}

export default RatingCard;
