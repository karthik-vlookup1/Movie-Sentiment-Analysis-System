import { useState } from 'react';

const API_URL = 'http://localhost:8000';

function SentimentAnalyzer() {
  const [review, setReview] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    setError('');
    setResult(null);
    if (!review.trim()) {
      setError('Please enter a movie review first.');
      return;
    }
    setIsLoading(true);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Unable to analyze this review.');
      setResult(data);
    } catch (requestError) {
      setError(requestError.message || 'Could not connect to the API. Start the FastAPI server first.');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <label htmlFor="review">Movie review</label>
      <textarea
        id="review"
        value={review}
        onChange={(event) => setReview(event.target.value)}
        placeholder="The movie was fantastic and the actors were excellent."
        rows="7"
      />
      <button type="submit" disabled={isLoading}>
        {isLoading ? 'Analyzing...' : 'Analyze Sentiment'}
      </button>
      {error && <p className="error" role="alert">{error}</p>}
      {result && (
        <section className={`result-card ${result.sentiment.toLowerCase()}`} aria-live="polite">
          <span>Sentiment</span>
          <strong>{result.sentiment}</strong>
          <p>Confidence: <b>{Math.round(result.confidence * 100)}%</b></p>
        </section>
      )}
    </form>
  );
}

export default SentimentAnalyzer;
