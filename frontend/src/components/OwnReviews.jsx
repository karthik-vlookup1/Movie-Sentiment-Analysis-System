import { useState } from 'react';
import { analyzeReviews } from '../api';
import RatingCard from './RatingCard';
import ReviewList from './ReviewList';

const EXAMPLE = `An absolute triumph. The acting is superb and the story kept me hooked.
I wanted to like it, but it was slow, boring and the ending made no sense.
Not perfect, but very entertaining and the cast clearly had fun.
A brilliant, moving film. One of the best I have seen this year.`;

function OwnReviews() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    const reviews = text.split('\n').filter((line) => line.trim() !== ''); // one review per line
    if (reviews.length === 0) {
      setError('Please enter at least one review.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      setResult(await analyzeReviews(reviews));
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section>
      <form className="card" onSubmit={handleSubmit}>
        <label htmlFor="reviews">Type or paste reviews, one review per line</label>
        <textarea
          id="reviews"
          rows={7}
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="This movie was fantastic, the acting was excellent."
        />
        <div className="form-buttons">
          <button type="button" className="secondary" onClick={() => setText(EXAMPLE)}>Use example</button>
          <button type="submit" disabled={loading}>{loading ? 'Analysing…' : 'Analyse'}</button>
        </div>
        {error && <p className="error">{error}</p>}
      </form>

      {result && (
        <>
          <RatingCard summary={result.summary} />
          <ReviewList reviews={result.reviews} />
        </>
      )}
    </section>
  );
}

export default OwnReviews;
