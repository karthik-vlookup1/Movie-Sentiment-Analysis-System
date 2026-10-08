import { useEffect, useState } from 'react';
import { getMovieAnalysis } from '../api';
import RatingCard from './RatingCard';
import ReviewList from './ReviewList';

function MovieAnalysis({ imdbId }) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!imdbId) return undefined;
    let ignore = false; // ignore the answer for a movie that is no longer selected
    getMovieAnalysis(imdbId)
      .then((data) => {
        if (!ignore) {
          setResult(data);
          setError('');
        }
      })
      .catch((err) => !ignore && setError(err.message));
    return () => {
      ignore = true;
    };
  }, [imdbId]);

  if (!imdbId) {
    return <section className="card placeholder">Select a movie to see its estimated rating.</section>;
  }
  if (error) return <section className="card error">{error}</section>;
  if (!result || result.movie.imdb_id !== imdbId) {
    return <section className="card placeholder">Analysing reviews…</section>;
  }

  const { movie, summary, reviews } = result;
  return (
    <section>
      <RatingCard title={movie.year ? `${movie.title} (${movie.year})` : movie.title} summary={summary} />
      <ReviewList reviews={reviews} />
    </section>
  );
}

export default MovieAnalysis;
