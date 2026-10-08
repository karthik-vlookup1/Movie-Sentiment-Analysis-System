import { useEffect, useState } from 'react';
import { searchMovies } from '../api';

function MovieSearch({ selectedId, onSelect }) {
  const [search, setSearch] = useState('');
  const [movies, setMovies] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    let ignore = false; // ignore old responses if the user has typed something new
    // Wait until the user stops typing for 300 ms before calling the API.
    const timer = setTimeout(() => {
      searchMovies(search)
        .then((results) => {
          if (!ignore) {
            setMovies(results);
            setError('');
          }
        })
        .catch((err) => !ignore && setError(err.message));
    }, 300);
    return () => {
      ignore = true;
      clearTimeout(timer);
    };
  }, [search]);

  return (
    <aside className="card search">
      <input
        type="search"
        placeholder="Search a movie, e.g. Batman"
        value={search}
        onChange={(event) => setSearch(event.target.value)}
        aria-label="Search movies"
      />
      {error && <p className="error">{error}</p>}
      {!error && movies.length === 0 && <p className="muted">No movies found.</p>}
      <ul className="movie-list">
        {movies.map((movie) => (
          <li key={movie.imdb_id}>
            <button
              type="button"
              className={movie.imdb_id === selectedId ? 'selected' : ''}
              onClick={() => onSelect(movie.imdb_id)}
            >
              <strong>{movie.title}</strong>
              <span className="muted">{movie.year && `${movie.year} · `}{movie.review_count} reviews</span>
            </button>
          </li>
        ))}
      </ul>
    </aside>
  );
}

export default MovieSearch;
