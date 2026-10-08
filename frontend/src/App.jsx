import { useState } from 'react';
import MovieAnalysis from './components/MovieAnalysis';
import MovieSearch from './components/MovieSearch';
import OwnReviews from './components/OwnReviews';

function App() {
  const [tab, setTab] = useState('movies');
  const [selectedId, setSelectedId] = useState(null);

  return (
    <div className="app">
      <header className="header">
        <h1>Movie Sentiment Analyzer</h1>
        <p>Estimates a movie&apos;s rating from the sentiment of its reviews (NLP + TF-IDF + Logistic Regression).</p>
        <nav className="tabs">
          <button type="button" className={tab === 'movies' ? 'active' : ''} onClick={() => setTab('movies')}>
            Rate a movie
          </button>
          <button type="button" className={tab === 'own' ? 'active' : ''} onClick={() => setTab('own')}>
            Analyse your own reviews
          </button>
        </nav>
      </header>

      {/* "hidden" keeps both tabs mounted, so switching tabs does not lose what you typed. */}
      <main className="movie-layout" hidden={tab !== 'movies'}>
        <MovieSearch selectedId={selectedId} onSelect={setSelectedId} />
        <MovieAnalysis imdbId={selectedId} />
      </main>
      <main hidden={tab !== 'own'}>
        <OwnReviews />
      </main>

      <footer className="footer">
        How it works: reviews → NLP preprocessing → TF-IDF features → Logistic Regression (positive / negative)
        → average of all reviews → rating out of 10 → Good / Average / Bad.
        Reviews from the Stanford IMDb dataset (movies the model never saw during training).
      </footer>
    </div>
  );
}

export default App;
