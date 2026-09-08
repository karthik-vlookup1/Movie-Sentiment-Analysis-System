import SentimentAnalyzer from './components/SentimentAnalyzer';

function App() {
  return (
    <main className="page-shell">
      <section className="analyzer-panel">
        <p className="eyebrow">SIMPLE NLP PROJECT</p>
        <h1>Movie Sentiment Analyzer</h1>
        <p className="subtitle">Analyze whether a movie review is positive or negative.</p>
        <SentimentAnalyzer />
      </section>
    </main>
  );
}

export default App;
