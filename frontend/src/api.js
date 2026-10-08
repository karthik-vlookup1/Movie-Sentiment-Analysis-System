// All calls to the FastAPI backend live in this file.
const API_URL = 'http://localhost:8000/api';

async function request(path, options) {
  let response;
  try {
    response = await fetch(API_URL + path, options);
  } catch {
    throw new Error('Cannot reach the backend. Start it with: uvicorn backend.main:app --reload');
  }
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Something went wrong.');
  return data;
}

export function searchMovies(search) {
  return request(`/movies?search=${encodeURIComponent(search)}`);
}

export function getMovieAnalysis(imdbId) {
  return request(`/movies/${imdbId}`);
}

export function analyzeReviews(reviews) {
  return request('/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reviews }),
  });
}
