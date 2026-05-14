import { useState, useRef, useCallback } from "react";

const API = "http://127.0.0.1:8000";

function scoreClass(score) {
  if (score <= 0.5) return "score-high";
  if (score <= 1.0) return "score-mid";
  return "score-low";
}

function scoreLabel(score) {
  if (score <= 0.5) return "High match";
  if (score <= 1.0) return "Partial match";
  return "Low match";
}

function fileIcon(filename) {
  const ext = filename?.split(".").pop()?.toLowerCase();
  if (ext === "pdf") return "ti-file-type-pdf";
  if (ext === "md") return "ti-file-description";
  return "ti-file-text";
}

export default function SearchBox() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [error, setError] = useState("");
  const abortRef = useRef(null);

  const handleSearch = useCallback(async (q = query) => {
    const trimmed = q.trim();
    if (!trimmed) return;

    abortRef.current?.abort();
    abortRef.current = new AbortController();

    setLoading(true);
    setError("");
    setSearched(true);

    try {
      const res = await fetch(
        `${API}/search?query=${encodeURIComponent(trimmed)}`,
        { signal: abortRef.current.signal }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setResults(Array.isArray(data.results) ? data.results : []);
    } catch (err) {
      if (err.name === "AbortError") return;
      setError(err.message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [query]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleSearch();
  };

  const isEmpty = searched && !loading && results.length === 0 && !error;

  return (
    <div className="search-wrap">
      <div className="search-header">
        <div className="page-title" style={{ marginBottom: 12 }}>Semantic Search</div>
        <div className="search-row">
          <input
            type="search"
            placeholder="Search across all documents…"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            aria-label="Search query"
          />
          <button
            className="btn btn-primary"
            onClick={() => handleSearch()}
            disabled={loading || !query.trim()}
            aria-label="Run search"
          >
            {loading
              ? <i className="ti ti-loader-2" style={{ animation: "spin 1s linear infinite" }} aria-hidden="true" />
              : <i className="ti ti-search" aria-hidden="true" />}
            {loading ? "Searching…" : "Search"}
          </button>
        </div>
        {searched && !loading && !error && (
          <div className="search-meta">
            {results.length > 0
              ? `${results.length} result${results.length !== 1 ? "s" : ""} found`
              : "No results"}
          </div>
        )}
        {error && (
          <div className="search-meta" style={{ color: "var(--red)" }}>
            <i className="ti ti-alert-circle" aria-hidden="true" /> {error}
          </div>
        )}
      </div>

      <div className="search-results">
        {!searched && !loading && (
          <div className="empty-state">
            <i className="ti ti-search" />
            <h3>Search your knowledge base</h3>
            <p>Enter a query to find semantically similar passages across all uploaded documents.</p>
          </div>
        )}

        {isEmpty && (
          <div className="empty-state">
            <i className="ti ti-file-off" />
            <h3>No matches found</h3>
            <p>Try different keywords or upload more documents.</p>
          </div>
        )}

        {results.map((r, i) => (
          <div key={i} className="result-card">
            <div className="result-card-header">
              <div className="result-filename">
                <i className={`ti ${fileIcon(r.source)}`} aria-hidden="true" />
                {r.source}
              </div>
              <span className={`result-score ${scoreClass(r.similarity_score)}`}>
                {scoreLabel(r.similarity_score)}
              </span>
            </div>
            <div className="result-content">{r.content}</div>
            <div className="result-footer">
              <span className="tag-pill">Chunk {r.chunk_id}</span>
              <span className="tag-pill">Score {r.similarity_score}</span>
            </div>
          </div>
        ))}
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}