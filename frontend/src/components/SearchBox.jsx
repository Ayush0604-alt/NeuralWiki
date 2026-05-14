import { useState } from "react";

function SearchBox() {

  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [message, setMessage] = useState("");

  const handleSearch = async () => {

    try {

      const response = await fetch(
        `http://127.0.0.1:8000/search?query=${query}`
      );

      const data = await response.json();

      console.log(data);

      if (Array.isArray(data.results)) {

        setResults(data.results);

        setMessage(
          `${data.results.length} results found`
        );

      } else {

        setResults([]);

        setMessage("Invalid response");
      }

    } catch (error) {

      console.error(error);

      setMessage("Search failed");
    }
  };

  return (
    <div>

      <h2>Semantic Search</h2>

      <input
        type="text"
        placeholder="Search..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        style={{
          padding: "10px",
          width: "300px"
        }}
      />

      <button
        onClick={handleSearch}
        style={{
          marginLeft: "10px",
          padding: "10px"
        }}
      >
        Search
      </button>

      <p>{message}</p>

      <div style={{ marginTop: "20px" }}>

        {results.map((result, index) => (

          <div
            key={index}
            style={{
              background: "#f4f4f4",
              padding: "20px",
              marginBottom: "20px",
              borderRadius: "10px"
            }}
          >

            <h3>
              {result.source}
            </h3>

            <p>
              Chunk ID: {result.chunk_id}
            </p>

            <p>
              Similarity: {result.similarity_score}
            </p>

            <p>
              {result.content}
            </p>

          </div>

        ))}

      </div>

    </div>
  );
}

export default SearchBox;