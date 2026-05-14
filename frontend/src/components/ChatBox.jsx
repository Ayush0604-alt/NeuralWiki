import { useState } from "react";

function ChatBox() {

  const [query, setQuery] = useState("");

  const [messages, setMessages] = useState([]);

  const [loading, setLoading] = useState(false);

  const handleChat = async () => {

    if (!query.trim()) return;

    // Add user message
    const userMessage = {
      role: "user",
      content: query
    };

    setMessages((prev) => [
      ...prev,
      userMessage
    ]);

    setLoading(true);

    try {

      const res = await fetch(
        "http://127.0.0.1:8000/chat",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({
            query
          })
        }
      );

      const data = await res.json();

      const aiMessage = {
        role: "ai",
        content: data.response,
        sources: data.sources || []
      };

      setMessages((prev) => [
        ...prev,
        aiMessage
      ]);

    } catch (error) {

      console.error(error);

      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          content: "Error generating response."
        }
      ]);
    }

    setLoading(false);

    setQuery("");
  };


  const clearConversation = async () => {

    try {

      await fetch(
        "http://127.0.0.1:8000/clear-memory",
        {
          method: "POST"
        }
      );

      setMessages([]);

    } catch (error) {

      console.error(error);
    }
  };


  return (
    <div>

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center"
        }}
      >

        <h2>NeuralWiki AI Chat</h2>

        <button
          onClick={clearConversation}
          style={{
            padding: "10px",
            background: "#ff4d4d",
            color: "white",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer"
          }}
        >
          Clear Chat
        </button>

      </div>


      {/* Chat Area */}

      <div
        style={{
          height: "500px",
          overflowY: "auto",
          border: "1px solid #ccc",
          borderRadius: "10px",
          padding: "20px",
          background: "#fafafa",
          marginBottom: "20px"
        }}
      >

        {messages.map((message, index) => (

          <div
            key={index}
            style={{
              display: "flex",
              justifyContent:
                message.role === "user"
                  ? "flex-end"
                  : "flex-start",

              marginBottom: "20px"
            }}
          >

            <div
              style={{
                maxWidth: "70%",
                padding: "15px",
                borderRadius: "15px",

                background:
                  message.role === "user"
                    ? "#007bff"
                    : "#e5e5ea",

                color:
                  message.role === "user"
                    ? "white"
                    : "black"
              }}
            >

              <p
                style={{
                  margin: 0,
                  whiteSpace: "pre-wrap"
                }}
              >
                {message.content}
              </p>

              {/* Sources */}

              {message.sources &&
                message.sources.length > 0 && (

                <div
                  style={{
                    marginTop: "10px",
                    fontSize: "12px"
                  }}
                >

                  <strong>Sources:</strong>

                  {message.sources.map(
                    (source, idx) => (

                    <div key={idx}>

                      {source.source}
                      {" | "}
                      Chunk {source.chunk_id}

                    </div>

                  ))}

                </div>
              )}

            </div>

          </div>

        ))}


        {loading && (

          <div
            style={{
              marginTop: "10px"
            }}
          >

            <p>
              AI is thinking...
            </p>

          </div>

        )}

      </div>


      {/* Input Area */}

      <div
        style={{
          display: "flex",
          gap: "10px"
        }}
      >

        <input
          type="text"
          placeholder="Ask NeuralWiki..."
          value={query}
          onChange={(e) =>
            setQuery(e.target.value)
          }

          style={{
            flex: 1,
            padding: "15px",
            borderRadius: "10px",
            border: "1px solid #ccc"
          }}
        />

        <button
          onClick={handleChat}

          style={{
            padding: "15px 25px",
            background: "#007bff",
            color: "white",
            border: "none",
            borderRadius: "10px",
            cursor: "pointer"
          }}
        >
          Send
        </button>

      </div>

    </div>
  );
}

export default ChatBox;