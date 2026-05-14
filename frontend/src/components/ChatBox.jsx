import { useState, useRef, useEffect, useCallback } from "react";

const API = "http://127.0.0.1:8000";

function formatContent(text) {
  // simple code block detection
  return text.split("```").map((part, i) =>
    i % 2 === 1
      ? <pre key={i}>{part.trim()}</pre>
      : <span key={i}>{part}</span>
  );
}

function getScoreLabel(score) {
  if (score <= 0.5) return "High relevance";
  if (score <= 1.0) return "Medium";
  return "Low";
}

export default function ChatBox() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const abortRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleChat = useCallback(async () => {
    const q = query.trim();
    if (!q || loading) return;

    const userMsg = { role: "user", content: q };
    setMessages(prev => [...prev, userMsg]);
    setQuery("");
    setLoading(true);

    // cancel previous in-flight request
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
        signal: abortRef.current.signal,
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      setMessages(prev => [
        ...prev,
        {
          role: "ai",
          content: data.response || "No response received.",
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      if (err.name === "AbortError") return;
      setMessages(prev => [
        ...prev,
        { role: "ai", content: `Error: ${err.message}. Is the backend running?`, sources: [] },
      ]);
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [query, loading]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleChat();
    }
  };

  const clearConversation = async () => {
    try {
      await fetch(`${API}/clear-memory`, { method: "POST" });
      setMessages([]);
      abortRef.current?.abort();
    } catch (err) {
      console.error("Clear failed:", err);
    }
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="chat-wrap">
      <div className="chat-header">
        <div>
          <div className="page-title">AI Chat</div>
          <div className="page-subtitle">Ask questions about your uploaded documents</div>
        </div>
        {!isEmpty && (
          <button className="btn btn-danger" onClick={clearConversation}>
            <i className="ti ti-trash" aria-hidden="true" />
            Clear
          </button>
        )}
      </div>

      <div className="chat-messages">
        {isEmpty && (
          <div className="empty-state">
            <i className="ti ti-message-2" />
            <h3>Start a conversation</h3>
            <p>Upload documents first, then ask anything about their content.</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`msg-row${msg.role === "user" ? " user-row" : ""}`}>
            <div className={`msg-avatar ${msg.role === "user" ? "user-avatar" : "ai-avatar"}`}>
              {msg.role === "user"
                ? <i className="ti ti-user" style={{ fontSize: 13 }} aria-hidden="true" />
                : "NW"}
            </div>
            <div className={`msg-bubble ${msg.role === "user" ? "user-bubble" : "ai-bubble"}`}>
              {formatContent(msg.content)}
              {msg.sources && msg.sources.length > 0 && (
                <div className="msg-sources">
                  {msg.sources.map((s, j) => (
                    <span key={j} className="source-chip" title={`Similarity: ${s.similarity_score}`}>
                      <i className="ti ti-file-text" aria-hidden="true" />
                      {s.source} · {j + 1}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="msg-row">
            <div className="msg-avatar ai-avatar">NW</div>
            <div className="msg-thinking">
              Thinking
              <div className="thinking-dots">
                <span /><span /><span />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-area">
        <div className="chat-input-row">
          <input
            ref={inputRef}
            type="text"
            placeholder="Ask anything about your documents… (Enter to send)"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            aria-label="Chat input"
          />
          <button
            className="btn btn-primary"
            onClick={handleChat}
            disabled={loading || !query.trim()}
            aria-label="Send message"
          >
            <i className="ti ti-send" aria-hidden="true" />
            Send
          </button>
        </div>
      </div>
    </div>
  );
}