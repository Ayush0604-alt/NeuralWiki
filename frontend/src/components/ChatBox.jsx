import { useState, useRef, useEffect, useCallback } from "react";

const API = "http://127.0.0.1:8000";

// ── Retrieval mode badge ──────────────────────────────────────────────────────
function RetrievalBadge({ mode, seedEntities }) {
  if (!mode || mode === "none") return null;

  const config = {
    hybrid: {
      color: "#1fc791",
      bg: "#1fc79114",
      border: "#1fc79130",
      icon: "ti-git-merge",
      label: "GraphRAG · Hybrid",
    },
    vector_only: {
      color: "#54a0ff",
      bg: "#54a0ff14",
      border: "#54a0ff30",
      icon: "ti-vector",
      label: "Vector only",
    },
    graph_only: {
      color: "#c47aff",
      bg: "#c47aff14",
      border: "#c47aff30",
      icon: "ti-share-2",
      label: "Graph only",
    },
  };

  const c = config[mode] || config.vector_only;

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 6,
      flexWrap: "wrap", marginTop: 8,
    }}>
      <span style={{
        display: "inline-flex", alignItems: "center", gap: 5,
        fontSize: 10, padding: "2px 8px",
        background: c.bg, border: `1px solid ${c.border}`,
        borderRadius: 999, color: c.color,
        fontFamily: "var(--font-mono)",
      }}>
        <i className={`ti ${c.icon}`} style={{ fontSize: 11 }} />
        {c.label}
      </span>
      {seedEntities && seedEntities.length > 0 && (
        <span style={{
          fontSize: 10, color: "var(--text-muted)",
          fontFamily: "var(--font-mono)",
        }}>
          entities: {seedEntities.slice(0, 4).join(", ")}
          {seedEntities.length > 4 ? ` +${seedEntities.length - 4}` : ""}
        </span>
      )}
    </div>
  );
}

// ── Rich markdown renderer ────────────────────────────────────────────────────

function RichContent({ text }) {
  const lines = text.split("\n");
  const elements = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block
    if (line.startsWith("```")) {
      const lang = line.slice(3).trim();
      const codeLines = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      elements.push(
        <div key={`code-${i}`} style={{ margin: "10px 0" }}>
          {lang && (
            <div style={{
              display: "flex", alignItems: "center", gap: 6,
              background: "var(--bg-active)", borderRadius: "6px 6px 0 0",
              padding: "4px 12px", borderBottom: "1px solid var(--border)",
            }}>
              <i className="ti ti-code" style={{ fontSize: 11, color: "var(--text-muted)" }} />
              <span style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>{lang}</span>
              <button
                title="Copy code"
                onClick={() => navigator.clipboard?.writeText(codeLines.join("\n"))}
                style={{ marginLeft: "auto", background: "none", border: "none", color: "var(--text-muted)", cursor: "pointer", padding: "2px 4px", fontSize: 11, display: "flex", alignItems: "center", gap: 3 }}
              >
                <i className="ti ti-copy" style={{ fontSize: 11 }} />copy
              </button>
            </div>
          )}
          <pre style={{ background: "var(--bg)", border: "1px solid var(--border)", borderTop: lang ? "none" : undefined, borderRadius: lang ? "0 0 6px 6px" : "6px", padding: "12px 14px", margin: 0, overflowX: "auto", fontSize: 12, fontFamily: "var(--font-mono)", lineHeight: 1.65, color: "var(--text-primary)", whiteSpace: "pre" }}>
            {codeLines.join("\n")}
          </pre>
        </div>
      );
      i++; continue;
    }

    // Blockquote
    if (line.startsWith("> ")) {
      const quoteLines = [];
      while (i < lines.length && lines[i].startsWith("> ")) { quoteLines.push(lines[i].slice(2)); i++; }
      elements.push(
        <div key={`bq-${i}`} style={{ borderLeft: "3px solid var(--accent-dim2)", paddingLeft: 12, margin: "8px 0", color: "var(--text-secondary)", fontSize: 13, fontStyle: "italic" }}>
          {quoteLines.map((l, j) => <div key={j}>{renderInline(l)}</div>)}
        </div>
      );
      continue;
    }

    // Table
    if (line.includes("|") && lines[i + 1]?.match(/^\|[-| :]+\|$/)) {
      const tableLines = [];
      while (i < lines.length && lines[i].includes("|")) { tableLines.push(lines[i]); i++; }
      const rows = tableLines
        .filter(l => !l.match(/^\|[-| :]+\|$/))
        .map(l => l.split("|").filter((_, idx, arr) => idx > 0 && idx < arr.length - 1).map(c => c.trim()));
      if (rows.length >= 2) {
        elements.push(
          <div key={`table-${i}`} style={{ overflowX: "auto", margin: "10px 0" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr>{rows[0].map((cell, j) => <th key={j} style={{ padding: "7px 12px", background: "var(--bg-raised)", border: "1px solid var(--border)", color: "var(--text-primary)", fontWeight: 600, textAlign: "left", fontSize: 12 }}>{renderInline(cell)}</th>)}</tr>
              </thead>
              <tbody>
                {rows.slice(1).map((row, ri) => (
                  <tr key={ri} style={{ background: ri % 2 === 0 ? "transparent" : "var(--bg-surface)" }}>
                    {row.map((cell, ci) => <td key={ci} style={{ padding: "7px 12px", border: "1px solid var(--border)", color: "var(--text-secondary)", fontSize: 13 }}>{renderInline(cell)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      }
      continue;
    }

    // Headings
    if (line.startsWith("### ")) { elements.push(<p key={`h3-${i}`} style={{ fontSize: 13, fontWeight: 600, color: "var(--accent-light)", margin: "12px 0 4px" }}>{renderInline(line.slice(4))}</p>); i++; continue; }
    if (line.startsWith("## ")) { elements.push(<p key={`h2-${i}`} style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)", margin: "14px 0 4px", borderBottom: "1px solid var(--border-subtle)", paddingBottom: 5 }}>{renderInline(line.slice(3))}</p>); i++; continue; }
    if (line.startsWith("# ")) { elements.push(<p key={`h1-${i}`} style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)", margin: "14px 0 6px" }}>{renderInline(line.slice(2))}</p>); i++; continue; }

    // HR
    if (line.match(/^---+$/) || line.match(/^\*\*\*+$/)) { elements.push(<hr key={`hr-${i}`} style={{ border: "none", borderTop: "1px solid var(--border)", margin: "12px 0" }} />); i++; continue; }

    // Numbered list
    if (/^\d+\.\s/.test(line)) {
      const listItems = [];
      while (i < lines.length && /^\d+\.\s/.test(lines[i])) {
        const match = lines[i].match(/^(\d+)\.\s(.*)/);
        listItems.push({ num: match[1], content: match[2] });
        i++;
      }
      elements.push(
        <div key={`ol-${i}`} style={{ margin: "8px 0", display: "flex", flexDirection: "column", gap: 5 }}>
          {listItems.map(({ num, content }, j) => (
            <div key={j} style={{ display: "flex", gap: 10, alignItems: "flex-start", fontSize: 13 }}>
              <span style={{ minWidth: 20, height: 20, borderRadius: "50%", background: "var(--accent-dim2)", color: "var(--accent-light)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 600, flexShrink: 0, marginTop: 1 }}>{num}</span>
              <span style={{ color: "var(--text-secondary)", lineHeight: 1.6 }}>{renderInline(content)}</span>
            </div>
          ))}
        </div>
      );
      continue;
    }

    // Bullet list
    if (/^[-*•]\s/.test(line)) {
      const listItems = [];
      while (i < lines.length && /^[-*•]\s/.test(lines[i])) {
        if (/^\s{2,}[-*•]\s/.test(lines[i])) {
          listItems.push({ sub: true, content: lines[i].replace(/^\s+[-*•]\s/, "") });
        } else {
          listItems.push({ sub: false, content: lines[i].slice(2) });
        }
        i++;
      }
      elements.push(
        <div key={`ul-${i}`} style={{ margin: "6px 0", display: "flex", flexDirection: "column", gap: 4 }}>
          {listItems.map((item, j) => (
            <div key={j} style={{ display: "flex", gap: 8, alignItems: "flex-start", fontSize: 13, paddingLeft: item.sub ? 18 : 0 }}>
              <span style={{ color: item.sub ? "var(--text-muted)" : "var(--accent-light)", marginTop: 5, fontSize: item.sub ? 6 : 8, flexShrink: 0 }}>●</span>
              <span style={{ color: "var(--text-secondary)", lineHeight: 1.6 }}>{renderInline(item.content)}</span>
            </div>
          ))}
        </div>
      );
      continue;
    }

    // Empty line
    if (line.trim() === "") { elements.push(<div key={`sp-${i}`} style={{ height: 6 }} />); i++; continue; }

    // Regular paragraph
    elements.push(<p key={`p-${i}`} style={{ fontSize: 14, color: "var(--text-secondary)", lineHeight: 1.7, margin: "3px 0" }}>{renderInline(line)}</p>);
    i++;
  }

  return <div style={{ display: "flex", flexDirection: "column" }}>{elements}</div>;
}

function renderInline(text) {
  if (!text) return null;
  const parts = [];
  const regex = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`|\[(.+?)\]\((.+?)\))/g;
  let last = 0, match;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    if (match[2]) parts.push(<strong key={match.index} style={{ color: "var(--text-primary)", fontWeight: 600 }}>{match[2]}</strong>);
    else if (match[3]) parts.push(<em key={match.index} style={{ fontStyle: "italic", color: "var(--text-secondary)" }}>{match[3]}</em>);
    else if (match[4]) parts.push(<code key={match.index} style={{ fontFamily: "var(--font-mono)", fontSize: "0.9em", background: "var(--bg-raised)", border: "1px solid var(--border)", borderRadius: 4, padding: "1px 5px", color: "var(--accent-light)" }}>{match[4]}</code>);
    else if (match[5] && match[6]) parts.push(<a key={match.index} href={match[6]} target="_blank" rel="noopener noreferrer" style={{ color: "var(--accent-light)", textDecoration: "underline" }}>{match[5]}</a>);
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts.length === 1 && typeof parts[0] === "string" ? parts[0] : parts;
}

// ── Main ChatBox ──────────────────────────────────────────────────────────────

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

    setMessages(prev => [...prev, { role: "user", content: q }]);
    setQuery("");
    setLoading(true);

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
          retrieval_mode: data.retrieval_mode,
          seed_entities: data.seed_entities || [],
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
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleChat(); }
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
          <div className="page-subtitle">GraphRAG — vector search + knowledge graph traversal</div>
        </div>
        {!isEmpty && (
          <button className="btn btn-danger" onClick={clearConversation}>
            <i className="ti ti-trash" /> Clear
          </button>
        )}
      </div>

      <div className="chat-messages">
        {isEmpty && (
          <div className="empty-state">
            <i className="ti ti-message-2" />
            <h3>Start a conversation</h3>
            <p>Upload documents first, then ask anything. GraphRAG combines vector search with multi-hop graph reasoning.</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`msg-row${msg.role === "user" ? " user-row" : ""}`}>
            <div className={`msg-avatar ${msg.role === "user" ? "user-avatar" : "ai-avatar"}`}>
              {msg.role === "user"
                ? <i className="ti ti-user" style={{ fontSize: 13 }} />
                : "NW"}
            </div>

            <div className={`msg-bubble ${msg.role === "user" ? "user-bubble" : "ai-bubble"}`}>
              {msg.role === "user"
                ? <span style={{ fontSize: 14, lineHeight: 1.65 }}>{msg.content}</span>
                : <RichContent text={msg.content} />
              }

              {/* GraphRAG retrieval mode badge */}
              {msg.role === "ai" && msg.retrieval_mode && (
                <RetrievalBadge
                  mode={msg.retrieval_mode}
                  seedEntities={msg.seed_entities}
                />
              )}

              {/* Sources */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="msg-sources">
                  <span style={{ fontSize: 10, color: "var(--text-muted)", marginRight: 4 }}>Sources:</span>
                  {msg.sources.map((s, j) => (
                    <span key={j} className="source-chip" title={`Similarity: ${s.similarity_score}`}>
                      <i className="ti ti-file-text" />
                      {s.source} · chunk {s.chunk_id}
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
              <div className="thinking-dots"><span /><span /><span /></div>
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
            placeholder="Ask anything… (Enter to send)"
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
            <i className="ti ti-send" /> Send
          </button>
        </div>
      </div>
    </div>
  );
}