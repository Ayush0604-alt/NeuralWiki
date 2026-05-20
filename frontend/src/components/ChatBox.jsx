import { useState, useRef, useEffect, useCallback } from "react";
import * as d3 from "d3";

const API = "http://127.0.0.1:8000";

// ── Cluster colours (mirrors KnowledgeGraph.jsx) ──────────────────────────────
const CLUSTER_CONFIG = {
  CENTER:   { color: "#8b84ff", icon: "◉" },
  ENTITY:   { color: "#8b84ff", icon: "◉" },
  CONCEPT:  { color: "#1fc791", icon: "◈" },
  LOCATION: { color: "#e05252", icon: "◎" },
  EVENT:    { color: "#ff9f43", icon: "◆" },
  DATE:     { color: "#00d2d3", icon: "◇" },
  ACTION:   { color: "#c47aff", icon: "▶" },
  QUANTITY: { color: "#54a0ff", icon: "▣" },
  TECH:     { color: "#47bfff", icon: "⬡" },
  MISC:     { color: "#6b6b80", icon: "·" },
};
const cc = (cluster) => CLUSTER_CONFIG[cluster] || CLUSTER_CONFIG.MISC;

// ── Retrieval mode badge ──────────────────────────────────────────────────────
function RetrievalBadge({ mode, seedEntities }) {
  if (!mode || mode === "none") return null;
  const config = {
    hybrid:      { color: "#1fc791", bg: "#1fc79114", border: "#1fc79130", icon: "ti-git-merge",  label: "GraphRAG · Hybrid" },
    vector_only: { color: "#54a0ff", bg: "#54a0ff14", border: "#54a0ff30", icon: "ti-vector",     label: "Vector only" },
    graph_only:  { color: "#c47aff", bg: "#c47aff14", border: "#c47aff30", icon: "ti-share-2",    label: "Graph only" },
  };
  const c = config[mode] || config.vector_only;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap", marginTop: 8 }}>
      <span style={{ display: "inline-flex", alignItems: "center", gap: 5, fontSize: 10, padding: "2px 8px", background: c.bg, border: `1px solid ${c.border}`, borderRadius: 999, color: c.color, fontFamily: "var(--font-mono)" }}>
        <i className={`ti ${c.icon}`} style={{ fontSize: 11 }} />
        {c.label}
      </span>
      {seedEntities?.length > 0 && (
        <span style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
          entities: {seedEntities.slice(0, 4).join(", ")}{seedEntities.length > 4 ? ` +${seedEntities.length - 4}` : ""}
        </span>
      )}
    </div>
  );
}

// ── Mini D3 subgraph ──────────────────────────────────────────────────────────
function SubgraphViz({ subgraph }) {
  const svgRef = useRef(null);
  const simRef = useRef(null);
  const W = 320, H = 220;

  useEffect(() => {
    if (!svgRef.current || !subgraph?.nodes?.length) return;

    // Deep-clone so D3 mutation doesn't bleed between renders
    const nodes = subgraph.nodes.map(n => ({ ...n }));
    const links = subgraph.links.map(l => ({ ...l }));

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const defs = svg.append("defs");
    const marker = defs.append("marker")
      .attr("id", "sg-arrow").attr("viewBox", "0 0 10 10")
      .attr("refX", 22).attr("refY", 5)
      .attr("markerWidth", 5).attr("markerHeight", 5)
      .attr("orient", "auto-start-reverse");
    marker.append("path").attr("d", "M1 1L9 5L1 9Z").attr("fill", "#6c63ff55");

    const g = svg.append("g");
    const zoom = d3.zoom().scaleExtent([0.3, 4]).on("zoom", e => g.attr("transform", e.transform));
    svg.call(zoom);

    if (simRef.current) simRef.current.stop();
    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(55).strength(0.6))
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collide", d3.forceCollide(18))
      .alphaDecay(0.025);
    simRef.current = sim;

    const linkEl = g.append("g").selectAll("line").data(links).join("line")
      .attr("stroke", "#6c63ff55").attr("stroke-width", 1.2)
      .attr("marker-end", "url(#sg-arrow)");

    const linkLabel = g.append("g").selectAll("text").data(links).join("text")
      .text(d => (d.label || "").slice(0, 10))
      .attr("text-anchor", "middle").attr("font-size", 7)
      .attr("fill", "var(--text-muted)").attr("pointer-events", "none");

    const nodeEl = g.append("g").selectAll("g").data(nodes).join("g")
      .call(d3.drag()
        .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
        .on("end", (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    nodeEl.append("circle")
      .attr("r", d => d.isSeed ? 10 : 7)
      .attr("fill", d => cc(d.cluster).color + "22")
      .attr("stroke", d => cc(d.cluster).color)
      .attr("stroke-width", d => d.isSeed ? 2 : 1.2);

    nodeEl.append("text")
      .text(d => { const l = d.id; return l.length > 14 ? l.slice(0, 13) + "…" : l; })
      .attr("text-anchor", "middle").attr("y", d => (d.isSeed ? 10 : 7) + 9)
      .attr("font-size", d => d.isSeed ? 9 : 8)
      .attr("font-weight", d => d.isSeed ? "600" : "400")
      .attr("fill", d => cc(d.cluster).color)
      .attr("pointer-events", "none");

    sim.on("tick", () => {
      linkEl.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
      linkLabel
        .attr("x", d => ((d.source.x||0)+(d.target.x||0))/2)
        .attr("y", d => ((d.source.y||0)+(d.target.y||0))/2);
      nodeEl.attr("transform", d => `translate(${d.x||0},${d.y||0})`);
    });

    return () => simRef.current?.stop();
  }, [subgraph]);

  if (!subgraph?.nodes?.length) return null;
  return (
    <svg ref={svgRef} width={W} height={H}
      style={{ display: "block", background: "var(--bg)", borderRadius: 8 }} />
  );
}

// ── Reasoning subgraph panel ─────────────────────────────────────────────────
function SubgraphPanel({ subgraph, onClose }) {
  if (!subgraph?.nodes?.length) return null;
  return (
    <div style={{
      marginTop: 10,
      background: "var(--bg-surface)",
      border: "1px solid var(--accent-dim2)",
      borderRadius: "var(--radius-lg)",
      overflow: "hidden",
    }}>
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "8px 12px", borderBottom: "1px solid var(--border-subtle)",
        background: "var(--bg-raised)",
      }}>
        <span style={{ fontSize: 11, color: "var(--accent-light)", fontWeight: 500, display: "flex", alignItems: "center", gap: 5 }}>
          <i className="ti ti-share-2" style={{ fontSize: 12 }} />
          Reasoning subgraph
          <span style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)", marginLeft: 2 }}>
            {subgraph.nodes.length} nodes · {subgraph.links.length} edges
          </span>
        </span>
        <button onClick={onClose} style={{ background: "none", border: "none", color: "var(--text-muted)", cursor: "pointer", fontSize: 15, display: "flex", padding: 2 }}>
          <i className="ti ti-x" />
        </button>
      </div>
      <div style={{ padding: 8 }}>
        <SubgraphViz subgraph={subgraph} />
        <p style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 6, padding: "0 4px" }}>
          Highlighted nodes were traversed during knowledge-graph retrieval for this answer.
          Seed entities (larger circles) were matched directly from your query.
        </p>
      </div>
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
      while (i < lines.length && !lines[i].startsWith("```")) { codeLines.push(lines[i]); i++; }
      elements.push(
        <div key={`code-${i}`} style={{ margin: "10px 0" }}>
          {lang && (
            <div style={{ display: "flex", alignItems: "center", gap: 6, background: "var(--bg-active)", borderRadius: "6px 6px 0 0", padding: "4px 12px", borderBottom: "1px solid var(--border)" }}>
              <i className="ti ti-code" style={{ fontSize: 11, color: "var(--text-muted)" }} />
              <span style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>{lang}</span>
              <button title="Copy" onClick={() => navigator.clipboard?.writeText(codeLines.join("\n"))}
                style={{ marginLeft: "auto", background: "none", border: "none", color: "var(--text-muted)", cursor: "pointer", fontSize: 11, display: "flex", alignItems: "center", gap: 3 }}>
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
    if (line.startsWith("## "))  { elements.push(<p key={`h2-${i}`} style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)", margin: "14px 0 4px", borderBottom: "1px solid var(--border-subtle)", paddingBottom: 5 }}>{renderInline(line.slice(3))}</p>); i++; continue; }
    if (line.startsWith("# "))   { elements.push(<p key={`h1-${i}`} style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)", margin: "14px 0 6px" }}>{renderInline(line.slice(2))}</p>); i++; continue; }

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
        if (/^\s{2,}[-*•]\s/.test(lines[i])) listItems.push({ sub: true, content: lines[i].replace(/^\s+[-*•]\s/, "") });
        else listItems.push({ sub: false, content: lines[i].slice(2) });
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

    if (line.trim() === "") { elements.push(<div key={`sp-${i}`} style={{ height: 6 }} />); i++; continue; }
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

// ── Streaming cursor blink ────────────────────────────────────────────────────
function StreamingCursor() {
  return (
    <span style={{
      display: "inline-block", width: 2, height: "1.1em",
      background: "var(--accent-light)", marginLeft: 2,
      verticalAlign: "text-bottom",
      animation: "cursorBlink 0.7s step-end infinite",
    }} />
  );
}

// ── Main ChatBox ──────────────────────────────────────────────────────────────

export default function ChatBox({ onKnowledgeBaseCleared }) {
  const [query, setQuery]         = useState("");
  const [messages, setMessages]   = useState([]);
  const [loading, setLoading]     = useState(false);
  // Track which message subgraph panels are open: Set of message indices
  const [openSubgraphs, setOpenSubgraphs] = useState(new Set());

  const bottomRef  = useRef(null);
  const inputRef   = useRef(null);
  const abortRef   = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleChat = useCallback(async () => {
    const q = query.trim();
    if (!q || loading) return;

    const userMsgIndex = messages.length;
    setMessages(prev => [...prev, { role: "user", content: q }]);
    setQuery("");
    setLoading(true);

    abortRef.current?.abort();
    abortRef.current = new AbortController();

    // Add a placeholder AI message we'll stream tokens into
    const aiMsgIndex = userMsgIndex + 1;
    setMessages(prev => [...prev, {
      role: "ai",
      content: "",
      streaming: true,
      sources: [],
      retrieval_mode: null,
      seed_entities: [],
      subgraph: null,
    }]);

    try {
      const res = await fetch(`${API}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
        signal: abortRef.current.signal,
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      if (!res.body) throw new Error("No response body — streaming not supported");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by double newlines
        const frames = buffer.split("\n\n");
        buffer = frames.pop(); // last incomplete frame stays in buffer

        for (const frame of frames) {
          if (!frame.startsWith("data: ")) continue;
          try {
            const payload = JSON.parse(frame.slice(6));

            if (payload.type === "token") {
              setMessages(prev => {
                const updated = [...prev];
                updated[aiMsgIndex] = {
                  ...updated[aiMsgIndex],
                  content: updated[aiMsgIndex].content + payload.content,
                };
                return updated;
              });
            } else if (payload.type === "meta") {
              setMessages(prev => {
                const updated = [...prev];
                updated[aiMsgIndex] = {
                  ...updated[aiMsgIndex],
                  streaming: false,
                  sources: payload.sources || [],
                  retrieval_mode: payload.retrieval_mode,
                  seed_entities: payload.seed_entities || [],
                  subgraph: payload.subgraph || null,
                };
                return updated;
              });
            } else if (payload.type === "done") {
              setMessages(prev => {
                const updated = [...prev];
                updated[aiMsgIndex] = { ...updated[aiMsgIndex], streaming: false };
                return updated;
              });
            } else if (payload.type === "error") {
              setMessages(prev => {
                const updated = [...prev];
                updated[aiMsgIndex] = {
                  ...updated[aiMsgIndex],
                  streaming: false,
                  content: updated[aiMsgIndex].content || `Error: ${payload.content}`,
                };
                return updated;
              });
            }
          } catch {
            // malformed JSON frame — ignore
          }
        }
      }
    } catch (err) {
      if (err.name === "AbortError") return;
      setMessages(prev => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last?.role === "ai") {
          updated[updated.length - 1] = {
            ...last,
            streaming: false,
            content: last.content || `Error: ${err.message}. Is the backend running?`,
          };
        }
        return updated;
      });
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [query, loading, messages.length]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleChat(); }
  };

  const clearConversation = async () => {
    try {
      await fetch(`${API}/clear-memory`, { method: "POST" });
      abortRef.current?.abort();
      setMessages([]);
      setOpenSubgraphs(new Set());
    } catch (err) {
      console.error("Clear failed:", err);
    }
  };

  // Called by parent (Upload) when clear-documents fires, so chat stays in sync
  // This is exported via a ref pattern — see App.jsx note at bottom
  useEffect(() => {
    if (onKnowledgeBaseCleared) {
      const unsub = onKnowledgeBaseCleared(() => {
        setMessages([]);
        setOpenSubgraphs(new Set());
      });
      return unsub;
    }
  }, [onKnowledgeBaseCleared]);

  const toggleSubgraph = (idx) => {
    setOpenSubgraphs(prev => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="chat-wrap">
      <style>{`
        @keyframes cursorBlink { 0%,100%{opacity:1} 50%{opacity:0} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
      `}</style>

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
              {msg.role === "user" ? <i className="ti ti-user" style={{ fontSize: 13 }} /> : "NW"}
            </div>

            <div className={`msg-bubble ${msg.role === "user" ? "user-bubble" : "ai-bubble"}`}
              style={{ maxWidth: msg.role === "ai" && openSubgraphs.has(i) ? "82%" : "72%" }}>
              {msg.role === "user"
                ? <span style={{ fontSize: 14, lineHeight: 1.65 }}>{msg.content}</span>
                : <>
                    <RichContent text={msg.content} />
                    {msg.streaming && <StreamingCursor />}
                  </>
              }

              {/* Retrieval badge + subgraph toggle */}
              {msg.role === "ai" && !msg.streaming && msg.retrieval_mode && (
                <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                  <RetrievalBadge mode={msg.retrieval_mode} seedEntities={msg.seed_entities} />
                  {msg.subgraph?.nodes?.length > 0 && (
                    <button
                      onClick={() => toggleSubgraph(i)}
                      style={{
                        display: "inline-flex", alignItems: "center", gap: 4,
                        fontSize: 10, padding: "2px 8px",
                        background: openSubgraphs.has(i) ? "var(--accent-dim2)" : "var(--bg-raised)",
                        border: `1px solid ${openSubgraphs.has(i) ? "var(--accent-dim2)" : "var(--border)"}`,
                        borderRadius: 999, color: openSubgraphs.has(i) ? "var(--accent-light)" : "var(--text-muted)",
                        cursor: "pointer", fontFamily: "var(--font-mono)",
                        transition: "all 150ms",
                      }}
                    >
                      <i className="ti ti-share-2" style={{ fontSize: 11 }} />
                      {openSubgraphs.has(i) ? "Hide" : "Show"} reasoning subgraph
                    </button>
                  )}
                </div>
              )}

              {/* Inline subgraph panel */}
              {msg.role === "ai" && openSubgraphs.has(i) && (
                <SubgraphPanel
                  subgraph={msg.subgraph}
                  onClose={() => toggleSubgraph(i)}
                />
              )}

              {/* Sources */}
              {msg.sources?.length > 0 && (
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

        {/* Only show old-style "Thinking" spinner if somehow streaming fails to start */}
        {loading && messages[messages.length - 1]?.role !== "ai" && (
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