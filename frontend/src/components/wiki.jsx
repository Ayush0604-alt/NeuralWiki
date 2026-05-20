import { useState, useEffect, useCallback } from "react";

const API = "http://127.0.0.1:8000";

// ── Markdown renderer ─────────────────────────────────────────────────────────
function WikiContent({ text }) {
  if (!text) return null;
  const lines = text.split("\n");
  const elements = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.includes("|") && lines[i + 1]?.match(/^\|[-| :]+\|$/)) {
      const tableLines = [];
      while (i < lines.length && lines[i].includes("|")) { tableLines.push(lines[i]); i++; }
      const rows = tableLines
        .filter(l => !l.match(/^\|[-| :]+\|$/))
        .map(l => l.split("|").filter((_, idx, arr) => idx > 0 && idx < arr.length - 1).map(c => c.trim()));
      if (rows.length >= 2) {
        elements.push(
          <div key={`t-${i}`} style={{ overflowX: "auto", margin: "12px 0" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead><tr>{rows[0].map((cell, j) => <th key={j} style={{ padding: "8px 14px", background: "var(--bg-raised)", border: "1px solid var(--border)", color: "var(--text-primary)", fontWeight: 600, textAlign: "left", fontSize: 12 }}>{inline(cell)}</th>)}</tr></thead>
              <tbody>{rows.slice(1).map((row, ri) => <tr key={ri} style={{ background: ri%2===0?"transparent":"var(--bg-surface)" }}>{row.map((cell, ci) => <td key={ci} style={{ padding: "8px 14px", border: "1px solid var(--border)", color: "var(--text-secondary)", fontSize: 13 }}>{inline(cell)}</td>)}</tr>)}</tbody>
            </table>
          </div>
        );
      }
      continue;
    }

    if (line.startsWith("## ")) {
      elements.push(<div key={`h2-${i}`} style={{ fontSize: 15, fontWeight: 700, color: "var(--text-primary)", margin: "22px 0 8px", paddingBottom: 6, borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", gap: 8 }}><span style={{ color: "var(--accent-light)", fontSize: 13 }}>§</span>{inline(line.slice(3))}</div>);
      i++; continue;
    }
    if (line.startsWith("### ")) {
      elements.push(<div key={`h3-${i}`} style={{ fontSize: 13, fontWeight: 600, color: "var(--accent-light)", margin: "14px 0 5px" }}>{inline(line.slice(4))}</div>);
      i++; continue;
    }

    if (/^\d+\.\s/.test(line)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s/.test(lines[i])) {
        const m = lines[i].match(/^(\d+)\.\s(.*)/);
        items.push({ num: m[1], content: m[2] }); i++;
      }
      elements.push(<div key={`ol-${i}`} style={{ margin: "8px 0", display: "flex", flexDirection: "column", gap: 6 }}>{items.map(({ num, content }, j) => <div key={j} style={{ display: "flex", gap: 10, alignItems: "flex-start", fontSize: 13 }}><span style={{ minWidth: 22, height: 22, borderRadius: "50%", background: "var(--accent-dim2)", color: "var(--accent-light)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 700, flexShrink: 0, marginTop: 1 }}>{num}</span><span style={{ color: "var(--text-secondary)", lineHeight: 1.65 }}>{inline(content)}</span></div>)}</div>);
      continue;
    }

    if (/^[-*•]\s/.test(line)) {
      const items = [];
      while (i < lines.length && /^[-*•]\s/.test(lines[i])) { items.push(lines[i].slice(2)); i++; }
      elements.push(<div key={`ul-${i}`} style={{ margin: "6px 0", display: "flex", flexDirection: "column", gap: 5 }}>{items.map((item, j) => <div key={j} style={{ display: "flex", gap: 8, alignItems: "flex-start", fontSize: 13 }}><span style={{ color: "var(--accent-light)", marginTop: 5, fontSize: 7, flexShrink: 0 }}>●</span><span style={{ color: "var(--text-secondary)", lineHeight: 1.65 }}>{inline(item)}</span></div>)}</div>);
      continue;
    }

    if (line.trim() === "") { elements.push(<div key={`sp-${i}`} style={{ height: 5 }} />); i++; continue; }
    elements.push(<p key={`p-${i}`} style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.75, margin: "3px 0" }}>{inline(line)}</p>);
    i++;
  }
  return <div>{elements}</div>;
}

function inline(text) {
  if (!text) return null;
  const parts = [];
  const rx = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`|\[(.+?)\]\((.+?)\))/g;
  let last = 0, m;
  while ((m = rx.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    if (m[2]) parts.push(<strong key={m.index} style={{ color: "var(--text-primary)", fontWeight: 600 }}>{m[2]}</strong>);
    else if (m[3]) parts.push(<em key={m.index} style={{ fontStyle: "italic" }}>{m[3]}</em>);
    else if (m[4]) parts.push(<code key={m.index} style={{ fontFamily: "var(--font-mono)", fontSize: "0.88em", background: "var(--bg-raised)", border: "1px solid var(--border)", borderRadius: 3, padding: "1px 5px", color: "var(--accent-light)" }}>{m[4]}</code>);
    else if (m[5] && m[6]) parts.push(<a key={m.index} href={m[6]} target="_blank" rel="noreferrer" style={{ color: "var(--accent-light)", textDecoration: "underline" }}>{m[5]}</a>);
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts.length === 1 && typeof parts[0] === "string" ? parts[0] : parts;
}

// ── Wiki Page View ────────────────────────────────────────────────────────────
function WikiPage({ page, onBack, onRegenerate, regenerating }) {
  const date = page.generated_at ? new Date(page.generated_at).toLocaleString() : "";
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", padding: "18px 28px 14px", borderBottom: "1px solid var(--border-subtle)", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <button className="btn" onClick={onBack} style={{ padding: "6px 10px" }}><i className="ti ti-arrow-left" /> Back</button>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, color: "var(--text-primary)" }}>{page.title}</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2, display: "flex", gap: 10 }}>
              <span><i className="ti ti-file-text" style={{ marginRight: 4 }} />{page.filename}</span>
              <span><i className="ti ti-clock" style={{ marginRight: 4 }} />{date}</span>
              <span><i className="ti ti-align-left" style={{ marginRight: 4 }} />{(page.word_count||0).toLocaleString()} words</span>
            </div>
          </div>
        </div>
        <button className="btn" onClick={() => onRegenerate(page.filename)} disabled={regenerating === page.filename}>
          <i className={`ti ${regenerating === page.filename ? "ti-loader-2" : "ti-refresh"}`} style={regenerating === page.filename ? { animation: "spin 1s linear infinite" } : {}} />
          {regenerating === page.filename ? "Regenerating…" : "Regenerate"}
        </button>
      </div>
      <div style={{ flex: 1, overflowY: "auto", padding: "20px 28px 32px" }}>
        <WikiContent text={page.content} />
      </div>
    </div>
  );
}

// ── Wiki List View ────────────────────────────────────────────────────────────
function WikiList({ pages, loading, onSelect, onRegenerate, regenerating, uploadedDocs, onGenerateAll }) {
  const pagesMap = new Set(pages.map(p => p.filename));
  const missing = uploadedDocs.filter(d => !pagesMap.has(d.filename));
  const fileIcon = n => { const e = n?.split(".").pop()?.toLowerCase(); return e==="pdf"?"ti-file-type-pdf":e==="md"?"ti-file-description":"ti-file-text"; };

  if (loading && !pages.length) {
    return <div className="empty-state"><i className="ti ti-loader-2" style={{ animation: "spin 1s linear infinite" }} /><h3>Loading…</h3></div>;
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <div style={{ padding: "20px 28px 16px", borderBottom: "1px solid var(--border-subtle)", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
          <div>
            <div className="page-title">Wiki</div>
            <div className="page-subtitle">AI-generated knowledge pages · persisted across restarts</div>
          </div>
          {missing.length > 0 && (
            <button className="btn btn-primary" onClick={onGenerateAll} disabled={!!regenerating}>
              <i className="ti ti-wand" />Generate {missing.length} missing page{missing.length !== 1 ? "s" : ""}
            </button>
          )}
        </div>
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: "16px 28px", display: "flex", flexDirection: "column", gap: 12 }}>
        {pages.map(p => (
          <div key={p.filename} onClick={() => onSelect(p.filename)}
            style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)", padding: "16px 20px", cursor: "pointer", transition: "all 150ms", display: "flex", alignItems: "flex-start", gap: 14 }}
            onMouseEnter={e => { e.currentTarget.style.borderColor="var(--accent-dim2)"; e.currentTarget.style.background="var(--bg-raised)"; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor="var(--border)"; e.currentTarget.style.background="var(--bg-surface)"; }}>
            <div style={{ width: 40, height: 40, borderRadius: 10, flexShrink: 0, background: "var(--accent-dim)", border: "1px solid var(--accent-dim2)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, color: "var(--accent-light)" }}>
              <i className={`ti ${fileIcon(p.filename)}`} />
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)", marginBottom: 3 }}>{p.title}</div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6, fontFamily: "var(--font-mono)" }}>{p.filename}</div>
              {p.preview && <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.6, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{p.preview}</div>}
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 6, flexShrink: 0 }}>
              <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 999, background: "var(--teal-dim)", color: "var(--teal)", fontFamily: "var(--font-mono)" }}><i className="ti ti-check" style={{ marginRight: 3 }} />wiki ready</span>
              <button className="btn" style={{ padding: "3px 10px", fontSize: 11 }} onClick={e => { e.stopPropagation(); onRegenerate(p.filename); }} disabled={regenerating === p.filename}>
                <i className={`ti ${regenerating===p.filename?"ti-loader-2":"ti-refresh"}`} style={regenerating===p.filename?{animation:"spin 1s linear infinite"}:{}} />
                {regenerating === p.filename ? "…" : "Regen"}
              </button>
            </div>
          </div>
        ))}

        {missing.map(d => (
          <div key={d.filename} style={{ background: "var(--bg-surface)", border: "1px dashed var(--border)", borderRadius: "var(--radius-lg)", padding: "16px 20px", display: "flex", alignItems: "center", gap: 14, opacity: 0.65 }}>
            <div style={{ width: 40, height: 40, borderRadius: 10, flexShrink: 0, background: "var(--bg-raised)", border: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, color: "var(--text-muted)" }}><i className={`ti ${fileIcon(d.filename)}`} /></div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 2 }}>{d.filename.replace(/\.[^.]+$/,"").replace(/[_-]/g," ")}</div>
              <div style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>{d.filename}</div>
            </div>
            <button className="btn btn-primary" onClick={() => onRegenerate(d.filename)} disabled={regenerating===d.filename} style={{ flexShrink: 0 }}>
              <i className={`ti ${regenerating===d.filename?"ti-loader-2":"ti-wand"}`} style={regenerating===d.filename?{animation:"spin 1s linear infinite"}:{}} />
              {regenerating===d.filename?"Generating…":"Generate wiki"}
            </button>
          </div>
        ))}

        {pages.length === 0 && missing.length === 0 && !loading && (
          <div className="empty-state"><i className="ti ti-book" /><h3>No wiki pages yet</h3><p>Upload documents to automatically generate wiki pages.</p></div>
        )}
      </div>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function Wiki({ onDocumentAdded, onDocumentRemoved }) {
  const [pages, setPages]               = useState([]);
  const [selectedPage, setSelectedPage] = useState(null);
  const [loading, setLoading]           = useState(false);
  const [regenerating, setRegenerating] = useState(null);
  const [uploadedDocs, setUploadedDocs] = useState([]);
  const [error, setError]               = useState("");

  const fetchPages = useCallback(async () => {
    setLoading(true);
    try {
      const [wikiRes, docsRes] = await Promise.all([
        fetch(`${API}/wiki/pages`),
        fetch(`${API}/documents`),
      ]);
      if (wikiRes.ok) { const d = await wikiRes.json(); setPages(d.pages || []); }
      if (docsRes.ok) { const d = await docsRes.json(); setUploadedDocs(d.documents || []); }
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { fetchPages(); }, [fetchPages]);

  // Auto-refresh when documents are added/removed
  useEffect(() => {
    const unsubs = [];
    if (onDocumentAdded)   unsubs.push(onDocumentAdded(fetchPages));
    if (onDocumentRemoved) unsubs.push(onDocumentRemoved(fetchPages));
    return () => unsubs.forEach(u => u?.());
  }, [onDocumentAdded, onDocumentRemoved, fetchPages]);

  const openPage = async (filename) => {
    try {
      const res = await fetch(`${API}/wiki/page/${encodeURIComponent(filename)}`);
      if (!res.ok) throw new Error("Page not found");
      const data = await res.json();
      setSelectedPage(data.page);
    } catch (e) { setError(e.message); }
  };

  const regenerate = async (filename) => {
    setRegenerating(filename); setError("");
    try {
      const res = await fetch(`${API}/wiki/generate`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename }),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.detail || "Generation failed");
      await fetchPages();
      if (selectedPage?.filename === filename) setSelectedPage(data.page);
    } catch (e) { setError(e.message); }
    finally { setRegenerating(null); }
  };

  const generateAll = async () => {
    const pagesMap = new Set(pages.map(p => p.filename));
    const missing = uploadedDocs.filter(d => !pagesMap.has(d.filename));
    for (const doc of missing) await regenerate(doc.filename);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      {error && (
        <div style={{ margin: "0 28px", marginTop: 12, padding: "10px 14px", background: "var(--red-dim)", border: "1px solid var(--red)", borderRadius: "var(--radius-md)", color: "var(--red)", fontSize: 12, display: "flex", gap: 8, alignItems: "center", flexShrink: 0 }}>
          <i className="ti ti-alert-circle" />{error}
          <button onClick={() => setError("")} style={{ marginLeft: "auto", background: "none", border: "none", color: "var(--red)", cursor: "pointer", fontSize: 14 }}>×</button>
        </div>
      )}
      {selectedPage ? (
        <WikiPage page={selectedPage} onBack={() => setSelectedPage(null)} onRegenerate={regenerate} regenerating={regenerating} />
      ) : (
        <WikiList pages={pages} loading={loading} onSelect={openPage} onRegenerate={regenerate} regenerating={regenerating} uploadedDocs={uploadedDocs} onGenerateAll={generateAll} />
      )}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}