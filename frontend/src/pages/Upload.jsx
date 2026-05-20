import { useState, useRef, useCallback, useEffect } from "react";

const API = "http://127.0.0.1:8000";
const ALLOWED = [".pdf", ".txt", ".md"];

function getExt(name) { return name?.split(".").pop()?.toLowerCase() || ""; }
function fileIcon(name) {
  const ext = getExt(name);
  if (ext === "pdf") return "ti-file-type-pdf";
  if (ext === "md")  return "ti-file-description";
  return "ti-file-text";
}
function formatBytes(bytes) {
  if (bytes < 1024)         return `${bytes} B`;
  if (bytes < 1024 * 1024)  return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// ── Confirmation Modal ────────────────────────────────────────────────────────
function ConfirmModal({ title, body, confirmLabel = "Confirm", danger = false, onConfirm, onCancel }) {
  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 200, background: "rgba(0,0,0,0.65)", backdropFilter: "blur(4px)", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)", padding: "28px 28px 22px", width: 380, boxShadow: "0 24px 60px rgba(0,0,0,0.6)", animation: "fadeUp 150ms ease" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
          <i className={`ti ${danger ? "ti-alert-triangle" : "ti-help-circle"}`} style={{ fontSize: 20, color: danger ? "var(--red)" : "var(--amber)" }} />
          <span style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)" }}>{title}</span>
        </div>
        <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.65, marginBottom: 22 }}>{body}</p>
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <button className="btn" onClick={onCancel}>Cancel</button>
          <button className={`btn ${danger ? "btn-danger" : "btn-primary"}`} onClick={onConfirm}>{confirmLabel}</button>
        </div>
      </div>
    </div>
  );
}

// ── Document Row ──────────────────────────────────────────────────────────────
function DocRow({ doc, onDelete, deleting }) {
  return (
    <div
      style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 16px", borderBottom: "1px solid var(--border-subtle)", transition: "background 150ms ease" }}
      onMouseEnter={e => e.currentTarget.style.background = "var(--bg-hover)"}
      onMouseLeave={e => e.currentTarget.style.background = "transparent"}
    >
      <i className={`ti ${fileIcon(doc.filename)}`} style={{ fontSize: 16, color: "var(--accent-light)", flexShrink: 0 }} />
      <span style={{ flex: 1, fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--text-primary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {doc.filename}
      </span>
      <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 999, background: "var(--accent-dim)", color: "var(--accent-light)", fontFamily: "var(--font-mono)", flexShrink: 0 }}>
        {doc.chunks} chunks
      </span>
      <button
        className="btn btn-danger" style={{ padding: "4px 10px", fontSize: 11, gap: 4 }}
        onClick={() => onDelete(doc.filename)} disabled={deleting === doc.filename}
        title={`Remove ${doc.filename} from knowledge base`}
      >
        {deleting === doc.filename
          ? <i className="ti ti-loader-2" style={{ animation: "spin 1s linear infinite" }} />
          : <i className="ti ti-trash" />}
        {deleting === doc.filename ? "Removing…" : "Remove"}
      </button>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function Upload({ onKnowledgeBaseCleared, onDocumentAdded, onDocumentRemoved }) {
  const [dragOver, setDragOver]           = useState(false);
  const [uploading, setUploading]         = useState(false);
  const [progress, setProgress]           = useState(0);
  const [currentFile, setCurrentFile]     = useState(null);
  const [stats, setStats]                 = useState(null);
  const [log, setLog]                     = useState([]);
  const [docs, setDocs]                   = useState([]);
  const [docsLoading, setDocsLoading]     = useState(false);
  const [deletingFile, setDeletingFile]   = useState(null);
  const [confirmClearAll, setConfirmClearAll] = useState(false);
  const [confirmDelete, setConfirmDelete]     = useState(null);

  const fileRef = useRef(null);

  const refreshDocs = useCallback(async () => {
    setDocsLoading(true);
    try {
      const res  = await fetch(`${API}/documents`);
      const data = await res.json();
      setDocs(data.documents || []);
    } catch { /* backend may be starting */ }
    finally { setDocsLoading(false); }
  }, []);

  useEffect(() => { refreshDocs(); }, [refreshDocs]);

  const addLog = (type, msg) =>
    setLog(prev => [...prev, { type, msg, time: new Date().toLocaleTimeString() }]);

  // ── Upload ─────────────────────────────────────────────────────────────────
  const uploadFile = useCallback(async (file) => {
    if (!file) return;
    const ext = `.${getExt(file.name)}`;
    if (!ALLOWED.includes(ext)) {
      addLog("error", `Unsupported type: ${ext}. Use PDF, TXT, or MD.`);
      return;
    }

    setCurrentFile(file); setUploading(true); setProgress(10); setStats(null);
    addLog("info", `Starting upload: ${file.name} (${formatBytes(file.size)})`);

    const formData = new FormData();
    formData.append("file", file);
    const ticker = setInterval(() => setProgress(p => Math.min(p + 8, 85)), 400);

    try {
      const res  = await fetch(`${API}/upload`, { method: "POST", body: formData });
      clearInterval(ticker);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (!data.success) throw new Error(data.message || "Upload failed");

      setProgress(100);
      setStats({
        chunks:        data.chunks_stored,
        entities:      data.entities_found,
        relationships: data.relationships_found,
      });
      addLog("success", `Parsed and embedded ${data.chunks_stored} chunks`);
      addLog("success", `Extracted ${data.entities_found} entities, ${data.relationships_found} relationships`);
      addLog("info", `Wiki generating in background…`);
      addLog("success", `${file.name} is ready for querying`);

      await refreshDocs();
      onDocumentAdded?.();          // ← notify Graph + Wiki to refresh
    } catch (err) {
      clearInterval(ticker);
      setProgress(0);
      addLog("error", `Upload failed: ${err.message}`);
    } finally {
      setUploading(false);
      setTimeout(() => setProgress(0), 2000);
    }
  }, [refreshDocs, onDocumentAdded]);

  const onFileChange = (e) => { const file = e.target.files?.[0]; if (file) uploadFile(file); e.target.value = ""; };
  const onDrop = (e) => { e.preventDefault(); setDragOver(false); uploadFile(e.dataTransfer.files?.[0]); };

  // ── Delete one ─────────────────────────────────────────────────────────────
  const handleDeleteConfirmed = async () => {
    const filename = confirmDelete;
    setConfirmDelete(null);
    setDeletingFile(filename);
    try {
      const res  = await fetch(`${API}/delete-document`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename }),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.message);
      addLog("success", `'${filename}' removed (${data.chunks_deleted} chunks deleted)`);
      await refreshDocs();
      onDocumentRemoved?.();        // ← notify Graph + Wiki to refresh
    } catch (err) {
      addLog("error", `Failed to delete '${filename}': ${err.message}`);
    } finally {
      setDeletingFile(null);
    }
  };

  // ── Clear all ──────────────────────────────────────────────────────────────
  const handleClearAllConfirmed = async () => {
    setConfirmClearAll(false);
    addLog("warning", "Clearing entire knowledge base…");
    try {
      const res  = await fetch(`${API}/clear-documents`, { method: "POST" });
      const data = await res.json();
      if (!data.success) throw new Error(data.message);
      addLog("success", `Cleared: ${data.chunks_deleted} chunks, ${data.files_deleted} files`);
      addLog("info", "Conversation memory and knowledge graph reset");
      setStats(null);
      setDocs([]);
      onKnowledgeBaseCleared?.();   // clears ChatBox history
      onDocumentRemoved?.();        // refreshes Graph + Wiki
    } catch (err) {
      addLog("error", `Clear failed: ${err.message}`);
    }
  };

  const logIconMap = {
    success: "ti-circle-check",
    error:   "ti-alert-circle",
    info:    "ti-info-circle",
    warning: "ti-alert-triangle",
  };
  const totalChunks = docs.reduce((s, d) => s + d.chunks, 0);

  return (
    <div className="upload-wrap">
      {confirmClearAll && (
        <ConfirmModal
          title="Clear entire knowledge base?"
          body="This will permanently delete all uploaded documents, their embeddings, the knowledge graph, all conversation sessions, and all wiki pages."
          confirmLabel="Yes, clear everything"
          danger
          onConfirm={handleClearAllConfirmed}
          onCancel={() => setConfirmClearAll(false)}
        />
      )}
      {confirmDelete && (
        <ConfirmModal
          title={`Remove "${confirmDelete}"?`}
          body="This will delete the file's embeddings from the vector store and the physical file from disk. Other documents and conversation history are unaffected."
          confirmLabel="Remove document"
          danger
          onConfirm={handleDeleteConfirmed}
          onCancel={() => setConfirmDelete(null)}
        />
      )}

      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
        <div>
          <div className="page-title">Upload Documents</div>
          <div className="page-subtitle">PDF, Markdown, or plain text · Max 50 MB</div>
        </div>
        {docs.length > 0 && (
          <button
            className="btn btn-danger" style={{ marginTop: 2 }}
            onClick={() => setConfirmClearAll(true)}
            title="Delete all documents and reset the knowledge base"
          >
            <i className="ti ti-database-off" /> Clear knowledge base
          </button>
        )}
      </div>

      {/* Drop zone */}
      <div
        className={`upload-zone${dragOver ? " drag-over" : ""}`}
        onClick={() => !uploading && fileRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        role="button" tabIndex={0} aria-label="Upload document"
        onKeyDown={e => e.key === "Enter" && fileRef.current?.click()}
      >
        <input ref={fileRef} type="file" accept=".pdf,.txt,.md" onChange={onFileChange} aria-hidden="true" />
        <i
          className={`ti ${uploading ? "ti-loader-2" : "ti-cloud-upload"} zone-icon`}
          style={uploading ? { animation: "spin 1s linear infinite", display: "block" } : {}}
          aria-hidden="true"
        />
        <div className="zone-title">
          {uploading ? `Processing ${currentFile?.name}…` : "Drop a file here or click to browse"}
        </div>
        <div className="zone-sub">
          {uploading ? "Parsing, chunking, embedding…" : "Supported: PDF · TXT · MD"}
        </div>
      </div>

      {/* Progress bar */}
      {(uploading || progress > 0) && currentFile && (
        <div className="upload-progress">
          <div className="upload-progress-header">
            <div className="progress-filename">
              <i className={`ti ${fileIcon(currentFile.name)}`} style={{ marginRight: 6 }} />
              {currentFile.name}
            </div>
            <span style={{ fontSize: 12, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>{progress}%</span>
          </div>
          <div className="progress-bar-track" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
            <div className="progress-bar-fill" style={{ width: `${progress}%`, background: progress === 100 ? "var(--teal)" : "var(--accent)" }} />
          </div>
        </div>
      )}

      {/* Stats */}
      {stats && (
        <div className="upload-stats">
          <div className="stat-card"><div className="stat-label">Chunks stored</div><div className="stat-value accent">{stats.chunks}</div></div>
          <div className="stat-card"><div className="stat-label">Entities found</div><div className="stat-value teal">{stats.entities}</div></div>
          <div className="stat-card"><div className="stat-label">Relationships</div><div className="stat-value amber">{stats.relationships}</div></div>
        </div>
      )}

      {/* Knowledge base list */}
      <div className="upload-log">
        <div className="upload-log-header" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span>
            <i className="ti ti-database" style={{ marginRight: 6, verticalAlign: "middle" }} />
            Knowledge base
            {docs.length > 0 && (
              <span style={{ marginLeft: 8, fontSize: 10, padding: "1px 7px", background: "var(--accent-dim)", color: "var(--accent-light)", borderRadius: 999, fontFamily: "var(--font-mono)" }}>
                {docs.length} {docs.length === 1 ? "doc" : "docs"} · {totalChunks} chunks
              </span>
            )}
          </span>
          <button className="btn" style={{ padding: "2px 8px", fontSize: 11, gap: 4 }} onClick={refreshDocs} disabled={docsLoading} title="Refresh list">
            <i className={`ti ${docsLoading ? "ti-loader-2" : "ti-refresh"}`} style={docsLoading ? { animation: "spin 1s linear infinite" } : {}} />
          </button>
        </div>
        <div style={{ maxHeight: 240, overflowY: "auto" }}>
          {docs.length === 0 && !docsLoading && (
            <div style={{ padding: "20px 16px", textAlign: "center", color: "var(--text-muted)", fontSize: 13, fontStyle: "italic" }}>
              No documents yet. Upload one above.
            </div>
          )}
          {docsLoading && docs.length === 0 && (
            <div style={{ padding: "16px", textAlign: "center", color: "var(--text-muted)", fontSize: 13 }}>
              <i className="ti ti-loader-2" style={{ animation: "spin 1s linear infinite", marginRight: 6 }} />Loading…
            </div>
          )}
          {docs.map(doc => (
            <DocRow
              key={doc.filename} doc={doc}
              onDelete={name => setConfirmDelete(name)}
              deleting={deletingFile}
            />
          ))}
        </div>
        {docs.length > 0 && (
          <div style={{ padding: "10px 16px", borderTop: "1px solid var(--border-subtle)", fontSize: 11, color: "var(--text-muted)", lineHeight: 1.6 }}>
            <i className="ti ti-bulb" style={{ marginRight: 5, color: "var(--amber)" }} />
            <strong style={{ color: "var(--text-secondary)" }}>Tip:</strong> Fewer, focused documents reduce retrieval noise.
          </div>
        )}
      </div>

      {/* Activity log */}
      {log.length > 0 && (
        <div className="upload-log">
          <div className="upload-log-header" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span><i className="ti ti-terminal" style={{ marginRight: 6, verticalAlign: "middle" }} />Activity log</span>
            <button className="btn" style={{ padding: "2px 8px", fontSize: 11, gap: 4, color: "var(--red)", borderColor: "var(--red-dim)" }} onClick={() => setLog([])} title="Clear log">
              <i className="ti ti-x" />Clear
            </button>
          </div>
          <div className="log-entries">
            {log.map((entry, i) => (
              <div key={i} className={`log-entry ${entry.type}`}>
                <i className={`ti ${logIconMap[entry.type] || "ti-info-circle"}`} />
                <span style={{ flex: 1 }}>{entry.msg}</span>
                <span style={{ color: "var(--text-muted)", fontSize: 10 }}>{entry.time}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeUp { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
      `}</style>
    </div>
  );
}