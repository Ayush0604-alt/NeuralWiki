import { useState, useRef, useCallback } from "react";

const API = "http://127.0.0.1:8000";
const ALLOWED = [".pdf", ".txt", ".md"];

function getExt(name) {
  return name?.split(".").pop()?.toLowerCase() || "";
}
function fileIcon(name) {
  const ext = getExt(name);
  if (ext === "pdf") return "ti-file-type-pdf";
  if (ext === "md") return "ti-file-description";
  return "ti-file-text";
}
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function Upload() {
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentFile, setCurrentFile] = useState(null);
  const [stats, setStats] = useState(null);
  const [log, setLog] = useState([]);
  const fileRef = useRef(null);

  const addLog = (type, msg) =>
    setLog(prev => [...prev, { type, msg, time: new Date().toLocaleTimeString() }]);

  const uploadFile = useCallback(async (file) => {
    if (!file) return;

    const ext = `.${getExt(file.name)}`;
    if (!ALLOWED.includes(ext)) {
      addLog("error", `Unsupported file type: ${ext}. Use PDF, TXT, or MD.`);
      return;
    }

    setCurrentFile(file);
    setUploading(true);
    setProgress(10);
    setStats(null);
    addLog("info", `Starting upload: ${file.name} (${formatBytes(file.size)})`);

    const formData = new FormData();
    formData.append("file", file);

    // simulate progress ticks while waiting
    const ticker = setInterval(() => {
      setProgress(p => Math.min(p + 8, 85));
    }, 400);

    try {
      const res = await fetch(`${API}/upload`, { method: "POST", body: formData });
      clearInterval(ticker);

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (!data.success) throw new Error(data.message || "Upload failed");

      setProgress(100);
      setStats({
        chunks: data.chunks_stored,
        entities: data.entities_found,
        relationships: data.relationships_found,
      });
      addLog("success", `Parsed and embedded ${data.chunks_stored} chunks`);
      addLog("success", `Extracted ${data.entities_found} entities, ${data.relationships_found} relationships`);
      addLog("success", `${file.name} ready for querying`);
    } catch (err) {
      clearInterval(ticker);
      setProgress(0);
      addLog("error", `Upload failed: ${err.message}`);
    } finally {
      setUploading(false);
      setTimeout(() => setProgress(0), 2000);
    }
  }, []);

  const onFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) uploadFile(file);
    e.target.value = "";
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) uploadFile(file);
  };

  const logIconMap = { success: "ti-circle-check", error: "ti-alert-circle", info: "ti-info-circle", warning: "ti-alert-triangle" };

  return (
    <div className="upload-wrap">
      <div className="page-title" style={{ marginBottom: 4 }}>Upload Documents</div>
      <div className="page-subtitle" style={{ marginBottom: 16 }}>PDF, Markdown, or plain text · Max 50 MB</div>

      {/* Drop zone */}
      <div
        className={`upload-zone${dragOver ? " drag-over" : ""}`}
        onClick={() => !uploading && fileRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        aria-label="Upload document"
        onKeyDown={e => e.key === "Enter" && fileRef.current?.click()}
      >
        <input ref={fileRef} type="file" accept=".pdf,.txt,.md" onChange={onFileChange} aria-hidden="true" />
        <i className={`ti ${uploading ? "ti-loader-2" : "ti-cloud-upload"} zone-icon`}
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

      {/* Progress */}
      {(uploading || progress > 0) && currentFile && (
        <div className="upload-progress">
          <div className="upload-progress-header">
            <div className="progress-filename">
              <i className={`ti ${fileIcon(currentFile.name)}`} aria-hidden="true" style={{ marginRight: 6 }} />
              {currentFile.name}
            </div>
            <span style={{ fontSize: 12, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
              {progress}%
            </span>
          </div>
          <div className="progress-bar-track" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
            <div className="progress-bar-fill" style={{ width: `${progress}%`, background: progress === 100 ? "var(--teal)" : "var(--accent)" }} />
          </div>
        </div>
      )}

      {/* Stats */}
      {stats && (
        <div className="upload-stats">
          <div className="stat-card">
            <div className="stat-label">Chunks stored</div>
            <div className="stat-value accent">{stats.chunks}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Entities found</div>
            <div className="stat-value teal">{stats.entities}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Relationships</div>
            <div className="stat-value amber">{stats.relationships}</div>
          </div>
        </div>
      )}

      {/* Activity log */}
      {log.length > 0 && (
        <div className="upload-log">
          <div className="upload-log-header">Activity Log</div>
          <div className="log-entries">
            {log.map((entry, i) => (
              <div key={i} className={`log-entry ${entry.type}`}>
                <i className={`ti ${logIconMap[entry.type] || "ti-info-circle"}`} aria-hidden="true" />
                <span style={{ flex: 1 }}>{entry.msg}</span>
                <span style={{ color: "var(--text-muted)", fontSize: 10 }}>{entry.time}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}