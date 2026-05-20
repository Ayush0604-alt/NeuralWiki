import { useState, useCallback, useRef } from "react";
import Upload from "./pages/Upload";
import ChatBox from "./components/ChatBox";
import KnowledgeGraph from "./components/KnowledgeGraph";
import Wiki from "./components/Wiki";
import { API_BASE_URL } from "./api/config";
import "./App.css";

const NAV = [
  { id: "chat",   icon: "ti-message-2",    label: "Chat" },
  { id: "wiki",   icon: "ti-book",         label: "Wiki" },
  { id: "graph",  icon: "ti-share-2",      label: "Graph" },
  { id: "upload", icon: "ti-cloud-upload", label: "Upload" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("chat");
  const [backendStatus, setBackendStatus] = useState("checking");

  // ── Lightweight pub/sub event bus ──────────────────────────────────────────
  // Supported events: "kbCleared" | "documentAdded" | "documentRemoved"
  // Components subscribe via the factory props below.
  const listeners = useRef({
    kbCleared:       [],
    documentAdded:   [],
    documentRemoved: [],
  });

  const emit = useCallback((event) => {
    (listeners.current[event] || []).forEach(fn => fn());
  }, []);

  const subscribe = useCallback((event, listener) => {
    listeners.current[event] = [...(listeners.current[event] || []), listener];
    return () => {
      listeners.current[event] = (listeners.current[event] || []).filter(fn => fn !== listener);
    };
  }, []);

  // Stable factories passed as props — each wraps `subscribe`
  const onKnowledgeBaseCleared = useCallback(
    (fn) => subscribe("kbCleared", fn),
    [subscribe],
  );
  const onDocumentAdded = useCallback(
    (fn) => subscribe("documentAdded", fn),
    [subscribe],
  );
  const onDocumentRemoved = useCallback(
    (fn) => subscribe("documentRemoved", fn),
    [subscribe],
  );

  useState(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/`);
        setBackendStatus(res.ok ? "online" : "offline");
      } catch {
        setBackendStatus("offline");
      }
    })();
  }, []);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-neural">Neural</span>
          <span className="brand-wiki">Wiki</span>
          <span className="brand-badge">AI</span>
        </div>
        <nav className="top-nav">
          {NAV.map(n => (
            <button
              key={n.id}
              className={`top-nav-btn${activeTab === n.id ? " active" : ""}`}
              onClick={() => setActiveTab(n.id)}
            >
              <i className={`ti ${n.icon}`} aria-hidden="true" />
              {n.label}
            </button>
          ))}
        </nav>
        <div className="status-pill" data-status={backendStatus}>
          <span className="status-dot" />
          {backendStatus === "online" ? "Backend online"
            : backendStatus === "offline" ? "Backend offline"
            : "Connecting…"}
        </div>
      </header>

      <main className="main-content">
        <div className={`tab-panel${activeTab === "chat"   ? " active" : ""}`}>
          <ChatBox onKnowledgeBaseCleared={onKnowledgeBaseCleared} />
        </div>
        <div className={`tab-panel${activeTab === "wiki"   ? " active" : ""}`}>
          {/* Wiki auto-refreshes when a document is added or removed */}
          <Wiki onDocumentAdded={onDocumentAdded} onDocumentRemoved={onDocumentRemoved} />
        </div>
        <div className={`tab-panel${activeTab === "graph"  ? " active" : ""}`}>
          {/* Graph auto-refreshes when a document is added or removed */}
          <KnowledgeGraph onDocumentAdded={onDocumentAdded} onDocumentRemoved={onDocumentRemoved} />
        </div>
        <div className={`tab-panel${activeTab === "upload" ? " active" : ""}`}>
          <Upload
            onKnowledgeBaseCleared={() => emit("kbCleared")}
            onDocumentAdded={() => emit("documentAdded")}
            onDocumentRemoved={() => emit("documentRemoved")}
          />
        </div>
      </main>
    </div>
  );
}