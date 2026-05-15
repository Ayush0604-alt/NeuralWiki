import { useState, useCallback } from "react";
import Upload from "./pages/Upload";
import SearchBox from "./components/SearchBox";
import ChatBox from "./components/ChatBox";
import KnowledgeGraph from "./components/KnowledgeGraph";
import Wiki from "./components/Wiki";
import "./App.css";

const NAV = [
  { id: "chat",   icon: "ti-message-2",   label: "Chat" },
  { id: "wiki",   icon: "ti-book",        label: "Wiki" },
  { id: "search", icon: "ti-search",      label: "Search" },
  { id: "graph",  icon: "ti-share-2",     label: "Graph" },
  { id: "upload", icon: "ti-cloud-upload",label: "Upload" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("chat");
  const [backendStatus, setBackendStatus] = useState("checking");

  const checkBackend = useCallback(async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/");
      setBackendStatus(res.ok ? "online" : "offline");
    } catch {
      setBackendStatus("offline");
    }
  }, []);

  useState(() => { checkBackend(); }, []);

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
          {backendStatus === "online" ? "Backend online" : backendStatus === "offline" ? "Backend offline" : "Connecting…"}
        </div>
      </header>

      <main className="main-content">
        <div className={`tab-panel${activeTab === "chat"   ? " active" : ""}`}><ChatBox /></div>
        <div className={`tab-panel${activeTab === "wiki"   ? " active" : ""}`}><Wiki /></div>
        <div className={`tab-panel${activeTab === "search" ? " active" : ""}`}><SearchBox /></div>
        <div className={`tab-panel${activeTab === "graph"  ? " active" : ""}`}><KnowledgeGraph /></div>
        <div className={`tab-panel${activeTab === "upload" ? " active" : ""}`}><Upload /></div>
      </main>
    </div>
  );
}