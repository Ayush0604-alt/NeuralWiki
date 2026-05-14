import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";

const API = "http://127.0.0.1:8000";

// Color by entity label category
const GROUP_COLORS = {
  ORG:      "#6c63ff",
  PERSON:   "#1fc791",
  GPE:      "#f5a623",
  LOC:      "#e05252",
  PRODUCT:  "#47a8e5",
  WORK_OF_ART: "#c47aff",
  EVENT:    "#ff6b6b",
  RELATION: "#9999aa",
  DEFAULT:  "#555568",
};

function groupColor(group) {
  return GROUP_COLORS[group] || GROUP_COLORS.DEFAULT;
}

// Memoized graph data transformation — avoids re-running on every render
function buildGraphData(rawGraph) {
  const nodes = [];
  const links = [];
  const nodeSet = new Set();

  rawGraph.forEach(item => {
    (item.entities || []).forEach(entity => {
      const id = entity.text?.trim();
      if (id && id.length >= 2 && !nodeSet.has(id)) {
        nodes.push({ id, group: entity.label || "DEFAULT", isEntity: true });
        nodeSet.add(id);
      }
    });

    (item.relationships || []).forEach(rel => {
      const src = rel.subject?.trim();
      const tgt = rel.object?.trim();
      const lbl = rel.relation?.trim();
      if (!src || !tgt || src.length < 2 || tgt.length < 2) return;

      links.push({ source: src, target: tgt, label: lbl });

      if (!nodeSet.has(src)) {
        nodes.push({ id: src, group: "RELATION", isEntity: false });
        nodeSet.add(src);
      }
      if (!nodeSet.has(tgt)) {
        nodes.push({ id: tgt, group: "RELATION", isEntity: false });
        nodeSet.add(tgt);
      }
    });
  });

  return { nodes, links };
}

export default function KnowledgeGraph() {
  const [rawGraph, setRawGraph] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedNode, setSelectedNode] = useState(null);
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 });

  const graphData = useMemo(() => buildGraphData(rawGraph), [rawGraph]);

  // Responsive canvas sizing
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width: Math.floor(width), height: Math.floor(height) });
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  const fetchGraph = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API}/knowledge-graph`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (!Array.isArray(data.graph)) throw new Error("Invalid graph format");
      setRawGraph(data.graph);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchGraph(); }, [fetchGraph]);

  const handleNodeClick = useCallback(node => {
    setSelectedNode(prev => prev?.id === node.id ? null : node);
  }, []);

  const nodeConnections = useMemo(() => {
    if (!selectedNode) return [];
    return graphData.links
      .filter(l => {
        const src = typeof l.source === "object" ? l.source.id : l.source;
        const tgt = typeof l.target === "object" ? l.target.id : l.target;
        return src === selectedNode.id || tgt === selectedNode.id;
      })
      .slice(0, 6);
  }, [selectedNode, graphData.links]);

  // Custom node painter
  const paintNode = useCallback((node, ctx, globalScale) => {
    const label = node.id;
    const fontSize = Math.max(10 / globalScale, 3);
    const r = node.isEntity ? 6 : 4;
    const color = groupColor(node.group);

    // ring for selected
    if (selectedNode?.id === node.id) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, r + 3, 0, 2 * Math.PI);
      ctx.fillStyle = `${color}33`;
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();

    if (globalScale > 0.6) {
      ctx.font = `${fontSize}px "IBM Plex Sans", sans-serif`;
      ctx.fillStyle = "#f0f0f5";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const maxLen = 18;
      const displayLabel = label.length > maxLen ? label.slice(0, maxLen) + "…" : label;
      ctx.fillText(displayLabel, node.x, node.y + r + fontSize + 1);
    }
  }, [selectedNode]);

  const uniqueGroups = useMemo(() => {
    const seen = new Set();
    return graphData.nodes
      .map(n => n.group)
      .filter(g => { if (seen.has(g)) return false; seen.add(g); return true; })
      .slice(0, 6);
  }, [graphData.nodes]);

  return (
    <div className="graph-wrap">
      <div className="graph-toolbar">
        <div>
          <div className="page-title">Knowledge Graph</div>
          <div className="page-subtitle">Entities and relationships extracted from documents</div>
        </div>
        <div className="graph-stats">
          <span className="graph-stat"><strong>{graphData.nodes.length}</strong> nodes</span>
          <span className="graph-stat"><strong>{graphData.links.length}</strong> edges</span>
        </div>
        <button
          className="btn"
          onClick={fetchGraph}
          disabled={loading}
          aria-label="Refresh graph"
        >
          <i className={`ti ${loading ? "ti-loader-2" : "ti-refresh"}`}
            style={loading ? { animation: "spin 1s linear infinite" } : {}}
            aria-hidden="true"
          />
          Refresh
        </button>
      </div>

      <div className="graph-canvas" ref={containerRef}>
        {error && (
          <div className="empty-state" style={{ position: "absolute", inset: 0 }}>
            <i className="ti ti-alert-circle" style={{ color: "var(--red)" }} />
            <h3>Failed to load graph</h3>
            <p>{error}</p>
            <button className="btn" onClick={fetchGraph} style={{ marginTop: 8 }}>Retry</button>
          </div>
        )}

        {!error && graphData.nodes.length === 0 && !loading && (
          <div className="empty-state" style={{ position: "absolute", inset: 0 }}>
            <i className="ti ti-share-2" />
            <h3>No graph data yet</h3>
            <p>Upload documents to automatically extract entities and relationships.</p>
          </div>
        )}

        {graphData.nodes.length > 0 && (
          <ForceGraph2D
            graphData={graphData}
            width={dimensions.width}
            height={dimensions.height}
            backgroundColor="#0f0f13"
            nodeCanvasObject={paintNode}
            nodeCanvasObjectMode={() => "replace"}
            linkColor={() => "#2a2a44"}
            linkWidth={1}
            linkDirectionalArrowLength={4}
            linkDirectionalArrowRelPos={1}
            nodeLabel={node => `${node.id} (${node.group})`}
            onNodeClick={handleNodeClick}
            cooldownTicks={80}
            nodeRelSize={6}
          />
        )}

        {/* Node detail panel */}
        {selectedNode && (
          <div className="graph-node-detail">
            <button
              className="node-detail-close"
              onClick={() => setSelectedNode(null)}
              aria-label="Close node detail"
            >
              <i className="ti ti-x" />
            </button>
            <div className="node-detail-title">{selectedNode.id}</div>
            <div className="node-detail-label">{selectedNode.group}</div>
            {nodeConnections.length > 0 && (
              <>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 6 }}>
                  CONNECTIONS
                </div>
                {nodeConnections.map((l, i) => {
                  const src = typeof l.source === "object" ? l.source.id : l.source;
                  const tgt = typeof l.target === "object" ? l.target.id : l.target;
                  const other = src === selectedNode.id ? tgt : src;
                  const dir = src === selectedNode.id ? "→" : "←";
                  return (
                    <div key={i} style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 4, display: "flex", gap: 4 }}>
                      <span style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>{dir}</span>
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--accent-light)" }}>{l.label}</span>
                      <span>{other}</span>
                    </div>
                  );
                })}
              </>
            )}
          </div>
        )}

        {/* Legend */}
        {uniqueGroups.length > 0 && (
          <div className="graph-legend">
            {uniqueGroups.map(g => (
              <div key={g} className="legend-item">
                <div className="legend-dot" style={{ background: groupColor(g) }} />
                {g}
              </div>
            ))}
          </div>
        )}
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}