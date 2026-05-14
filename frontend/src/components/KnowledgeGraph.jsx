import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";

const API = "http://127.0.0.1:8000";

// ── Color palette by entity label ────────────────────────────────────────────
const GROUP_COLORS = {
  ORG:         "#6c63ff",
  PERSON:      "#1fc791",
  GPE:         "#f5a623",
  LOC:         "#e05252",
  PRODUCT:     "#47a8e5",
  WORK_OF_ART: "#c47aff",
  EVENT:       "#ff6b6b",
  FAC:         "#ff9f43",
  LANGUAGE:    "#54a0ff",
  LAW:         "#5f27cd",
  NORP:        "#00d2d3",
  RELATION:    "#555568",
  DEFAULT:     "#444458",
};

// Human-readable descriptions for the legend
const GROUP_LABELS = {
  ORG:         "Organization",
  PERSON:      "Person",
  GPE:         "Place / Country",
  LOC:         "Location",
  PRODUCT:     "Product",
  WORK_OF_ART: "Work of Art",
  EVENT:       "Event",
  FAC:         "Facility",
  LANGUAGE:    "Language",
  LAW:         "Law / Regulation",
  NORP:        "Group / Nationality",
  RELATION:    "Relation (inferred)",
};

function groupColor(group) {
  return GROUP_COLORS[group] || GROUP_COLORS.DEFAULT;
}

// ── Graph data builder ────────────────────────────────────────────────────────
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
      const lbl = rel.relation?.trim() || "→";
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

// ── Truncate helper ───────────────────────────────────────────────────────────
const truncate = (s, n) => (s && s.length > n ? s.slice(0, n) + "…" : s);

// ── Component ─────────────────────────────────────────────────────────────────
export default function KnowledgeGraph() {
  const [rawGraph, setRawGraph] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoveredLink, setHoveredLink] = useState(null);
  const containerRef = useRef(null);
  const fgRef = useRef(null);
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

  // Connections for the selected node detail panel
  const nodeConnections = useMemo(() => {
    if (!selectedNode) return [];
    return graphData.links
      .filter(l => {
        const src = typeof l.source === "object" ? l.source.id : l.source;
        const tgt = typeof l.target === "object" ? l.target.id : l.target;
        return src === selectedNode.id || tgt === selectedNode.id;
      })
      .slice(0, 8);
  }, [selectedNode, graphData.links]);

  // ── Custom node painter ───────────────────────────────────────────────────
  const paintNode = useCallback((node, ctx, globalScale) => {
    const label = node.id;
    const fontSize = Math.max(10 / globalScale, 3);
    const isSelected = selectedNode?.id === node.id;
    const r = node.isEntity ? 6 : 4;
    const color = groupColor(node.group);

    // Glow ring for selected node
    if (isSelected) {
      const gradient = ctx.createRadialGradient(node.x, node.y, r, node.x, node.y, r + 8);
      gradient.addColorStop(0, `${color}55`);
      gradient.addColorStop(1, `${color}00`);
      ctx.beginPath();
      ctx.arc(node.x, node.y, r + 8, 0, 2 * Math.PI);
      ctx.fillStyle = gradient;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(node.x, node.y, r + 2.5, 0, 2 * Math.PI);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5 / globalScale;
      ctx.stroke();
    }

    // Node circle with subtle inner highlight
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();

    // Tiny white specular dot
    if (globalScale > 0.5) {
      ctx.beginPath();
      ctx.arc(node.x - r * 0.28, node.y - r * 0.28, r * 0.28, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(255,255,255,0.25)";
      ctx.fill();
    }

    // Label — only when zoomed in enough
    if (globalScale > 0.55) {
      ctx.font = `${fontSize}px "IBM Plex Sans", sans-serif`;
      ctx.fillStyle = isSelected ? color : "#c0c0d0";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(truncate(label, 20), node.x, node.y + r + 2);
    }
  }, [selectedNode]);

  // ── Custom link painter (shows relation label on hover) ───────────────────
  const paintLink = useCallback((link, ctx, globalScale) => {
    const src = link.source;
    const tgt = link.target;
    if (!src || !tgt || src.x == null || tgt.x == null) return;

    const isHovered = hoveredLink === link;

    ctx.beginPath();
    ctx.moveTo(src.x, src.y);
    ctx.lineTo(tgt.x, tgt.y);
    ctx.strokeStyle = isHovered ? "#6c63ff88" : "#2a2a4488";
    ctx.lineWidth = isHovered ? 1.5 / globalScale : 0.8 / globalScale;
    ctx.stroke();

    // Draw relation label at the midpoint when hovered or zoomed in
    if ((isHovered || globalScale > 1.2) && link.label) {
      const mx = (src.x + tgt.x) / 2;
      const my = (src.y + tgt.y) / 2;
      const fontSize = Math.max(8 / globalScale, 2.5);

      ctx.font = `italic ${fontSize}px "IBM Plex Sans", sans-serif`;
      const text = link.label;
      const tw = ctx.measureText(text).width;

      // Pill background
      const pad = 2 / globalScale;
      ctx.fillStyle = "rgba(15,15,19,0.85)";
      ctx.fillRect(mx - tw / 2 - pad, my - fontSize / 2 - pad, tw + pad * 2, fontSize + pad * 2);

      ctx.fillStyle = isHovered ? "#8b84ff" : "#6666aa";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, mx, my);
    }
  }, [hoveredLink]);

  // Unique groups actually present in the current graph (for legend)
  const uniqueGroups = useMemo(() => {
    const seen = new Set();
    return graphData.nodes
      .map(n => n.group)
      .filter(g => { if (seen.has(g)) return false; seen.add(g); return true; })
      .slice(0, 7);
  }, [graphData.nodes]);

  // ── Zoom to fit ───────────────────────────────────────────────────────────
  const handleZoomFit = () => {
    fgRef.current?.zoomToFit(400, 40);
  };

  return (
    <div className="graph-wrap">
      {/* ── Toolbar ── */}
      <div className="graph-toolbar">
        <div>
          <div className="page-title">Knowledge Graph</div>
          <div className="page-subtitle">
            Entities and relationships extracted using spaCy's dependency parser
          </div>
        </div>
        <div className="graph-stats">
          <span className="graph-stat"><strong>{graphData.nodes.length}</strong> nodes</span>
          <span className="graph-stat"><strong>{graphData.links.length}</strong> edges</span>
        </div>
        <button
          className="btn"
          onClick={handleZoomFit}
          disabled={graphData.nodes.length === 0}
          aria-label="Zoom to fit"
          title="Zoom to fit all nodes"
        >
          <i className="ti ti-focus-2" aria-hidden="true" />
          Fit
        </button>
        <button
          className="btn"
          onClick={fetchGraph}
          disabled={loading}
          aria-label="Refresh graph"
        >
          <i
            className={`ti ${loading ? "ti-loader-2" : "ti-refresh"}`}
            style={loading ? { animation: "spin 1s linear infinite" } : {}}
            aria-hidden="true"
          />
          Refresh
        </button>
      </div>

      {/* ── Canvas ── */}
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
            <p>
              Upload documents to automatically extract named entities and
              subject-verb-object relationships.
            </p>
          </div>
        )}

        {graphData.nodes.length > 0 && (
          <ForceGraph2D
            ref={fgRef}
            graphData={graphData}
            width={dimensions.width}
            height={dimensions.height}
            backgroundColor="#0f0f13"
            nodeCanvasObject={paintNode}
            nodeCanvasObjectMode={() => "replace"}
            linkCanvasObject={paintLink}
            linkCanvasObjectMode={() => "replace"}
            linkDirectionalArrowLength={4}
            linkDirectionalArrowRelPos={0.85}
            linkDirectionalArrowColor={() => "#6c63ff55"}
            nodeLabel={node => `${node.id} · ${GROUP_LABELS[node.group] || node.group}`}
            onNodeClick={handleNodeClick}
            onLinkHover={link => setHoveredLink(link)}
            cooldownTicks={100}
            nodeRelSize={6}
            d3AlphaDecay={0.02}
            d3VelocityDecay={0.3}
            onEngineStop={handleZoomFit}
          />
        )}

        {/* ── Node detail panel ── */}
        {selectedNode && (
          <div className="graph-node-detail">
            <button
              className="node-detail-close"
              onClick={() => setSelectedNode(null)}
              aria-label="Close node detail"
            >
              <i className="ti ti-x" />
            </button>

            <div
              className="node-detail-title"
              style={{ color: groupColor(selectedNode.group) }}
            >
              {selectedNode.id}
            </div>
            <div className="node-detail-label">
              {GROUP_LABELS[selectedNode.group] || selectedNode.group}
            </div>

            {nodeConnections.length > 0 && (
              <>
                <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 8, letterSpacing: "0.6px", textTransform: "uppercase" }}>
                  Relationships
                </div>
                {nodeConnections.map((l, i) => {
                  const src = typeof l.source === "object" ? l.source.id : l.source;
                  const tgt = typeof l.target === "object" ? l.target.id : l.target;
                  const isOutgoing = src === selectedNode.id;
                  const other = isOutgoing ? tgt : src;
                  return (
                    <div
                      key={i}
                      style={{
                        fontSize: 12,
                        color: "var(--text-secondary)",
                        marginBottom: 6,
                        display: "flex",
                        alignItems: "center",
                        gap: 5,
                        flexWrap: "wrap",
                      }}
                    >
                      <span style={{ fontSize: 10, color: isOutgoing ? "var(--accent-light)" : "var(--teal)", fontFamily: "var(--font-mono)" }}>
                        {isOutgoing ? "→" : "←"}
                      </span>
                      <span
                        style={{
                          fontFamily: "var(--font-mono)",
                          fontSize: 10,
                          color: "var(--text-muted)",
                          background: "var(--bg-raised)",
                          border: "1px solid var(--border)",
                          borderRadius: 4,
                          padding: "1px 5px",
                        }}
                      >
                        {l.label}
                      </span>
                      <span style={{ fontSize: 12 }}>{truncate(other, 22)}</span>
                    </div>
                  );
                })}
              </>
            )}

            {nodeConnections.length === 0 && (
              <div style={{ fontSize: 12, color: "var(--text-muted)", fontStyle: "italic" }}>
                No relationships found
              </div>
            )}
          </div>
        )}

        {/* ── Legend ── */}
        {uniqueGroups.length > 0 && (
          <div className="graph-legend" style={{ flexWrap: "wrap", maxWidth: 420 }}>
            {uniqueGroups.map(g => (
              <div key={g} className="legend-item">
                <div className="legend-dot" style={{ background: groupColor(g) }} />
                {GROUP_LABELS[g] || g}
              </div>
            ))}
            <div className="legend-item" style={{ marginLeft: "auto", color: "var(--text-muted)", fontSize: 10 }}>
              Hover edges for labels
            </div>
          </div>
        )}
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}