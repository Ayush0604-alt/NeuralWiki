import { useEffect, useState, useCallback, useRef } from "react";
import * as d3 from "d3";

const API = "http://127.0.0.1:8000";

// ── Cluster visual config ────────────────────────────────────────────────────
const CLUSTER_CONFIG = {
  CENTER:    { color: "#8b84ff", bg: "#8b84ff22", icon: "★", label: "Identity" },
  SKILL:     { color: "#1fc791", bg: "#1fc79115", icon: "⚡", label: "Skills & Tech" },
  COMPANY:   { color: "#f5a623", bg: "#f5a62315", icon: "🏢", label: "Companies" },
  EDUCATION: { color: "#47a8e5", bg: "#47a8e515", icon: "🎓", label: "Education" },
  LOCATION:  { color: "#e05252", bg: "#e0525215", icon: "📍", label: "Locations" },
  ROLE:      { color: "#c47aff", bg: "#c47aff15", icon: "👤", label: "Roles" },
  PROJECT:   { color: "#ff9f43", bg: "#ff9f4315", icon: "🔧", label: "Projects" },
  CONTACT:   { color: "#54a0ff", bg: "#54a0ff15", icon: "✉", label: "Contact" },
  DATE:      { color: "#00d2d3", bg: "#00d2d315", icon: "📅", label: "Dates" },
  MISC:      { color: "#555568", bg: "#55556815", icon: "•", label: "Other" },
};

function clusterColor(cluster) {
  return (CLUSTER_CONFIG[cluster] || CLUSTER_CONFIG.MISC).color;
}

// ── Flatten graph data from API into D3 nodes+links ──────────────────────────
function buildD3Graph(rawGraphArray) {
  const nodes = [];
  const links = [];
  const nodeMap = new Map();

  if (!rawGraphArray || rawGraphArray.length === 0) return { nodes, links };

  // Merge all graph segments
  let centerNode = null;
  const allNodes = new Map(); // id → cluster
  const allLinks = [];

  for (const segment of rawGraphArray) {
    if (!segment) continue;

    // Center node
    if (segment.center && segment.center.id) {
      if (!centerNode) centerNode = segment.center;
      allNodes.set(segment.center.id, "CENTER");
    }

    // Cluster nodes
    if (segment.clusters) {
      for (const [clusterName, clusterNodes] of Object.entries(segment.clusters)) {
        for (const node of clusterNodes) {
          if (!allNodes.has(node.id)) {
            allNodes.set(node.id, clusterName);
          }
        }
      }
    }

    // Relationships
    if (segment.relationships) {
      for (const rel of segment.relationships) {
        allLinks.push(rel);
      }
    }
  }

  // Build node list
  for (const [id, cluster] of allNodes) {
    const n = {
      id,
      cluster,
      r: cluster === "CENTER" ? 18 : 10,
    };
    nodes.push(n);
    nodeMap.set(id, n);
  }

  // Build link list (only between known nodes)
  const seenLinks = new Set();
  for (const rel of allLinks) {
    const key = `${rel.source}||${rel.target}`;
    if (seenLinks.has(key)) continue;
    if (!nodeMap.has(rel.source) || !nodeMap.has(rel.target)) continue;
    seenLinks.add(key);
    links.push({
      source: rel.source,
      target: rel.target,
      label: rel.label || "→",
    });
  }

  return { nodes, links, center: centerNode };
}

// ── Tooltip ──────────────────────────────────────────────────────────────────
function Tooltip({ node, x, y, links }) {
  if (!node) return null;
  const cfg = CLUSTER_CONFIG[node.cluster] || CLUSTER_CONFIG.MISC;
  const connections = links.filter(l => {
    const s = typeof l.source === "object" ? l.source.id : l.source;
    const t = typeof l.target === "object" ? l.target.id : l.target;
    return s === node.id || t === node.id;
  });

  return (
    <div style={{
      position: "fixed",
      left: x + 16,
      top: y - 8,
      background: "var(--bg-surface)",
      border: `1px solid ${cfg.color}44`,
      borderRadius: 10,
      padding: "12px 14px",
      minWidth: 200,
      maxWidth: 280,
      zIndex: 100,
      pointerEvents: "none",
      boxShadow: `0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px ${cfg.color}22`,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <div style={{
          width: 28, height: 28, borderRadius: "50%",
          background: cfg.bg, border: `1.5px solid ${cfg.color}`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 12,
        }}>
          {cfg.icon}
        </div>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: cfg.color, lineHeight: 1.3 }}>
            {node.id}
          </div>
          <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.5px" }}>
            {cfg.label}
          </div>
        </div>
      </div>
      {connections.length > 0 && (
        <div style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: 8, marginTop: 6 }}>
          {connections.slice(0, 4).map((l, i) => {
            const src = typeof l.source === "object" ? l.source.id : l.source;
            const tgt = typeof l.target === "object" ? l.target.id : l.target;
            const other = src === node.id ? tgt : src;
            const arrow = src === node.id ? "→" : "←";
            return (
              <div key={i} style={{
                fontSize: 11, color: "var(--text-secondary)",
                display: "flex", gap: 6, alignItems: "center", marginBottom: 3,
              }}>
                <span style={{ color: cfg.color, fontSize: 10 }}>{arrow}</span>
                <span style={{
                  background: "var(--bg-raised)",
                  border: "1px solid var(--border)",
                  borderRadius: 4, padding: "1px 5px",
                  fontSize: 10, color: "var(--text-muted)",
                  fontFamily: "var(--font-mono)",
                }}>
                  {l.label}
                </span>
                <span style={{ fontSize: 11 }}>{other.length > 24 ? other.slice(0, 24) + "…" : other}</span>
              </div>
            );
          })}
          {connections.length > 4 && (
            <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>
              +{connections.length - 4} more connections
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Legend ────────────────────────────────────────────────────────────────────
function Legend({ clusters }) {
  if (!clusters || clusters.length === 0) return null;
  return (
    <div style={{
      position: "absolute", bottom: 16, left: 16,
      background: "var(--bg-surface)",
      border: "1px solid var(--border)",
      borderRadius: 10,
      padding: "10px 14px",
      display: "flex", flexWrap: "wrap", gap: "8px 16px",
      maxWidth: 520,
      zIndex: 10,
    }}>
      {clusters.map(c => {
        const cfg = CLUSTER_CONFIG[c] || CLUSTER_CONFIG.MISC;
        return (
          <div key={c} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{
              width: 8, height: 8, borderRadius: "50%",
              background: cfg.color, flexShrink: 0,
            }} />
            <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{cfg.label}</span>
          </div>
        );
      })}
      <div style={{ fontSize: 10, color: "var(--text-muted)", marginLeft: "auto", alignSelf: "center" }}>
        Hover nodes for details
      </div>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────
export default function KnowledgeGraph() {
  const [rawGraph, setRawGraph]       = useState([]);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState("");
  const [tooltip, setTooltip]         = useState(null); // { node, x, y }
  const [graphData, setGraphData]     = useState({ nodes: [], links: [] });
  const svgRef     = useRef(null);
  const wrapRef    = useRef(null);
  const simRef     = useRef(null);
  const [dims, setDims] = useState({ w: 900, h: 600 });

  // Responsive sizing
  useEffect(() => {
    if (!wrapRef.current) return;
    const ro = new ResizeObserver(e => {
      const { width, height } = e[0].contentRect;
      setDims({ w: Math.floor(width), h: Math.floor(height) });
    });
    ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, []);

  // Fetch graph data
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

  // Build D3 graph data when rawGraph changes
  useEffect(() => {
    setGraphData(buildD3Graph(rawGraph));
  }, [rawGraph]);

  // D3 rendering
  useEffect(() => {
    if (!svgRef.current || graphData.nodes.length === 0) return;

    const { w, h } = dims;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // ── Arrow marker ──────────────────────────────────────────────────────
    const defs = svg.append("defs");
    // One marker per cluster color
    const clusterNames = [...new Set(graphData.nodes.map(n => n.cluster))];
    for (const cluster of clusterNames) {
      const color = clusterColor(cluster);
      defs.append("marker")
        .attr("id", `arrow-${cluster}`)
        .attr("viewBox", "0 0 10 10")
        .attr("refX", 22)
        .attr("refY", 5)
        .attr("markerWidth", 5)
        .attr("markerHeight", 5)
        .attr("orient", "auto-start-reverse")
        .append("path")
        .attr("d", "M2 1L8 5L2 9")
        .attr("fill", "none")
        .attr("stroke", color)
        .attr("stroke-width", 1.5)
        .attr("stroke-linecap", "round")
        .attr("stroke-linejoin", "round");
    }

    const g = svg.append("g");

    // ── Zoom ──────────────────────────────────────────────────────────────
    const zoom = d3.zoom()
      .scaleExtent([0.2, 4])
      .on("zoom", e => g.attr("transform", e.transform));
    svg.call(zoom);

    // ── Force simulation ──────────────────────────────────────────────────
    const centerNode = graphData.nodes.find(n => n.cluster === "CENTER");

    // Group non-center nodes by cluster for radial positioning hints
    const clusterGroups = {};
    for (const n of graphData.nodes) {
      if (n.cluster === "CENTER") continue;
      if (!clusterGroups[n.cluster]) clusterGroups[n.cluster] = [];
      clusterGroups[n.cluster].push(n);
    }

    // Assign angular sectors to clusters
    const clusterList = Object.keys(clusterGroups);
    const angleStep = (2 * Math.PI) / Math.max(clusterList.length, 1);
    const sectorAngles = {};
    clusterList.forEach((cl, i) => { sectorAngles[cl] = i * angleStep; });

    // Initial positions in a hub-and-spoke pattern
    const orbitRadius = Math.min(w, h) * 0.32;
    const innerRadius = Math.min(w, h) * 0.18;

    graphData.nodes.forEach(n => {
      if (n.cluster === "CENTER") {
        n.x = w / 2; n.y = h / 2; n.fx = w / 2; n.fy = h / 2;
      } else {
        const baseAngle = sectorAngles[n.cluster] || 0;
        const grpNodes = clusterGroups[n.cluster] || [];
        const idx = grpNodes.indexOf(n);
        const spread = (grpNodes.length > 1)
          ? (idx - (grpNodes.length - 1) / 2) * 0.25
          : 0;
        const angle = baseAngle + spread;
        const dist = innerRadius + (idx % 2) * (orbitRadius - innerRadius) * 0.5;
        n.x = w / 2 + Math.cos(angle) * dist;
        n.y = h / 2 + Math.sin(angle) * dist;
      }
    });

    if (simRef.current) simRef.current.stop();

    const sim = d3.forceSimulation(graphData.nodes)
      .force("link", d3.forceLink(graphData.links)
        .id(d => d.id)
        .distance(d => {
          const tgt = typeof d.target === "object" ? d.target : graphData.nodes.find(n => n.id === d.target);
          return tgt?.cluster === "CENTER" ? orbitRadius * 0.85 : 80;
        })
        .strength(0.6)
      )
      .force("charge", d3.forceManyBody().strength(d => d.cluster === "CENTER" ? -800 : -220))
      .force("collide", d3.forceCollide().radius(d => d.r + 14))
      .force("cluster", () => {
        // Pull nodes towards their cluster sector angle
        for (const n of graphData.nodes) {
          if (n.cluster === "CENTER" || !sectorAngles[n.cluster]) continue;
          const angle = sectorAngles[n.cluster];
          const tx = w / 2 + Math.cos(angle) * orbitRadius;
          const ty = h / 2 + Math.sin(angle) * orbitRadius;
          n.vx += (tx - n.x) * 0.015;
          n.vy += (ty - n.y) * 0.015;
        }
      })
      .alphaDecay(0.025)
      .velocityDecay(0.4);

    simRef.current = sim;

    // ── Links ─────────────────────────────────────────────────────────────
    const linkGroup = g.append("g").attr("class", "links");

    const linkEl = linkGroup.selectAll("line")
      .data(graphData.links)
      .join("line")
      .attr("stroke", d => {
        const src = typeof d.source === "object" ? d.source : graphData.nodes.find(n => n.id === d.source);
        const tgt = typeof d.target === "object" ? d.target : graphData.nodes.find(n => n.id === d.target);
        const cluster = tgt?.cluster || src?.cluster || "MISC";
        return clusterColor(cluster) + "55";
      })
      .attr("stroke-width", d => {
        const tgt = typeof d.target === "object" ? d.target : graphData.nodes.find(n => n.id === d.target);
        return tgt?.cluster === "CENTER" ? 1.5 : 1;
      })
      .attr("marker-end", d => {
        const tgt = typeof d.target === "object" ? d.target : graphData.nodes.find(n => n.id === d.target);
        return `url(#arrow-${tgt?.cluster || "MISC"})`;
      });

    // Link labels (only shown when hovered via tooltip, not on canvas)

    // ── Nodes ─────────────────────────────────────────────────────────────
    const nodeGroup = g.append("g").attr("class", "nodes");

    const nodeEl = nodeGroup.selectAll("g")
      .data(graphData.nodes)
      .join("g")
      .attr("cursor", "pointer")
      .call(
        d3.drag()
          .on("start", (event, d) => {
            if (!event.active) sim.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x; d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) sim.alphaTarget(0);
            if (d.cluster !== "CENTER") { d.fx = null; d.fy = null; }
          })
      )
      .on("mouseover", (event, d) => {
        setTooltip({ node: d, x: event.clientX, y: event.clientY });
        // Highlight connected links
        linkEl.attr("stroke-opacity", l => {
          const s = typeof l.source === "object" ? l.source.id : l.source;
          const t = typeof l.target === "object" ? l.target.id : l.target;
          return (s === d.id || t === d.id) ? 1 : 0.08;
        }).attr("stroke-width", l => {
          const s = typeof l.source === "object" ? l.source.id : l.source;
          const t = typeof l.target === "object" ? l.target.id : l.target;
          return (s === d.id || t === d.id) ? 2.5 : 1;
        });
        // Dim other nodes
        nodeEl.attr("opacity", n => {
          if (n.id === d.id) return 1;
          const connected = graphData.links.some(l => {
            const s = typeof l.source === "object" ? l.source.id : l.source;
            const t = typeof l.target === "object" ? l.target.id : l.target;
            return (s === d.id && t === n.id) || (t === d.id && s === n.id);
          });
          return connected ? 1 : 0.25;
        });
      })
      .on("mousemove", (event) => {
        setTooltip(prev => prev ? { ...prev, x: event.clientX, y: event.clientY } : prev);
      })
      .on("mouseout", () => {
        setTooltip(null);
        linkEl.attr("stroke-opacity", 1).attr("stroke-width", d => {
          const tgt = typeof d.target === "object" ? d.target : graphData.nodes.find(n => n.id === d.target);
          return tgt?.cluster === "CENTER" ? 1.5 : 1;
        });
        nodeEl.attr("opacity", 1);
      });

    // Node glow ring (for center)
    nodeEl.filter(d => d.cluster === "CENTER")
      .append("circle")
      .attr("r", d => d.r + 10)
      .attr("fill", d => clusterColor(d.cluster) + "15")
      .attr("stroke", d => clusterColor(d.cluster) + "40")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "4 3");

    // Node circle
    nodeEl.append("circle")
      .attr("r", d => d.r)
      .attr("fill", d => {
        const cfg = CLUSTER_CONFIG[d.cluster] || CLUSTER_CONFIG.MISC;
        return cfg.bg;
      })
      .attr("stroke", d => clusterColor(d.cluster))
      .attr("stroke-width", d => d.cluster === "CENTER" ? 2 : 1.5);

    // Cluster icon (small, inside circle)
    nodeEl.append("text")
      .text(d => (CLUSTER_CONFIG[d.cluster] || CLUSTER_CONFIG.MISC).icon)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "central")
      .attr("y", d => d.cluster === "CENTER" ? -3 : 0)
      .attr("font-size", d => d.cluster === "CENTER" ? 11 : 8)
      .attr("pointer-events", "none");

    // Node label (below circle)
    nodeEl.append("text")
      .text(d => {
        const label = d.id;
        return label.length > 18 ? label.slice(0, 17) + "…" : label;
      })
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "hanging")
      .attr("y", d => d.r + 5)
      .attr("font-size", d => d.cluster === "CENTER" ? 13 : 10)
      .attr("font-weight", d => d.cluster === "CENTER" ? "600" : "400")
      .attr("fill", d => clusterColor(d.cluster))
      .attr("font-family", "var(--font, 'IBM Plex Sans', sans-serif)")
      .attr("pointer-events", "none");

    // Center name label (extra large, above circle)
    nodeEl.filter(d => d.cluster === "CENTER")
      .append("text")
      .text(d => d.id)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "auto")
      .attr("y", d => -(d.r + 10))
      .attr("font-size", 15)
      .attr("font-weight", "600")
      .attr("fill", d => clusterColor(d.cluster))
      .attr("font-family", "var(--font, 'IBM Plex Sans', sans-serif)")
      .attr("pointer-events", "none");

    // ── Simulation tick ───────────────────────────────────────────────────
    sim.on("tick", () => {
      linkEl
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      nodeEl.attr("transform", d => `translate(${d.x},${d.y})`);
    });

    // Initial zoom to fit
    sim.on("end", () => {
      const bounds = g.node().getBBox();
      if (!bounds.width || !bounds.height) return;
      const scale = Math.min(0.9, Math.min(w / (bounds.width + 80), h / (bounds.height + 80)));
      const tx = (w - bounds.width * scale) / 2 - bounds.x * scale;
      const ty = (h - bounds.height * scale) / 2 - bounds.y * scale;
      svg.transition().duration(600).call(
        zoom.transform,
        d3.zoomIdentity.translate(tx, ty).scale(scale)
      );
    });

    return () => { if (simRef.current) simRef.current.stop(); };
  }, [graphData, dims]);

  // Stats
  const nodeCount = graphData.nodes.length;
  const linkCount = graphData.links.length;
  const presentClusters = [...new Set(graphData.nodes.map(n => n.cluster))].filter(c => c !== "CENTER");

  const handleZoomFit = () => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const g = svg.select("g");
    if (g.empty()) return;
    const { w, h } = dims;
    const bounds = g.node().getBBox();
    if (!bounds.width || !bounds.height) return;
    const scale = Math.min(0.9, Math.min(w / (bounds.width + 80), h / (bounds.height + 80)));
    const tx = (w - bounds.width * scale) / 2 - bounds.x * scale;
    const ty = (h - bounds.height * scale) / 2 - bounds.y * scale;
    const zoom = d3.zoom().scaleExtent([0.2, 4]).on("zoom", e => g.attr("transform", e.transform));
    svg.call(zoom);
    svg.transition().duration(500).call(
      zoom.transform,
      d3.zoomIdentity.translate(tx, ty).scale(scale)
    );
  };

  return (
    <div className="graph-wrap">
      {/* Toolbar */}
      <div className="graph-toolbar">
        <div>
          <div className="page-title">Semantic Knowledge Graph</div>
          <div className="page-subtitle">
            Hierarchical entity graph — hub-and-spoke by entity type
          </div>
        </div>
        <div className="graph-stats">
          <span className="graph-stat"><strong>{nodeCount}</strong> nodes</span>
          <span className="graph-stat"><strong>{linkCount}</strong> edges</span>
          <span className="graph-stat"><strong>{presentClusters.length}</strong> clusters</span>
        </div>
        <button className="btn" onClick={handleZoomFit} disabled={nodeCount === 0} title="Zoom to fit">
          <i className="ti ti-focus-2" aria-hidden="true" /> Fit
        </button>
        <button className="btn" onClick={fetchGraph} disabled={loading}>
          <i
            className={`ti ${loading ? "ti-loader-2" : "ti-refresh"}`}
            style={loading ? { animation: "spin 1s linear infinite" } : {}}
            aria-hidden="true"
          />
          Refresh
        </button>
      </div>

      {/* Canvas */}
      <div className="graph-canvas" ref={wrapRef} style={{ position: "relative" }}>
        {error && (
          <div className="empty-state" style={{ position: "absolute", inset: 0 }}>
            <i className="ti ti-alert-circle" style={{ color: "var(--red)" }} />
            <h3>Failed to load graph</h3>
            <p>{error}</p>
            <button className="btn" onClick={fetchGraph} style={{ marginTop: 8 }}>Retry</button>
          </div>
        )}

        {!error && nodeCount === 0 && !loading && (
          <div className="empty-state" style={{ position: "absolute", inset: 0 }}>
            <i className="ti ti-share-2" />
            <h3>No graph data yet</h3>
            <p>
              Upload a resume, report, or any document — the graph will map
              the central identity outward to skills, companies, education,
              locations, and more.
            </p>
          </div>
        )}

        {nodeCount > 0 && (
          <svg
            ref={svgRef}
            width={dims.w}
            height={dims.h}
            style={{ display: "block", background: "transparent" }}
          />
        )}

        {/* Tooltip */}
        {tooltip && (
          <Tooltip
            node={tooltip.node}
            x={tooltip.x}
            y={tooltip.y}
            links={graphData.links}
          />
        )}

        {/* Legend */}
        <Legend clusters={presentClusters} />

        {/* Cluster color pills (top-right) */}
        {nodeCount > 0 && (
          <div style={{
            position: "absolute", top: 16, right: 16,
            display: "flex", flexDirection: "column", gap: 6, zIndex: 10,
          }}>
            {presentClusters.map(c => {
              const cfg = CLUSTER_CONFIG[c] || CLUSTER_CONFIG.MISC;
              const count = graphData.nodes.filter(n => n.cluster === c).length;
              return (
                <div key={c} style={{
                  display: "flex", alignItems: "center", gap: 8,
                  background: "var(--bg-surface)",
                  border: `1px solid ${cfg.color}44`,
                  borderRadius: 999,
                  padding: "4px 10px 4px 6px",
                  fontSize: 11,
                }}>
                  <div style={{
                    width: 8, height: 8, borderRadius: "50%",
                    background: cfg.color, flexShrink: 0,
                  }} />
                  <span style={{ color: "var(--text-secondary)" }}>{cfg.label}</span>
                  <span style={{
                    marginLeft: "auto",
                    background: cfg.bg, color: cfg.color,
                    borderRadius: 999, padding: "1px 6px",
                    fontSize: 10, fontFamily: "var(--font-mono)",
                  }}>{count}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}