import { useEffect, useState, useCallback, useRef } from "react";
import * as d3 from "d3";

const API = "http://127.0.0.1:8000";

// ── Universal cluster config (matches backend knowledge_graph_service.py) ────
const CLUSTER_CONFIG = {
  CENTER:   { color: "#8b84ff", bg: "#8b84ff18", icon: "◉", label: "Center" },
  ENTITY:   { color: "#8b84ff", bg: "#8b84ff14", icon: "◉", label: "Entities" },
  CONCEPT:  { color: "#1fc791", bg: "#1fc79114", icon: "◈", label: "Concepts" },
  LOCATION: { color: "#e05252", bg: "#e0525214", icon: "◎", label: "Locations" },
  EVENT:    { color: "#ff9f43", bg: "#ff9f4314", icon: "◆", label: "Events" },
  DATE:     { color: "#00d2d3", bg: "#00d2d314", icon: "◇", label: "Dates" },
  ACTION:   { color: "#c47aff", bg: "#c47aff14", icon: "▶", label: "Actions" },
  QUANTITY: { color: "#54a0ff", bg: "#54a0ff14", icon: "▣", label: "Quantities" },
  RELATION: { color: "#f5a623", bg: "#f5a62314", icon: "⟷", label: "Relations" },
  TECH:     { color: "#47bfff", bg: "#47bfff14", icon: "⬡", label: "Technologies" },
  MISC:     { color: "#6b6b80", bg: "#6b6b8014", icon: "·",  label: "Other" },
  // Legacy resume-style clusters (backward compat)
  SKILL:     { color: "#1fc791", bg: "#1fc79114", icon: "⚡", label: "Skills" },
  COMPANY:   { color: "#f5a623", bg: "#f5a62314", icon: "🏢", label: "Companies" },
  EDUCATION: { color: "#47bfff", bg: "#47bfff14", icon: "🎓", label: "Education" },
  ROLE:      { color: "#c47aff", bg: "#c47aff14", icon: "👤", label: "Roles" },
  PROJECT:   { color: "#ff9f43", bg: "#ff9f4314", icon: "🔧", label: "Projects" },
  CONTACT:   { color: "#54a0ff", bg: "#54a0ff14", icon: "✉",  label: "Contact" },
};

function clusterCfg(cluster) {
  return CLUSTER_CONFIG[cluster] || CLUSTER_CONFIG.MISC;
}

// ── Flatten multi-doc graph from API into D3 nodes + links ───────────────────
function buildD3Graph(rawGraphArray) {
  if (!rawGraphArray || rawGraphArray.length === 0) return { nodes: [], links: [] };

  const allNodes = new Map(); // id → cluster
  const allLinks = [];
  let centerNode = null;

  for (const seg of rawGraphArray) {
    if (!seg) continue;

    if (seg.center?.id) {
      if (!centerNode) centerNode = seg.center;
      allNodes.set(seg.center.id, "CENTER");
    }

    if (seg.clusters) {
      for (const [clusterName, nodes] of Object.entries(seg.clusters)) {
        for (const node of nodes) {
          if (!allNodes.has(node.id)) {
            allNodes.set(node.id, clusterName);
          }
        }
      }
    }

    if (seg.relationships) {
      for (const rel of seg.relationships) {
        allLinks.push(rel);
      }
    }
  }

  const nodeMap = new Map();
  const nodes = [];
  for (const [id, cluster] of allNodes) {
    const n = { id, cluster, r: cluster === "CENTER" ? 20 : 11 };
    nodes.push(n);
    nodeMap.set(id, n);
  }

  const seenLinks = new Set();
  const links = [];
  for (const rel of allLinks) {
    const key = `${rel.source}||${rel.target}`;
    if (seenLinks.has(key)) continue;
    if (!nodeMap.has(rel.source) || !nodeMap.has(rel.target)) continue;
    seenLinks.add(key);
    links.push({ source: rel.source, target: rel.target, label: rel.label || "→" });
  }

  return { nodes, links, center: centerNode };
}

// ── Tooltip ──────────────────────────────────────────────────────────────────
function Tooltip({ node, x, y, links }) {
  if (!node) return null;
  const cfg = clusterCfg(node.cluster);

  const connections = links.filter(l => {
    const s = typeof l.source === "object" ? l.source.id : l.source;
    const t = typeof l.target === "object" ? l.target.id : l.target;
    return s === node.id || t === node.id;
  });

  // keep tooltip inside viewport
  const left = Math.min(x + 16, window.innerWidth - 310);
  const top  = Math.max(y - 8, 10);

  return (
    <div style={{
      position: "fixed", left, top,
      background: "var(--bg-surface)",
      border: `1px solid ${cfg.color}55`,
      borderRadius: 12, padding: "13px 15px",
      minWidth: 210, maxWidth: 290, zIndex: 200,
      pointerEvents: "none",
      boxShadow: `0 12px 36px rgba(0,0,0,0.45), 0 0 0 1px ${cfg.color}20`,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 8 }}>
        <div style={{
          width: 30, height: 30, borderRadius: "50%",
          background: cfg.bg, border: `1.5px solid ${cfg.color}`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 13, flexShrink: 0,
        }}>
          {cfg.icon}
        </div>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: cfg.color, lineHeight: 1.3 }}>
            {node.id.length > 30 ? node.id.slice(0, 29) + "…" : node.id}
          </div>
          <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.6px", marginTop: 1 }}>
            {cfg.label}
          </div>
        </div>
      </div>

      {connections.length > 0 && (
        <div style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: 8 }}>
          <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 6, letterSpacing: "0.4px" }}>
            {connections.length} connection{connections.length !== 1 ? "s" : ""}
          </div>
          {connections.slice(0, 5).map((l, i) => {
            const src = typeof l.source === "object" ? l.source.id : l.source;
            const tgt = typeof l.target === "object" ? l.target.id : l.target;
            const other = src === node.id ? tgt : src;
            const arrow = src === node.id ? "→" : "←";
            const otherCfg = clusterCfg(
              (typeof l.source === "object" ? l.source : { cluster: "MISC" }).cluster
            );
            return (
              <div key={i} style={{
                display: "flex", gap: 6, alignItems: "center",
                fontSize: 11, color: "var(--text-secondary)", marginBottom: 4,
              }}>
                <span style={{ color: cfg.color, fontSize: 10, fontWeight: 600 }}>{arrow}</span>
                <span style={{
                  background: "var(--bg-raised)", border: "1px solid var(--border)",
                  borderRadius: 4, padding: "1px 5px", fontSize: 10,
                  color: "var(--text-muted)", fontFamily: "var(--font-mono)",
                  flexShrink: 0, maxWidth: 80, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                }}>
                  {l.label}
                </span>
                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {other.length > 22 ? other.slice(0, 21) + "…" : other}
                </span>
              </div>
            );
          })}
          {connections.length > 5 && (
            <div style={{ fontSize: 10, color: "var(--text-muted)" }}>
              +{connections.length - 5} more
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Search highlight ─────────────────────────────────────────────────────────
function SearchBar({ onSearch, onClear, matchCount, total }) {
  const [val, setVal] = useState("");
  const handle = (v) => { setVal(v); onSearch(v); };
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ position: "relative" }}>
        <i className="ti ti-search" style={{
          position: "absolute", left: 9, top: "50%", transform: "translateY(-50%)",
          fontSize: 13, color: "var(--text-muted)", pointerEvents: "none",
        }} aria-hidden="true" />
        <input
          type="text"
          value={val}
          onChange={e => handle(e.target.value)}
          placeholder="Find node…"
          style={{
            background: "var(--bg-raised)", border: "1px solid var(--border)",
            borderRadius: "var(--radius-md)", color: "var(--text-primary)",
            fontFamily: "var(--font)", fontSize: 12, padding: "6px 10px 6px 30px",
            outline: "none", width: 160,
          }}
          aria-label="Search nodes"
        />
        {val && (
          <button onClick={() => handle("")} style={{
            position: "absolute", right: 6, top: "50%", transform: "translateY(-50%)",
            background: "none", border: "none", cursor: "pointer",
            color: "var(--text-muted)", fontSize: 13, padding: 0, display: "flex",
          }} aria-label="Clear search">
            <i className="ti ti-x" />
          </button>
        )}
      </div>
      {val && (
        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
          {matchCount}/{total}
        </span>
      )}
    </div>
  );
}

// ── Cluster filter sidebar ───────────────────────────────────────────────────
function ClusterPills({ clusters, active, onToggle, counts }) {
  if (clusters.length === 0) return null;
  return (
    <div style={{
      position: "absolute", top: 16, right: 16,
      display: "flex", flexDirection: "column", gap: 5, zIndex: 10,
    }}>
      {clusters.map(c => {
        const cfg = clusterCfg(c);
        const isActive = active.has(c);
        const count = counts[c] || 0;
        return (
          <button key={c} onClick={() => onToggle(c)} style={{
            display: "flex", alignItems: "center", gap: 8,
            background: isActive ? cfg.bg : "var(--bg-surface)",
            border: `1px solid ${isActive ? cfg.color + "66" : "var(--border)"}`,
            borderRadius: 999, padding: "4px 10px 4px 8px",
            fontSize: 11, cursor: "pointer", transition: "all 150ms",
            opacity: isActive ? 1 : 0.45,
          }}>
            <div style={{
              width: 8, height: 8, borderRadius: "50%",
              background: cfg.color, flexShrink: 0,
            }} />
            <span style={{ color: isActive ? cfg.color : "var(--text-secondary)" }}>
              {cfg.label}
            </span>
            <span style={{
              marginLeft: 2, background: isActive ? cfg.color + "25" : "var(--bg-raised)",
              color: isActive ? cfg.color : "var(--text-muted)",
              borderRadius: 999, padding: "1px 6px",
              fontSize: 10, fontFamily: "var(--font-mono)",
            }}>
              {count}
            </span>
          </button>
        );
      })}
      <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2, paddingLeft: 4 }}>
        Click to filter
      </div>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────
export default function KnowledgeGraph() {
  const [rawGraph, setRawGraph]       = useState([]);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState("");
  const [tooltip, setTooltip]         = useState(null);
  const [graphData, setGraphData]     = useState({ nodes: [], links: [] });
  const [searchQuery, setSearchQuery] = useState("");
  const [activeFilters, setActiveFilters] = useState(new Set());

  const svgRef  = useRef(null);
  const wrapRef = useRef(null);
  const simRef  = useRef(null);
  const zoomRef = useRef(null);
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

  const fetchGraph = useCallback(async () => {
    setLoading(true); setError("");
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

  useEffect(() => {
    const g = buildD3Graph(rawGraph);
    setGraphData(g);
    // init filters to all active
    const clusters = [...new Set(g.nodes.map(n => n.cluster))].filter(c => c !== "CENTER");
    setActiveFilters(new Set(clusters));
  }, [rawGraph]);

  // Derived: visible nodes/links after filter + search
  const visibleData = useCallback(() => {
    let nodes = graphData.nodes;
    let links = graphData.links;

    // cluster filter (center always visible)
    nodes = nodes.filter(n => n.cluster === "CENTER" || activeFilters.has(n.cluster));
    const visIds = new Set(nodes.map(n => n.id));
    links = links.filter(l => {
      const s = typeof l.source === "object" ? l.source.id : l.source;
      const t = typeof l.target === "object" ? l.target.id : l.target;
      return visIds.has(s) && visIds.has(t);
    });

    return { nodes, links };
  }, [graphData, activeFilters]);

  // Search match IDs
  const matchIds = useCallback(() => {
    if (!searchQuery.trim()) return new Set();
    const q = searchQuery.toLowerCase();
    return new Set(graphData.nodes.filter(n => n.id.toLowerCase().includes(q)).map(n => n.id));
  }, [graphData, searchQuery]);

  // ── D3 rendering ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (!svgRef.current) return;
    const { nodes, links } = visibleData();
    if (nodes.length === 0) return;

    const { w, h } = dims;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const matches = matchIds();

    // Arrow defs per cluster
    const defs = svg.append("defs");
    const clusterNames = [...new Set(nodes.map(n => n.cluster))];
    for (const cluster of clusterNames) {
      const color = clusterCfg(cluster).color;
      defs.append("marker")
        .attr("id", `arr-${cluster}`)
        .attr("viewBox", "0 0 10 10").attr("refX", 24).attr("refY", 5)
        .attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto-start-reverse")
        .append("path").attr("d", "M2 1L8 5L2 9")
        .attr("fill", "none").attr("stroke", color)
        .attr("stroke-width", 1.5).attr("stroke-linecap", "round").attr("stroke-linejoin", "round");
    }

    const g = svg.append("g");

    // Zoom
    const zoom = d3.zoom()
      .scaleExtent([0.15, 5])
      .on("zoom", e => g.attr("transform", e.transform));
    svg.call(zoom);
    zoomRef.current = zoom;

    // ── Force simulation ───────────────────────────────────────────────────
    const centerNode = nodes.find(n => n.cluster === "CENTER");

    // Group non-center nodes by cluster for radial layout
    const clusterGroups = {};
    for (const n of nodes) {
      if (n.cluster === "CENTER") continue;
      (clusterGroups[n.cluster] = clusterGroups[n.cluster] || []).push(n);
    }
    const clusterList = Object.keys(clusterGroups);
    const angleStep = (2 * Math.PI) / Math.max(clusterList.length, 1);
    const sectorAngles = {};
    clusterList.forEach((cl, i) => { sectorAngles[cl] = i * angleStep - Math.PI / 2; });

    const orbit = Math.min(w, h) * 0.3;
    const inner = Math.min(w, h) * 0.14;

    nodes.forEach(n => {
      if (n.cluster === "CENTER") {
        n.x = w / 2; n.y = h / 2; n.fx = w / 2; n.fy = h / 2;
      } else {
        const grp = clusterGroups[n.cluster] || [];
        const idx = grp.indexOf(n);
        const spread = grp.length > 1 ? (idx - (grp.length - 1) / 2) * 0.28 : 0;
        const angle = (sectorAngles[n.cluster] || 0) + spread;
        const dist = inner + (idx % 2) * (orbit - inner) * 0.5;
        n.x = w / 2 + Math.cos(angle) * dist;
        n.y = h / 2 + Math.sin(angle) * dist;
      }
    });

    if (simRef.current) simRef.current.stop();

    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id)
        .distance(d => {
          const tgt = typeof d.target === "object" ? d.target : nodes.find(n => n.id === d.target);
          return tgt?.cluster === "CENTER" ? orbit * 0.9 : 90;
        })
        .strength(0.55)
      )
      .force("charge", d3.forceManyBody().strength(d => d.cluster === "CENTER" ? -900 : -260))
      .force("collide", d3.forceCollide().radius(d => d.r + 18).strength(0.8))
      .force("radial", d3.forceRadial(d => d.cluster === "CENTER" ? 0 : orbit, w / 2, h / 2).strength(0.12))
      .alphaDecay(0.022).velocityDecay(0.38);
    simRef.current = sim;

    // ── Links ──────────────────────────────────────────────────────────────
    const linkEl = g.append("g").attr("class", "links").selectAll("line")
      .data(links).join("line")
      .attr("stroke", d => {
        const tgt = typeof d.target === "object" ? d.target : nodes.find(n => n.id === d.target);
        return clusterCfg(tgt?.cluster || "MISC").color + "44";
      })
      .attr("stroke-width", d => {
        const tgt = typeof d.target === "object" ? d.target : nodes.find(n => n.id === d.target);
        return tgt?.cluster === "CENTER" ? 1.8 : 1;
      })
      .attr("marker-end", d => {
        const tgt = typeof d.target === "object" ? d.target : nodes.find(n => n.id === d.target);
        return `url(#arr-${tgt?.cluster || "MISC"})`;
      });

    // ── Nodes ──────────────────────────────────────────────────────────────
    const nodeEl = g.append("g").attr("class", "nodes").selectAll("g")
      .data(nodes).join("g")
      .attr("cursor", "pointer")
      .call(
        d3.drag()
          .on("start", (event, d) => {
            if (!event.active) sim.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on("end", (event, d) => {
            if (!event.active) sim.alphaTarget(0);
            if (d.cluster !== "CENTER") { d.fx = null; d.fy = null; }
          })
      )
      .on("mouseover", (event, d) => {
        setTooltip({ node: d, x: event.clientX, y: event.clientY });
        linkEl.attr("stroke-opacity", l => {
          const s = typeof l.source === "object" ? l.source.id : l.source;
          const t = typeof l.target === "object" ? l.target.id : l.target;
          return (s === d.id || t === d.id) ? 1 : 0.07;
        }).attr("stroke-width", l => {
          const s = typeof l.source === "object" ? l.source.id : l.source;
          const t = typeof l.target === "object" ? l.target.id : l.target;
          return (s === d.id || t === d.id) ? 2.8 : 1;
        });
        nodeEl.attr("opacity", n => {
          if (n.id === d.id) return 1;
          return links.some(l => {
            const s = typeof l.source === "object" ? l.source.id : l.source;
            const t = typeof l.target === "object" ? l.target.id : l.target;
            return (s === d.id && t === n.id) || (t === d.id && s === n.id);
          }) ? 1 : 0.2;
        });
      })
      .on("mousemove", event => {
        setTooltip(prev => prev ? { ...prev, x: event.clientX, y: event.clientY } : prev);
      })
      .on("mouseout", () => {
        setTooltip(null);
        linkEl.attr("stroke-opacity", 1).attr("stroke-width", d => {
          const tgt = typeof d.target === "object" ? d.target : nodes.find(n => n.id === d.target);
          return tgt?.cluster === "CENTER" ? 1.8 : 1;
        });
        nodeEl.attr("opacity", 1);
      });

    // Search dim/highlight
    if (matches.size > 0) {
      nodeEl.attr("opacity", d => matches.has(d.id) || d.cluster === "CENTER" ? 1 : 0.15);
    }

    // Outer pulse ring for center
    nodeEl.filter(d => d.cluster === "CENTER")
      .append("circle")
      .attr("r", d => d.r + 12)
      .attr("fill", d => clusterCfg(d.cluster).color + "10")
      .attr("stroke", d => clusterCfg(d.cluster).color + "30")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "4 4");

    // Search ring
    if (matches.size > 0) {
      nodeEl.filter(d => matches.has(d.id))
        .append("circle")
        .attr("r", d => d.r + 7)
        .attr("fill", "none")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "3 3");
    }

    // Circle fill
    nodeEl.append("circle")
      .attr("r", d => d.r)
      .attr("fill", d => clusterCfg(d.cluster).bg)
      .attr("stroke", d => clusterCfg(d.cluster).color)
      .attr("stroke-width", d => d.cluster === "CENTER" ? 2.2 : 1.5);

    // Icon
    nodeEl.append("text")
      .text(d => clusterCfg(d.cluster).icon)
      .attr("text-anchor", "middle").attr("dominant-baseline", "central")
      .attr("font-size", d => d.cluster === "CENTER" ? 12 : 8)
      .attr("pointer-events", "none");

    // Label
    nodeEl.append("text")
      .text(d => {
        const label = d.id;
        return label.length > 20 ? label.slice(0, 19) + "…" : label;
      })
      .attr("text-anchor", "middle").attr("dominant-baseline", "hanging")
      .attr("y", d => d.r + 5)
      .attr("font-size", d => d.cluster === "CENTER" ? 13 : 10)
      .attr("font-weight", d => d.cluster === "CENTER" ? "600" : "400")
      .attr("fill", d => clusterCfg(d.cluster).color)
      .attr("font-family", "var(--font, system-ui, sans-serif)")
      .attr("pointer-events", "none");

    // Extra label above center
    nodeEl.filter(d => d.cluster === "CENTER")
      .append("text")
      .text(d => d.id.length > 24 ? d.id.slice(0, 23) + "…" : d.id)
      .attr("text-anchor", "middle").attr("dominant-baseline", "auto")
      .attr("y", d => -(d.r + 10))
      .attr("font-size", 14).attr("font-weight", "700")
      .attr("fill", d => clusterCfg(d.cluster).color)
      .attr("font-family", "var(--font, system-ui, sans-serif)")
      .attr("pointer-events", "none");

    // Tick
    sim.on("tick", () => {
      linkEl.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
             .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
      nodeEl.attr("transform", d => `translate(${d.x},${d.y})`);
    });

    // Fit after settle
    sim.on("end", () => {
      const b = g.node().getBBox();
      if (!b.width || !b.height) return;
      const scale = Math.min(0.88, Math.min(w / (b.width + 100), h / (b.height + 100)));
      const tx = (w - b.width * scale) / 2 - b.x * scale;
      const ty = (h - b.height * scale) / 2 - b.y * scale;
      svg.transition().duration(700).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
    });

    return () => { if (simRef.current) simRef.current.stop(); };
  }, [graphData, dims, activeFilters, searchQuery]);

  // Zoom fit handler
  const handleFit = () => {
    if (!svgRef.current || !zoomRef.current) return;
    const svg = d3.select(svgRef.current);
    const g = svg.select("g");
    if (g.empty()) return;
    const { w, h } = dims;
    const b = g.node().getBBox();
    if (!b.width || !b.height) return;
    const scale = Math.min(0.88, Math.min(w / (b.width + 100), h / (b.height + 100)));
    const tx = (w - b.width * scale) / 2 - b.x * scale;
    const ty = (h - b.height * scale) / 2 - b.y * scale;
    svg.transition().duration(500).call(zoomRef.current.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
  };

  // Stats
  const { nodes, links } = visibleData();
  const allClusters = [...new Set(graphData.nodes.map(n => n.cluster))].filter(c => c !== "CENTER");
  const clusterCounts = Object.fromEntries(allClusters.map(c => [c, graphData.nodes.filter(n => n.cluster === c).length]));
  const matches = matchIds();

  const toggleFilter = (cluster) => {
    setActiveFilters(prev => {
      const next = new Set(prev);
      if (next.has(cluster)) {
        if (next.size > 1) next.delete(cluster); // keep at least one
      } else {
        next.add(cluster);
      }
      return next;
    });
  };

  return (
    <div className="graph-wrap">
      <div className="graph-toolbar">
        <div>
          <div className="page-title">Knowledge Graph</div>
          <div className="page-subtitle">Semantic entity map — hover nodes for connections</div>
        </div>

        <SearchBar
          onSearch={setSearchQuery}
          matchCount={matches.size}
          total={graphData.nodes.length}
        />

        <div className="graph-stats">
          <span className="graph-stat"><strong>{nodes.length}</strong> nodes</span>
          <span className="graph-stat"><strong>{links.length}</strong> edges</span>
          <span className="graph-stat"><strong>{allClusters.length}</strong> types</span>
        </div>

        <button className="btn" onClick={handleFit} disabled={nodes.length === 0} title="Zoom to fit">
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

      <div className="graph-canvas" ref={wrapRef} style={{ position: "relative" }}>
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
            <p>Upload and process a document — entities, concepts, locations, and more will appear here as a navigable knowledge graph.</p>
          </div>
        )}

        {graphData.nodes.length > 0 && (
          <>
            <svg
              ref={svgRef}
              width={dims.w}
              height={dims.h}
              style={{ display: "block", background: "transparent" }}
            />

            <ClusterPills
              clusters={allClusters}
              active={activeFilters}
              onToggle={toggleFilter}
              counts={clusterCounts}
            />

            {tooltip && (
              <Tooltip node={tooltip.node} x={tooltip.x} y={tooltip.y} links={graphData.links} />
            )}

            {/* Bottom legend */}
            <div style={{
              position: "absolute", bottom: 14, left: 14,
              background: "var(--bg-surface)", border: "1px solid var(--border)",
              borderRadius: 10, padding: "8px 14px",
              display: "flex", flexWrap: "wrap", gap: "7px 14px",
              maxWidth: 500, zIndex: 10,
            }}>
              {allClusters.filter(c => activeFilters.has(c)).map(c => {
                const cfg = clusterCfg(c);
                return (
                  <div key={c} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                    <div style={{ width: 7, height: 7, borderRadius: "50%", background: cfg.color }} />
                    <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{cfg.label}</span>
                  </div>
                );
              })}
              <span style={{ fontSize: 10, color: "var(--text-muted)", marginLeft: "auto", alignSelf: "center" }}>
                Drag to reposition · Scroll to zoom
              </span>
            </div>
          </>
        )}
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}