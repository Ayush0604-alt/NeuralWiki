import { useEffect, useState, useCallback, useRef } from "react";
import * as d3 from "d3";

const API = "http://127.0.0.1:8000";

const CLUSTER_CONFIG = {
  CENTER:   { color: "#8b84ff", bg: "#8b84ff18", icon: "◉", label: "Center",       size: 20 },
  ENTITY:   { color: "#8b84ff", bg: "#8b84ff14", icon: "◉", label: "Entities",     size: 11 },
  CONCEPT:  { color: "#1fc791", bg: "#1fc79114", icon: "◈", label: "Concepts",     size: 11 },
  LOCATION: { color: "#e05252", bg: "#e0525214", icon: "◎", label: "Locations",    size: 11 },
  EVENT:    { color: "#ff9f43", bg: "#ff9f4314", icon: "◆", label: "Events",       size: 10 },
  DATE:     { color: "#00d2d3", bg: "#00d2d314", icon: "◇", label: "Dates",        size: 9  },
  ACTION:   { color: "#c47aff", bg: "#c47aff14", icon: "▶", label: "Actions",      size: 10 },
  QUANTITY: { color: "#54a0ff", bg: "#54a0ff14", icon: "▣", label: "Quantities",   size: 9  },
  TECH:     { color: "#47bfff", bg: "#47bfff14", icon: "⬡", label: "Technologies", size: 12 },
  MISC:     { color: "#6b6b80", bg: "#6b6b8014", icon: "·", label: "Other",        size: 8  },
};

const cfg = (cluster) => CLUSTER_CONFIG[cluster] || CLUSTER_CONFIG.MISC;

// Edge label colours by relationship type
const EDGE_COLORS = {
  "co-occurs with":     "#ffffff18",
  "related to":         "#ffffff22",
  "associated with":    "#ffffff22",
  "integrates with":    "#47bfff55",
  "uses":               "#47bfff44",
  "is part of":         "#1fc79144",
  "depends on":         "#f5a62344",
  "outperforms":        "#e0525244",
  "causes":             "#ff9f4344",
  "is type of":         "#c47aff44",
  "based on":           "#8b84ff44",
  "derived from":       "#8b84ff33",
  "enables":            "#1fc79133",
  "contradicts":        "#e0525255",
  "collaborated with":  "#54a0ff44",
};

function edgeColor(label) {
  if (!label) return "#ffffff18";
  const lo = label.toLowerCase();
  for (const [key, col] of Object.entries(EDGE_COLORS)) {
    if (lo.includes(key)) return col;
  }
  return "#ffffff20";
}

function edgeWidth(weight) {
  if (!weight) return 0.8;
  if (weight >= 3) return 2.2;   // LLM-extracted
  if (weight >= 2) return 1.5;   // SVO
  if (weight >= 1) return 1.0;   // co-occurrence
  return 0.6;                     // bridge edges (weight 0.5)
}

// ── Build flat graph from multi-doc API response ──────────────────────────────
function buildGraph(rawArray) {
  if (!rawArray?.length) return { nodes: [], links: [] };

  const nodeMap = new Map();
  const linkMap = new Map(); // key → link (deduplicate)

  rawArray.forEach((seg, segIdx) => {
    if (!seg) return;
    const centerId = seg.center?.id || `Document_${segIdx}`;

    // Center node
    if (!nodeMap.has(centerId)) {
      nodeMap.set(centerId, {
        id: centerId, cluster: "CENTER",
        r: rawArray.length > 1 ? 16 : 20,
        segIdx, docCenter: true,
      });
    }

    // Cluster nodes
    for (const [clusterName, nodes] of Object.entries(seg.clusters || {})) {
      for (const node of (nodes || [])) {
        if (!node.id || nodeMap.has(node.id)) continue;
        nodeMap.set(node.id, {
          id: node.id, cluster: clusterName,
          r: cfg(clusterName).size || 10,
          segIdx,
        });
      }
    }

    // Relationships — preserve weight and label
    for (const rel of (seg.relationships || [])) {
      const src = rel.source;
      const tgt = rel.target;
      if (!nodeMap.has(src) || !nodeMap.has(tgt)) continue;
      // Use undirected key for dedup — keep highest-weight version
      const key = [src, tgt].sort().join("|||");
      const existing = linkMap.get(key);
      const weight = rel.weight || 1;
      if (!existing || weight > (existing.weight || 1)) {
        linkMap.set(key, {
          source: src, target: tgt,
          label: rel.label || "",
          weight,
          segIdx,
        });
      }
    }
  });

  return {
    nodes: [...nodeMap.values()],
    links: [...linkMap.values()],
  };
}

// ── Tooltip ───────────────────────────────────────────────────────────────────
function Tooltip({ node, x, y, links }) {
  if (!node) return null;
  const c = cfg(node.cluster);
  const conns = links.filter(l => {
    const s = typeof l.source === "object" ? l.source.id : l.source;
    const t = typeof l.target === "object" ? l.target.id : l.target;
    return s === node.id || t === node.id;
  });
  const left = Math.min(x + 14, window.innerWidth - 310);
  const top  = Math.max(y - 10, 8);

  return (
    <div style={{
      position: "fixed", left, top, zIndex: 200, pointerEvents: "none",
      background: "var(--bg-surface)", border: `1px solid ${c.color}55`,
      borderRadius: 12, padding: "14px 16px", minWidth: 210, maxWidth: 290,
      boxShadow: `0 16px 40px rgba(0,0,0,0.5), 0 0 0 1px ${c.color}20`,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
        <div style={{
          width: 32, height: 32, borderRadius: "50%", flexShrink: 0,
          background: c.bg, border: `1.5px solid ${c.color}`,
          display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14,
        }}>{c.icon}</div>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: c.color, lineHeight: 1.3 }}>
            {node.id.length > 28 ? node.id.slice(0, 27) + "…" : node.id}
          </div>
          <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.7px" }}>
            {c.label} · {conns.length} connection{conns.length !== 1 ? "s" : ""}
          </div>
        </div>
      </div>
      {conns.slice(0, 5).map((l, i) => {
        const s = typeof l.source === "object" ? l.source.id : l.source;
        const t = typeof l.target === "object" ? l.target.id : l.target;
        const other = s === node.id ? t : s;
        const arrow = s === node.id ? "→" : "←";
        return (
          <div key={i} style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 11, color: "var(--text-secondary)", marginBottom: 4 }}>
            <span style={{ color: c.color, fontWeight: 700, flexShrink: 0 }}>{arrow}</span>
            <span style={{
              background: "var(--bg-raised)", border: "1px solid var(--border)",
              borderRadius: 3, padding: "0 5px", fontSize: 10,
              color: edgeColor(l.label) === "#ffffff18" ? "var(--text-muted)" : "#ccc",
              flexShrink: 0, maxWidth: 90, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
            }}>{l.label || "—"}</span>
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {other.length > 20 ? other.slice(0, 19) + "…" : other}
            </span>
          </div>
        );
      })}
      {conns.length > 5 && (
        <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>
          +{conns.length - 5} more connections
        </div>
      )}
    </div>
  );
}

// ── Filter legend ─────────────────────────────────────────────────────────────
function Legend({ clusters, activeFilters, onToggle, counts }) {
  if (!clusters.length) return null;
  return (
    <div style={{
      position: "absolute", top: 14, right: 14,
      display: "flex", flexDirection: "column", gap: 4, zIndex: 10,
    }}>
      <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 2, paddingLeft: 2,
        letterSpacing: "0.5px", textTransform: "uppercase" }}>Filter</div>
      {clusters.map(c => {
        const cl = cfg(c);
        const on = activeFilters.has(c);
        return (
          <button key={c} onClick={() => onToggle(c)} style={{
            display: "flex", alignItems: "center", gap: 7,
            background: on ? cl.bg : "var(--bg-surface)",
            border: `1px solid ${on ? cl.color + "55" : "var(--border)"}`,
            borderRadius: 999, padding: "4px 10px 4px 7px",
            fontSize: 11, cursor: "pointer", transition: "all 120ms",
            opacity: on ? 1 : 0.4,
          }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: cl.color, flexShrink: 0 }} />
            <span style={{ color: on ? cl.color : "var(--text-secondary)" }}>{cl.label}</span>
            <span style={{
              marginLeft: 2, borderRadius: 999, padding: "1px 5px",
              background: on ? cl.color + "22" : "var(--bg-raised)",
              color: on ? cl.color : "var(--text-muted)",
              fontSize: 10, fontFamily: "var(--font-mono)",
            }}>{counts[c] || 0}</span>
          </button>
        );
      })}
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────
export default function KnowledgeGraph() {
  const [rawGraph, setRawGraph]   = useState([]);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState("");
  const [tooltip, setTooltip]     = useState(null);
  const [searchQ, setSearchQ]     = useState("");
  const [activeFilters, setActiveFilters] = useState(new Set());
  const [edgeMode, setEdgeMode]   = useState("all"); // all | strong | semantic

  const svgRef  = useRef(null);
  const wrapRef = useRef(null);
  const simRef  = useRef(null);
  const zoomRef = useRef(null);
  const [dims, setDims] = useState({ w: 900, h: 600 });

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
      if (!Array.isArray(data.graph)) throw new Error("Invalid format");
      setRawGraph(data.graph);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { fetchGraph(); }, [fetchGraph]);

  useEffect(() => {
    const g = buildGraph(rawGraph);
    setGraphData(g);
    const clusters = [...new Set(g.nodes.map(n => n.cluster))].filter(c => c !== "CENTER");
    setActiveFilters(new Set(clusters));
  }, [rawGraph]);

  // Edge weight filter
  const filterWeight = edgeMode === "strong"   ? 2.0
                     : edgeMode === "semantic"  ? 2.5
                     : 0;

  const visible = useCallback(() => {
    const nodes = graphData.nodes.filter(n =>
      n.cluster === "CENTER" || activeFilters.has(n.cluster)
    );
    const ids = new Set(nodes.map(n => n.id));
    const links = graphData.links.filter(l => {
      const s = typeof l.source === "object" ? l.source.id : l.source;
      const t = typeof l.target === "object" ? l.target.id : l.target;
      return ids.has(s) && ids.has(t) && (l.weight || 1) >= filterWeight;
    });
    return { nodes, links };
  }, [graphData, activeFilters, filterWeight]);

  const matchIds = useCallback(() => {
    if (!searchQ.trim()) return new Set();
    const q = searchQ.toLowerCase();
    return new Set(graphData.nodes.filter(n => n.id.toLowerCase().includes(q)).map(n => n.id));
  }, [graphData, searchQ]);

  // ── D3 render ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!svgRef.current) return;
    const { nodes, links } = visible();
    if (!nodes.length) return;
    const { w, h } = dims;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const matches = matchIds();
    const centerNodes = nodes.filter(n => n.cluster === "CENTER");
    const isMulti = centerNodes.length > 1;

    // Defs
    const defs = svg.append("defs");

    // Glow
    const glow = defs.append("filter").attr("id", "glow")
      .attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
    glow.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "blur");
    const merge = glow.append("feMerge");
    merge.append("feMergeNode").attr("in", "blur");
    merge.append("feMergeNode").attr("in", "SourceGraphic");

    // Arrows per cluster
    [...new Set(nodes.map(n => n.cluster))].forEach(cluster => {
      const color = cfg(cluster).color;
      defs.append("marker").attr("id", `arr-${cluster}`)
        .attr("viewBox", "0 0 10 10").attr("refX", 26).attr("refY", 5)
        .attr("markerWidth", 4).attr("markerHeight", 4).attr("orient", "auto-start-reverse")
        .append("path").attr("d", "M1 1L9 5L1 9Z").attr("fill", color).attr("opacity", 0.6);
    });

    const g = svg.append("g");
    const zoom = d3.zoom().scaleExtent([0.05, 10])
      .on("zoom", e => g.attr("transform", e.transform));
    svg.call(zoom);
    zoomRef.current = zoom;

    // Initial positions
    if (isMulti) {
      const angle = (2 * Math.PI) / centerNodes.length;
      const orbit = Math.min(w, h) * 0.28;
      centerNodes.forEach((cn, i) => {
        cn.x = w / 2 + Math.cos(angle * i - Math.PI / 2) * orbit;
        cn.y = h / 2 + Math.sin(angle * i - Math.PI / 2) * orbit;
      });
      nodes.filter(n => n.cluster !== "CENTER").forEach(n => {
        const center = centerNodes.find(c => c.segIdx === n.segIdx) || centerNodes[0];
        const a = Math.random() * Math.PI * 2;
        const d = 60 + Math.random() * 120;
        n.x = (center.x || w / 2) + Math.cos(a) * d;
        n.y = (center.y || h / 2) + Math.sin(a) * d;
      });
    } else {
      const cn = centerNodes[0];
      if (cn) { cn.x = w / 2; cn.y = h / 2; }
      // Cluster-sector placement so same-type nodes start grouped
      const clGroups = {};
      nodes.filter(n => n.cluster !== "CENTER").forEach(n => {
        (clGroups[n.cluster] = clGroups[n.cluster] || []).push(n);
      });
      const clKeys = Object.keys(clGroups);
      clKeys.forEach((cl, ci) => {
        const base = (ci / clKeys.length) * 2 * Math.PI - Math.PI / 2;
        clGroups[cl].forEach((n, ni) => {
          const spread = (ni - (clGroups[cl].length - 1) / 2) * 0.4;
          const a = base + spread;
          const d = 100 + (ni % 4) * 45;
          n.x = w / 2 + Math.cos(a) * d;
          n.y = h / 2 + Math.sin(a) * d;
        });
      });
    }

    // Simulation — link distance based on edge weight
    if (simRef.current) simRef.current.stop();
    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id)
        .distance(l => {
          const w = l.weight || 1;
          // Higher weight = stronger relationship = shorter distance
          if (w >= 3) return 60;   // LLM-extracted
          if (w >= 2) return 80;   // SVO
          if (w >= 1) return 110;  // co-occurrence
          return 180;              // bridge edges
        })
        .strength(l => {
          const w = l.weight || 1;
          if (w >= 3) return 0.8;
          if (w >= 2) return 0.6;
          if (w >= 1) return 0.4;
          return 0.15;  // bridge: very weak pull
        })
      )
      .force("charge", d3.forceManyBody()
        .strength(d => d.cluster === "CENTER" ? -800 : -200)
        .distanceMax(500)
      )
      .force("collide", d3.forceCollide().radius(d => d.r + 18).strength(0.9))
      .force("center", isMulti ? null : d3.forceCenter(w / 2, h / 2).strength(0.03))
      .alphaDecay(0.015)
      .velocityDecay(0.38);

    simRef.current = sim;

    // ── Draw links ─────────────────────────────────────────────────────────
    const linkG = g.append("g");
    const linkEl = linkG.selectAll("line").data(links).join("line")
      .attr("stroke", l => edgeColor(l.label))
      .attr("stroke-width", l => edgeWidth(l.weight))
      .attr("stroke-opacity", l => {
        const w = l.weight || 1;
        return w >= 3 ? 0.75 : w >= 2 ? 0.55 : w >= 1 ? 0.35 : 0.15;
      })
      .attr("marker-end", l => {
        const t = typeof l.target === "object" ? l.target : nodes.find(n => n.id === l.target);
        return `url(#arr-${t?.cluster || "MISC"})`;
      });

    // Edge labels for strong edges only
    const strongLinks = links.filter(l => (l.weight || 1) >= 2);
    const linkLabelEl = linkG.selectAll("text").data(strongLinks).join("text")
      .attr("text-anchor", "middle").attr("font-size", 8)
      .attr("fill", "var(--text-muted)").attr("pointer-events", "none")
      .attr("opacity", 0.6)
      .text(l => (l.label || "").slice(0, 14));

    // ── Draw nodes ─────────────────────────────────────────────────────────
    const nodeG = g.append("g");
    const nodeEl = nodeG.selectAll("g").data(nodes).join("g")
      .attr("cursor", "pointer")
      .call(d3.drag()
        .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag",  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
        .on("end",   (ev, d) => {
          if (!ev.active) sim.alphaTarget(0);
          if (d.cluster !== "CENTER") { d.fx = null; d.fy = null; }
        })
      )
      .on("mouseover", (ev, d) => {
        setTooltip({ node: d, x: ev.clientX, y: ev.clientY });
        const connIds = new Set([d.id]);
        links.forEach(l => {
          const s = typeof l.source === "object" ? l.source.id : l.source;
          const t = typeof l.target === "object" ? l.target.id : l.target;
          if (s === d.id) connIds.add(t);
          if (t === d.id) connIds.add(s);
        });
        nodeEl.attr("opacity", n => connIds.has(n.id) ? 1 : 0.1);
        linkEl
          .attr("stroke-opacity", l => {
            const s = typeof l.source === "object" ? l.source.id : l.source;
            const t = typeof l.target === "object" ? l.target.id : l.target;
            return (s === d.id || t === d.id) ? 0.95 : 0.04;
          })
          .attr("stroke-width", l => {
            const s = typeof l.source === "object" ? l.source.id : l.source;
            const t = typeof l.target === "object" ? l.target.id : l.target;
            return (s === d.id || t === d.id) ? edgeWidth(l.weight) * 2 : 0.4;
          });
      })
      .on("mousemove", ev => setTooltip(p => p ? { ...p, x: ev.clientX, y: ev.clientY } : p))
      .on("mouseout", () => {
        setTooltip(null);
        nodeEl.attr("opacity", 1);
        linkEl
          .attr("stroke-opacity", l => {
            const w = l.weight || 1;
            return w >= 3 ? 0.75 : w >= 2 ? 0.55 : w >= 1 ? 0.35 : 0.15;
          })
          .attr("stroke-width", l => edgeWidth(l.weight));
      });

    if (matches.size > 0)
      nodeEl.attr("opacity", d => matches.has(d.id) || d.cluster === "CENTER" ? 1 : 0.08);

    // Pulse ring for center nodes
    nodeEl.filter(d => d.cluster === "CENTER").append("circle")
      .attr("r", d => d.r + 12)
      .attr("fill", "none")
      .attr("stroke", d => cfg(d.cluster).color + "25")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "5 3");

    // Search ring
    if (matches.size > 0) {
      nodeEl.filter(d => matches.has(d.id)).append("circle")
        .attr("r", d => d.r + 8)
        .attr("fill", "none").attr("stroke", "#fff")
        .attr("stroke-width", 1.5).attr("stroke-dasharray", "3 3");
    }

    // Main circle
    nodeEl.append("circle")
      .attr("r", d => d.r)
      .attr("fill", d => cfg(d.cluster).bg)
      .attr("stroke", d => cfg(d.cluster).color)
      .attr("stroke-width", d => d.cluster === "CENTER" ? 2 : 1.3)
      .attr("filter", d => d.cluster === "CENTER" ? "url(#glow)" : null);

    // Icon
    nodeEl.append("text")
      .text(d => cfg(d.cluster).icon)
      .attr("text-anchor", "middle").attr("dominant-baseline", "central")
      .attr("font-size", d => d.cluster === "CENTER" ? 12 : 8)
      .attr("pointer-events", "none");

    // Label
    nodeEl.append("text")
      .text(d => {
        const max = d.cluster === "CENTER" ? 20 : 16;
        return d.id.length > max ? d.id.slice(0, max - 1) + "…" : d.id;
      })
      .attr("text-anchor", "middle").attr("dominant-baseline", "hanging")
      .attr("y", d => d.r + 5)
      .attr("font-size", d => d.cluster === "CENTER" ? 11 : 8)
      .attr("font-weight", d => d.cluster === "CENTER" ? "600" : "400")
      .attr("fill", d => cfg(d.cluster).color)
      .attr("pointer-events", "none");

    // Tick
    sim.on("tick", () => {
      linkEl
        .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
      linkLabelEl
        .attr("x", d => ((d.source.x || 0) + (d.target.x || 0)) / 2)
        .attr("y", d => ((d.source.y || 0) + (d.target.y || 0)) / 2 - 4);
      nodeEl.attr("transform", d => `translate(${d.x || 0},${d.y || 0})`);
    });

    sim.on("end", () => {
      const b = g.node().getBBox();
      if (!b.width || !b.height) return;
      const scale = Math.min(0.88, Math.min(w / (b.width + 100), h / (b.height + 100)));
      const tx = (w - b.width * scale) / 2 - b.x * scale;
      const ty = (h - b.height * scale) / 2 - b.y * scale;
      svg.transition().duration(700).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
    });

    return () => simRef.current?.stop();
  }, [graphData, dims, activeFilters, searchQ, edgeMode]);

  const handleFit = () => {
    if (!svgRef.current || !zoomRef.current) return;
    const svg = d3.select(svgRef.current);
    const gEl = svg.select("g");
    if (gEl.empty()) return;
    const { w, h } = dims;
    const b = gEl.node().getBBox();
    if (!b.width) return;
    const scale = Math.min(0.88, Math.min(w / (b.width + 100), h / (b.height + 100)));
    const tx = (w - b.width * scale) / 2 - b.x * scale;
    const ty = (h - b.height * scale) / 2 - b.y * scale;
    svg.transition().duration(500).call(zoomRef.current.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
  };

  const { nodes, links } = visible();
  const allClusters = [...new Set(graphData.nodes.map(n => n.cluster))].filter(c => c !== "CENTER");
  const clusterCounts = Object.fromEntries(allClusters.map(c => [c, graphData.nodes.filter(n => n.cluster === c).length]));
  const matches = matchIds();
  const docCount = rawGraph.length;

  // Edge breakdown stats
  const llmEdges  = graphData.links.filter(l => (l.weight || 0) >= 3).length;
  const svoEdges  = graphData.links.filter(l => (l.weight || 0) >= 2 && (l.weight || 0) < 3).length;
  const coocEdges = graphData.links.filter(l => (l.weight || 0) >= 1 && (l.weight || 0) < 2).length;

  const toggleFilter = (c) => {
    setActiveFilters(prev => {
      const next = new Set(prev);
      if (next.has(c)) { if (next.size > 1) next.delete(c); }
      else next.add(c);
      return next;
    });
  };

  return (
    <div className="graph-wrap">
      <div className="graph-toolbar">
        <div>
          <div className="page-title">Knowledge Graph</div>
          <div className="page-subtitle">
            {docCount > 1 ? `${docCount} docs · multi-hub` : "Multi-relational entity graph"}
            {" — "}{graphData.links.length} edges
            {llmEdges > 0 && ` (${llmEdges} semantic, ${svoEdges} SVO, ${coocEdges} co-occurrence)`}
          </div>
        </div>

        {/* Search */}
        <div style={{ position: "relative" }}>
          <i className="ti ti-search" style={{ position: "absolute", left: 9, top: "50%", transform: "translateY(-50%)", fontSize: 12, color: "var(--text-muted)", pointerEvents: "none" }} />
          <input
            type="text" value={searchQ}
            onChange={e => setSearchQ(e.target.value)}
            placeholder="Find node…"
            style={{ background: "var(--bg-raised)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", color: "var(--text-primary)", fontFamily: "var(--font)", fontSize: 12, padding: "6px 26px 6px 28px", outline: "none", width: 140 }}
          />
          {searchQ && (
            <button onClick={() => setSearchQ("")} style={{ position: "absolute", right: 7, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", fontSize: 12, display: "flex", padding: 0 }}>
              <i className="ti ti-x" />
            </button>
          )}
        </div>
        {searchQ && <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{matches.size}/{graphData.nodes.length}</span>}

        {/* Edge filter */}
        <div style={{ display: "flex", gap: 4 }}>
          {[
            { id: "all",      label: "All edges" },
            { id: "strong",   label: "SVO+" },
            { id: "semantic", label: "Semantic" },
          ].map(m => (
            <button key={m.id} onClick={() => setEdgeMode(m.id)}
              className="btn"
              style={{
                padding: "5px 10px", fontSize: 11,
                background: edgeMode === m.id ? "var(--accent-dim)" : "var(--bg-raised)",
                borderColor: edgeMode === m.id ? "var(--accent-dim2)" : "var(--border)",
                color: edgeMode === m.id ? "var(--accent-light)" : "var(--text-muted)",
              }}>
              {m.label}
            </button>
          ))}
        </div>

        <div className="graph-stats">
          <span className="graph-stat"><strong>{nodes.length}</strong> nodes</span>
          <span className="graph-stat"><strong>{links.length}</strong> edges</span>
        </div>

        <button className="btn" onClick={handleFit} disabled={!nodes.length}>
          <i className="ti ti-focus-2" /> Fit
        </button>
        <button className="btn" onClick={fetchGraph} disabled={loading}>
          <i className={`ti ${loading ? "ti-loader-2" : "ti-refresh"}`}
             style={loading ? { animation: "spin 1s linear infinite" } : {}} />
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

        {!error && !graphData.nodes.length && !loading && (
          <div className="empty-state" style={{ position: "absolute", inset: 0 }}>
            <i className="ti ti-share-2" />
            <h3>No graph data yet</h3>
            <p>Upload documents — entities connect via semantic relationships, co-occurrence, and LLM-extracted links, not a hub-and-spoke structure.</p>
          </div>
        )}

        {graphData.nodes.length > 0 && (
          <>
            <svg ref={svgRef} width={dims.w} height={dims.h}
              style={{ display: "block", background: "transparent" }} />

            <Legend clusters={allClusters} activeFilters={activeFilters}
              onToggle={toggleFilter} counts={clusterCounts} />

            {tooltip && (
              <Tooltip node={tooltip.node} x={tooltip.x} y={tooltip.y}
                links={graphData.links} />
            )}

            {/* Edge type legend */}
            <div style={{
              position: "absolute", bottom: 12, left: 14,
              background: "var(--bg-surface)", border: "1px solid var(--border)",
              borderRadius: 10, padding: "8px 14px",
              display: "flex", gap: 14, alignItems: "center", zIndex: 10,
            }}>
              {[
                { label: "LLM semantic",  color: "#8b84ff", w: 2.2, count: llmEdges },
                { label: "SVO verb",      color: "#1fc791", w: 1.5, count: svoEdges },
                { label: "Co-occurrence", color: "#ffffff", w: 1.0, count: coocEdges },
                { label: "Bridge",        color: "#6b6b80", w: 0.6, count: graphData.links.filter(l => (l.weight||1) < 1).length },
              ].filter(e => e.count > 0).map((e, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <svg width="22" height="10">
                    <line x1="0" y1="5" x2="22" y2="5"
                      stroke={e.color} strokeWidth={e.w} strokeOpacity={0.7}
                      strokeDasharray={e.w < 1 ? "3 3" : "none"} />
                  </svg>
                  <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
                    {e.label} <span style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>({e.count})</span>
                  </span>
                </div>
              ))}
              <span style={{ fontSize: 10, color: "var(--text-muted)", marginLeft: 4 }}>
                Drag · Scroll to zoom
              </span>
            </div>

            {/* Multi-doc indicator */}
            {docCount > 1 && (
              <div style={{
                position: "absolute", top: 14, left: 14,
                background: "var(--bg-surface)", border: "1px solid var(--accent-dim2)",
                borderRadius: 10, padding: "7px 12px", zIndex: 10,
                display: "flex", flexDirection: "column", gap: 4, maxWidth: 180,
              }}>
                <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.6px", marginBottom: 2 }}>Documents</div>
                {rawGraph.map((seg, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{ width: 7, height: 7, borderRadius: "50%", background: cfg("CENTER").color, flexShrink: 0 }} />
                    <span style={{ fontSize: 11, color: "var(--text-secondary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {(seg.center?.id || `Doc ${i + 1}`).slice(0, 22)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}