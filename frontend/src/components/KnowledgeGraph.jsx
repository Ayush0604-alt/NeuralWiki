import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import * as d3 from "d3";

const API = "http://127.0.0.1:8000";

// ── Cluster config — used for SHAPE/ICON only, color is now community-based ──
const CLUSTER_CONFIG = {
  CENTER:   { icon: "◉", label: "Center",       baseSize: 20 },
  ENTITY:   { icon: "◉", label: "Entities",     baseSize: 10 },
  CONCEPT:  { icon: "◈", label: "Concepts",     baseSize: 10 },
  LOCATION: { icon: "◎", label: "Locations",    baseSize: 10 },
  EVENT:    { icon: "◆", label: "Events",        baseSize: 9  },
  DATE:     { icon: "◇", label: "Dates",         baseSize: 8  },
  ACTION:   { icon: "▶", label: "Actions",       baseSize: 9  },
  QUANTITY: { icon: "▣", label: "Quantities",    baseSize: 8  },
  TECH:     { icon: "⬡", label: "Technologies",  baseSize: 10 },
  MISC:     { icon: "·", label: "Other",          baseSize: 7  },
};

// 12-color categorical palette for communities
const COMMUNITY_PALETTE = [
  "#8b84ff", "#1fc791", "#ff9f43", "#e05252",
  "#47bfff", "#c47aff", "#00d2d3", "#54a0ff",
  "#f368e0", "#ffd32a", "#0be881", "#ff4d4d",
];

function communityColor(communityId) {
  return COMMUNITY_PALETTE[communityId % COMMUNITY_PALETTE.length];
}

const cfg = (cluster) => CLUSTER_CONFIG[cluster] || CLUSTER_CONFIG.MISC;

// Extraction method config
const EXTRACTION_METHODS = {
  "llm_semantic":   { label: "LLM Semantic", icon: "⚡", color: "#8b84ff" },
  "svo_verb":       { label: "SVO Verb",     icon: "→", color: "#1fc791" },
  "co_occurrence":  { label: "Co-occurrence", icon: "○", color: "#ffffff" },
  "bridge":         { label: "Bridge",       icon: "·", color: "#6b6b80" },
};

// extractionColor and extractionLabel helpers are available for future enhancements

// Node radius: degree + betweenness-driven for importance
function nodeRadius(degree, betweenness, isCenter) {
  const betweennessScore = Math.sqrt(betweenness || 0) * 8;
  const degreeScore = Math.sqrt(degree + 1) * 3.5;
  const totalScore = degreeScore + betweennessScore;
  
  if (isCenter) return Math.max(18, Math.min(24, totalScore * 2));
  return Math.max(6, Math.min(16, totalScore));
}

const EDGE_COLORS = {
  "co-occurs with":  "#8f98b6",
  "related to":      "#a2adc7",
  "associated with": "#a2adc7",
  "integrates with": "#47bfff",
  "uses":            "#47bfff",
  "is part of":      "#1fc791",
  "depends on":      "#f5a623",
  "outperforms":     "#e05252",
  "causes":          "#ff9f43",
  "is type of":      "#c47aff",
  "based on":        "#8b84ff",
  "derived from":    "#8b84ff",
  "enables":         "#1fc791",
  "contradicts":     "#ff5c5c",
};

function edgeColor(label, method) {
  // Use extraction method color if available for better categorization
  if (method && EXTRACTION_METHODS[method]) {
    return EXTRACTION_METHODS[method].color;
  }
  if (!label) return "#8f98b6";
  const lo = label.toLowerCase();
  for (const [key, col] of Object.entries(EDGE_COLORS)) {
    if (lo.includes(key)) return col;
  }
  return "#a2adc7";
}

function edgeWidth(weight) {
  if (!weight) return 0.8;
  if (weight >= 3) return 2.2;
  if (weight >= 2) return 1.5;
  if (weight >= 1) return 1.0;
  return 0.5;
}

// Convex hull for community clustering visualization
// ── Parse flat + legacy formats from API ──────────────────────────────────────
function parseGraphResponse(rawArray) {
  if (!rawArray?.length) return { nodes: [], links: [], numCommunities: 0 };

  // Prefer flat format (last entry with _flat: true)
  const flatEntry = rawArray.find(e => e._flat);
  if (flatEntry) {
    return {
      nodes: flatEntry.nodes || [],
      links: flatEntry.links || [],
      numCommunities: flatEntry.num_communities || 0,
    };
  }

  // Fallback: reconstruct from legacy document-segmented format
  const nodeMap = new Map();
  const linkMap = new Map();

  rawArray.forEach((seg, segIdx) => {
    if (!seg || seg._flat) return;
    const centerId = seg.center?.id || `Document_${segIdx}`;

    if (!nodeMap.has(centerId)) {
      nodeMap.set(centerId, {
        id: centerId, cluster: "CENTER", community_id: segIdx,
        degree: 0, freq: 1, doc_ids: [], is_center: true,
      });
    }

    for (const [clusterName, nodes] of Object.entries(seg.clusters || {})) {
      for (const node of (nodes || [])) {
        if (!node.id || nodeMap.has(node.id)) continue;
        nodeMap.set(node.id, {
          id: node.id, cluster: clusterName, community_id: segIdx,
          degree: 0, freq: 1, doc_ids: [], is_center: false,
        });
      }
    }

    for (const rel of (seg.relationships || [])) {
      if (!nodeMap.has(rel.source) || !nodeMap.has(rel.target)) continue;
      const key = [rel.source, rel.target].sort().join("|||");
      const existing = linkMap.get(key);
      const weight = rel.weight || 1;
      if (!existing || weight > (existing.weight || 1)) {
        linkMap.set(key, { source: rel.source, target: rel.target, label: rel.label || "", weight });
      }
    }
  });

  // Compute degree
  for (const link of linkMap.values()) {
    const s = nodeMap.get(link.source);
    const t = nodeMap.get(link.target);
    if (s) s.degree++;
    if (t) t.degree++;
  }

  return {
    nodes: [...nodeMap.values()],
    links: [...linkMap.values()],
    numCommunities: rawArray.filter(e => !e._flat).length,
  };
}

// ── Tooltip ───────────────────────────────────────────────────────────────────
function Tooltip({ node, x, y, links }) {
  if (!node) return null;
  const color = communityColor(node.community_id || 0);
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
      background: "var(--bg-surface)", border: `1px solid ${color}55`,
      borderRadius: 12, padding: "14px 16px", minWidth: 210, maxWidth: 290,
      boxShadow: `0 16px 40px rgba(0,0,0,0.5), 0 0 0 1px ${color}20`,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
        <div style={{
          width: 32, height: 32, borderRadius: "50%", flexShrink: 0,
          background: color + "22", border: `1.5px solid ${color}`,
          display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14,
        }}>{c.icon}</div>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color, lineHeight: 1.3 }}>
            {node.id.length > 28 ? node.id.slice(0, 27) + "…" : node.id}
          </div>
          <div style={{ fontSize: 10, color: "var(--text-muted)", letterSpacing: "0.7px" }}>
            {c.label} · community {node.community_id} · degree {node.degree}            {node.betweenness > 0 && ` · centrality ${(node.betweenness * 100).toFixed(1)}%`}          </div>
        </div>
      </div>
      {conns.slice(0, 5).map((l, i) => {
        const s = typeof l.source === "object" ? l.source.id : l.source;
        const t = typeof l.target === "object" ? l.target.id : l.target;
        const other = s === node.id ? t : s;
        const arrow = s === node.id ? "→" : "←";
        return (
          <div key={i} style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 11, color: "var(--text-secondary)", marginBottom: 4 }}>
            <span style={{ color, fontWeight: 700, flexShrink: 0 }}>{arrow}</span>
            <span style={{ background: "var(--bg-raised)", border: "1px solid var(--border)", borderRadius: 3, padding: "0 5px", fontSize: 10, color: "#ccc", flexShrink: 0, maxWidth: 90, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {l.label || "—"}
            </span>
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {other.length > 20 ? other.slice(0, 19) + "…" : other}
            </span>
          </div>
        );
      })}
      {conns.length > 5 && <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>+{conns.length - 5} more</div>}
    </div>
  );
}

// ── Cluster filter legend ─────────────────────────────────────────────────────
function ClusterLegend({ clusters, activeFilters, onToggle, counts }) {
  if (!clusters.length) return null;
  return (
    <div style={{ position: "absolute", top: 14, right: 14, display: "flex", flexDirection: "column", gap: 4, zIndex: 10 }}>
      <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 2, paddingLeft: 2, letterSpacing: "0.5px", textTransform: "uppercase" }}>Filter by type</div>
      {clusters.map(c => {
        const cl = cfg(c);
        const on = activeFilters.has(c);
        return (
          <button key={c} onClick={() => onToggle(c)} style={{
            display: "flex", alignItems: "center", gap: 7,
            background: on ? "var(--bg-raised)" : "var(--bg-surface)",
            border: `1px solid ${on ? "var(--border-focus)" : "var(--border)"}`,
            borderRadius: 999, padding: "4px 10px 4px 7px", fontSize: 11,
            cursor: "pointer", transition: "all 120ms", opacity: on ? 1 : 0.4,
          }}>
            <span style={{ fontSize: 10 }}>{cl.icon}</span>
            <span style={{ color: on ? "var(--text-primary)" : "var(--text-secondary)" }}>{cl.label}</span>
            <span style={{ marginLeft: 2, borderRadius: 999, padding: "1px 5px", background: on ? "var(--accent-dim)" : "var(--bg-raised)", color: on ? "var(--accent-light)" : "var(--text-muted)", fontSize: 10, fontFamily: "var(--font-mono)" }}>
              {counts[c] || 0}
            </span>
          </button>
        );
      })}
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────
export default function KnowledgeGraph({ onDocumentAdded, onDocumentRemoved }) {
  const [rawArray, setRawArray]     = useState([]);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState("");
  const [tooltip, setTooltip]       = useState(null);
  const [searchQ, setSearchQ]       = useState("");
  const [activeFilters, setActiveFilters] = useState(new Set());
  const [edgeMode, setEdgeMode]     = useState("all");
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [focusOnSelection, setFocusOnSelection] = useState(false);

  const svgRef  = useRef(null);
  const wrapRef = useRef(null);
  const simRef  = useRef(null);
  const zoomRef = useRef(null);
  // nodeElRef and linkElRef hold D3 selections so filter/search can update
  // opacity without restarting the simulation
  const nodeElRef = useRef(null);
  const linkElRef = useRef(null);
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
      setRawArray(data.graph);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => {
    let cancelled = false;
    Promise.resolve().then(() => {
      if (!cancelled) fetchGraph();
    });
    return () => { cancelled = true; };
  }, [fetchGraph]);

  // Auto-refresh when a document is uploaded or deleted
  useEffect(() => {
    const unsubs = [];
    if (onDocumentAdded)   unsubs.push(onDocumentAdded(fetchGraph));
    if (onDocumentRemoved) unsubs.push(onDocumentRemoved(fetchGraph));
    return () => unsubs.forEach(u => u?.());
  }, [onDocumentAdded, onDocumentRemoved, fetchGraph]);

  const graphData = useMemo(() => parseGraphResponse(rawArray), [rawArray]);
  const allClusters = useMemo(
    () => [...new Set(graphData.nodes.map(n => n.cluster))].filter(c => c !== "CENTER"),
    [graphData.nodes],
  );
  const effectiveActiveFilters = useMemo(() => {
    if (!activeFilters.size) return new Set(allClusters);
    const next = new Set([...activeFilters].filter(c => allClusters.includes(c)));
    return next.size ? next : new Set(allClusters);
  }, [activeFilters, allClusters]);

  // ── Visible subsets ──────────────────────────────────────────────────────
  const filterWeight = edgeMode === "strong" ? 2.0 : edgeMode === "semantic" ? 2.5 : 0;

  const getVisible = useCallback(() => {
    const baseNodes = graphData.nodes.filter(n => n.is_center || effectiveActiveFilters.has(n.cluster));
    const ids = new Set(baseNodes.map(n => n.id));

    let links = graphData.links.filter(l => {
      const s = typeof l.source === "object" ? l.source.id : l.source;
      const t = typeof l.target === "object" ? l.target.id : l.target;
      return ids.has(s) && ids.has(t) && (l.weight || 1) >= filterWeight;
    });

    const linkedIds = new Set();
    links.forEach(l => {
      const s = typeof l.source === "object" ? l.source.id : l.source;
      const t = typeof l.target === "object" ? l.target.id : l.target;
      linkedIds.add(s);
      linkedIds.add(t);
    });

    const prunedNodes = baseNodes.filter(n => n.is_center || linkedIds.has(n.id));
    const finalNodes = prunedNodes.length ? prunedNodes : baseNodes;

    // Show connections for selected node
    if (focusOnSelection && selectedNodeId) {
      const focusIds = new Set([selectedNodeId]);
      links = links.filter(l => {
        const s = typeof l.source === "object" ? l.source.id : l.source;
        const t = typeof l.target === "object" ? l.target.id : l.target;
        return s === selectedNodeId || t === selectedNodeId;
      });
      links.forEach(l => {
        const s = typeof l.source === "object" ? l.source.id : l.source;
        const t = typeof l.target === "object" ? l.target.id : l.target;
        focusIds.add(s);
        focusIds.add(t);
      });
      return {
        nodes: finalNodes.filter(n => focusIds.has(n.id)),
        links,
      };
    }

    return { nodes: finalNodes, links };
  }, [
    graphData,
    effectiveActiveFilters,
    filterWeight,
    focusOnSelection,
    selectedNodeId,
  ]);

  const getMatchIds = useCallback(() => {
    if (!searchQ.trim()) return new Set();
    const q = searchQ.toLowerCase();
    return new Set(graphData.nodes.filter(n => n.id.toLowerCase().includes(q)).map(n => n.id));
  }, [graphData, searchQ]);

  // ── Update visual opacity without restarting simulation ───────────────────
  // This runs when filters/search change but graphData is the same
  useEffect(() => {
    if (!nodeElRef.current || !linkElRef.current) return;
    const { nodes: visNodes, links: visLinks } = getVisible();
    const visIds = new Set(visNodes.map(n => n.id));
    const matches = getMatchIds();
    const connectedToSelected = new Set();

    if (selectedNodeId) {
      connectedToSelected.add(selectedNodeId);
      visLinks.forEach(l => {
        const s = typeof l.source === "object" ? l.source.id : l.source;
        const t = typeof l.target === "object" ? l.target.id : l.target;
        if (s === selectedNodeId) connectedToSelected.add(t);
        if (t === selectedNodeId) connectedToSelected.add(s);
      });
    }

    nodeElRef.current
      .attr("display", d => visIds.has(d.id) ? null : "none")
      .attr("opacity", d => {
        if (!visIds.has(d.id)) return 0;
        if (selectedNodeId) {
          if (d.id === selectedNodeId) return 1;
          return connectedToSelected.has(d.id) ? 0.92 : 0.08;
        }
        if (matches.size > 0) return (matches.has(d.id) || d.is_center) ? 1 : 0.08;
        return 1;
      });

    linkElRef.current
      .attr("display", l => {
        const s = typeof l.source === "object" ? l.source.id : l.source;
        const t = typeof l.target === "object" ? l.target.id : l.target;
        const w = l.weight || 1;
        return (visIds.has(s) && visIds.has(t) && w >= filterWeight) ? null : "none";
      })
      .attr("stroke-opacity", l => {
        const w = l.weight || 1;
        const base = w >= 3 ? 0.9 : w >= 2 ? 0.75 : w >= 1 ? 0.45 : 0.2;
        if (!selectedNodeId) return base;
        const s = typeof l.source === "object" ? l.source.id : l.source;
        const t = typeof l.target === "object" ? l.target.id : l.target;
        return (s === selectedNodeId || t === selectedNodeId) ? 0.98 : 0.06;
      });
  }, [activeFilters, edgeMode, searchQ, getVisible, getMatchIds, filterWeight, selectedNodeId]);

  // ── Full D3 simulation — only re-runs when graphData or dims change ───────
  useEffect(() => {
    if (!svgRef.current) return;
    const { nodes, links } = getVisible();
    if (!nodes.length) return;
    const { w, h } = dims;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    nodeElRef.current = null;
    linkElRef.current = null;

    const defs = svg.append("defs");

    // Glow filter for high-degree nodes
    const glow = defs.append("filter").attr("id", "glow")
      .attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
    glow.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "blur");
    const merge = glow.append("feMerge");
    merge.append("feMergeNode").attr("in", "blur");
    merge.append("feMergeNode").attr("in", "SourceGraphic");

    // Arrowheads per community color
    const communityIds = [...new Set(nodes.map(n => n.community_id || 0))];
    communityIds.forEach(cid => {
      const color = communityColor(cid);
      defs.append("marker").attr("id", `arr-c${cid}`)
        .attr("viewBox", "0 0 10 10").attr("refX", 26).attr("refY", 5)
        .attr("markerWidth", 4).attr("markerHeight", 4).attr("orient", "auto-start-reverse")
        .append("path").attr("d", "M1 1L9 5L1 9Z").attr("fill", color).attr("opacity", 0.5);
    });

    const g = svg.append("g");
    svg.on("click", () => {
      setSelectedNodeId(null);
      setFocusOnSelection(false);
    });
    const zoom = d3.zoom().scaleExtent([0.05, 10])
      .on("zoom", e => g.attr("transform", e.transform));
    svg.call(zoom);
    zoomRef.current = zoom;

    // Place nodes randomly — let physics find natural structure
    // (no pre-placement bias that creates artificial star topology)
    nodes.forEach(n => {
      n.x = w / 2 + (Math.random() - 0.5) * w * 0.6;
      n.y = h / 2 + (Math.random() - 0.5) * h * 0.6;
    });

    // Compute node radii from degree AND betweenness centrality
    nodes.forEach(n => {
      n._r = nodeRadius(n.degree, n.betweenness || 0, n.is_center);
    });

    // ── Simulation ── no forceCenter (it fights natural clustering) ──────────
    if (simRef.current) simRef.current.stop();

    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id)
        .distance(l => {
          const w = l.weight || 1;
          if (w >= 3) return 55;    // LLM semantic: pull tight
          if (w >= 2) return 75;    // SVO
          if (w >= 1) return 100;   // co-occurrence
          return 200;               // bridge: let float
        })
        .strength(l => {
          const w = l.weight || 1;
          if (w >= 3) return 0.8;
          if (w >= 2) return 0.6;
          if (w >= 1) return 0.35;
          return 0.05;              // bridge: nearly no pull
        })
      )
      // forceManyBody with stronger negative charge — replaces forceCenter
      .force("charge", d3.forceManyBody()
        .strength(d => -(120 + d._r * 15))  // hub nodes repel more (ForceAtlas2-inspired)
        .distanceMax(600)
      )
      .force("collide", d3.forceCollide().radius(d => d._r + 14).strength(0.85))
      // Gentle centering to avoid disconnected islands drifting off-canvas
      .force("x", d3.forceX(w / 2).strength(0.04))
      .force("y", d3.forceY(h / 2).strength(0.04))
      .alphaDecay(0.012)
      .velocityDecay(0.4);

    simRef.current = sim;

    // ── Links ─────────────────────────────────────────────────────────────
    const linkG = g.append("g");
    const nodeById = new Map(nodes.map(n => [n.id, n]));

    const isCrossCommunity = (l) => {
      const s = typeof l.source === "object" ? l.source.id : l.source;
      const t = typeof l.target === "object" ? l.target.id : l.target;
      const sc = nodeById.get(s)?.community_id ?? 0;
      const tc = nodeById.get(t)?.community_id ?? 0;
      return sc !== tc;
    };

    const linkEl = linkG.selectAll("line").data(links).join("line")
      .attr("stroke", l => edgeColor(l.label, l.extraction_method))
      .attr("stroke-width", l => edgeWidth(l.weight))
      .attr("stroke-opacity", l => {
        const w = l.weight || 1;
        const base = w >= 3 ? 0.9 : w >= 2 ? 0.75 : w >= 1 ? 0.45 : 0.2;
        return isCrossCommunity(l) ? Math.min(1, base + 0.15) : base;
      })
      .attr("stroke-dasharray", l => isCrossCommunity(l) ? "5 3" : null)
      .attr("marker-end", l => {
        const t = typeof l.target === "object" ? l.target : nodes.find(n => n.id === l.target);
        const cid = t?.community_id || 0;
        return `url(#arr-c${cid})`;
      });
    linkElRef.current = linkEl;

    // ── Nodes ─────────────────────────────────────────────────────────────
    const nodeG = g.append("g");
    const nodeEl = nodeG.selectAll("g").data(nodes).join("g")
      .attr("cursor", "pointer")
      .call(d3.drag()
        .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag",  (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
        .on("end",   (ev, d) => {
          if (!ev.active) sim.alphaTarget(0);
          if (!d.is_center) { d.fx = null; d.fy = null; }
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
        nodeEl.attr("opacity", n => connIds.has(n.id) ? 1 : 0.08);
        linkEl
          .attr("stroke-opacity", l => {
            const s = typeof l.source === "object" ? l.source.id : l.source;
            const t = typeof l.target === "object" ? l.target.id : l.target;
            return (s === d.id || t === d.id) ? 0.95 : 0.04;
          })
          .attr("stroke-width", l => {
            const s = typeof l.source === "object" ? l.source.id : l.source;
            const t = typeof l.target === "object" ? l.target.id : l.target;
            return (s === d.id || t === d.id) ? edgeWidth(l.weight) * 2.5 : 0.3;
          });
      })
      .on("mousemove", ev => setTooltip(p => p ? { ...p, x: ev.clientX, y: ev.clientY } : p))
      .on("click", (ev, d) => {
        ev.stopPropagation();
        setSelectedNodeId(prev => (prev === d.id ? null : d.id));
      })
      .on("mouseout", () => {
        setTooltip(null);
        nodeEl.attr("opacity", 1);
        linkEl
          .attr("stroke-opacity", l => {
            const w = l.weight || 1;
            return w >= 3 ? 0.9 : w >= 2 ? 0.75 : w >= 1 ? 0.45 : 0.2;
          })
          .attr("stroke-width", l => edgeWidth(l.weight));
      });
    nodeElRef.current = nodeEl;

    // Pulse ring for hub nodes (degree ≥ 5)
    nodeEl.filter(d => d.degree >= 5 || d.is_center).append("circle")
      .attr("r", d => d._r + 9)
      .attr("fill", "none")
      .attr("stroke", d => communityColor(d.community_id || 0) + "25")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "5 3");

    // Main circle — colored by COMMUNITY
    nodeEl.append("circle")
      .attr("r", d => d._r)
      .attr("fill", d => communityColor(d.community_id || 0) + "22")
      .attr("stroke", d => communityColor(d.community_id || 0))
      .attr("stroke-width", d => d.is_center ? 2.2 : 1.3)
      .attr("filter", d => d.degree >= 8 ? "url(#glow)" : null);

    // Betweenness centrality badge (★) for hub connectors
    nodeEl.filter(d => (d.betweenness || 0) > 0.05).append("text")
      .text("★")
      .attr("text-anchor", "middle").attr("dominant-baseline", "central")
      .attr("font-size", d => d.is_center ? 10 : 6)
      .attr("fill", "#ffd32a")
      .attr("pointer-events", "none")
      .attr("x", d => d._r + 3)
      .attr("y", d => -(d._r + 3));

    // Icon (cluster-type shape)
    nodeEl.append("text")
      .text(d => cfg(d.cluster).icon)
      .attr("text-anchor", "middle").attr("dominant-baseline", "central")
      .attr("font-size", d => d.is_center ? 12 : 8)
      .attr("fill", d => communityColor(d.community_id || 0))
      .attr("pointer-events", "none");

    // Label
    nodeEl.append("text")
      .text(d => {
        const max = d.is_center ? 20 : 15;
        return d.id.length > max ? d.id.slice(0, max - 1) + "…" : d.id;
      })
      .attr("text-anchor", "middle").attr("dominant-baseline", "hanging")
      .attr("y", d => d._r + 4)
      .attr("font-size", d => d.is_center ? 11 : 8)
      .attr("font-weight", d => d.is_center || d.degree >= 5 ? "600" : "400")
      .attr("fill", d => communityColor(d.community_id || 0))
      .attr("pointer-events", "none");

    sim.on("tick", () => {
      linkEl
        .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
      nodeEl.attr("transform", d => `translate(${d.x||0},${d.y||0})`);
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
  }, [graphData, dims, getVisible]);

  const handleFit = () => {
    if (!svgRef.current || !zoomRef.current) return;
    const svg = d3.select(svgRef.current);
    const gEl = svg.select("g");
    if (gEl.empty()) return;
    const b = gEl.node().getBBox();
    if (!b.width) return;
    const { w, h } = dims;
    const scale = Math.min(0.88, Math.min(w / (b.width + 100), h / (b.height + 100)));
    const tx = (w - b.width * scale) / 2 - b.x * scale;
    const ty = (h - b.height * scale) / 2 - b.y * scale;
    svg.transition().duration(500).call(zoomRef.current.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
  };

  const { nodes: visNodes, links: visLinks } = getVisible();
  const clusterCounts = Object.fromEntries(allClusters.map(c => [c, graphData.nodes.filter(n => n.cluster === c).length]));
  const matches = getMatchIds();

  const llmEdges  = graphData.links.filter(l => (l.weight||0) >= 3).length;
  const svoEdges  = graphData.links.filter(l => (l.weight||0) >= 2 && (l.weight||0) < 3).length;
  const coocEdges = graphData.links.filter(l => (l.weight||0) >= 1 && (l.weight||0) < 2).length;
  const bridgeEdges = graphData.links.filter(l => (l.weight||0) < 1).length;

  const toggleFilter = (c) => {
    setActiveFilters(prev => {
      const next = prev.size ? new Set(prev) : new Set(allClusters);
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
            {graphData.numCommunities > 1 ? `${graphData.numCommunities} communities · ` : ""}
            {graphData.nodes.length} nodes · {graphData.links.length} edges
            {llmEdges > 0 && ` (${llmEdges} semantic, ${svoEdges} SVO, ${coocEdges} co-occ)`}
          </div>
        </div>

        {/* Search */}
        <div style={{ position: "relative" }}>
          <i className="ti ti-search" style={{ position: "absolute", left: 9, top: "50%", transform: "translateY(-50%)", fontSize: 12, color: "var(--text-muted)", pointerEvents: "none" }} />
          <input type="text" value={searchQ} onChange={e => setSearchQ(e.target.value)}
            placeholder="Find node…"
            style={{ background: "var(--bg-raised)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", color: "var(--text-primary)", fontFamily: "var(--font)", fontSize: 12, padding: "6px 26px 6px 28px", outline: "none", width: 140 }} />
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
            <button key={m.id} onClick={() => setEdgeMode(m.id)} className="btn"
              style={{ padding: "5px 10px", fontSize: 11,
                background: edgeMode === m.id ? "var(--accent-dim)" : "var(--bg-raised)",
                borderColor: edgeMode === m.id ? "var(--accent-dim2)" : "var(--border)",
                color: edgeMode === m.id ? "var(--accent-light)" : "var(--text-muted)" }}>
              {m.label}
            </button>
          ))}
        </div>

        {selectedNodeId && (
          <button className="btn" onClick={() => setFocusOnSelection(v => !v)}
            style={{ padding: "5px 10px", fontSize: 11,
              background: focusOnSelection ? "var(--amber-dim)" : "var(--bg-raised)",
              borderColor: focusOnSelection ? "var(--amber)" : "var(--border)",
              color: focusOnSelection ? "var(--amber)" : "var(--text-muted)" }}>
            Focus selection
          </button>
        )}

        <div className="graph-stats">
          <span className="graph-stat"><strong>{visNodes.length}</strong> nodes</span>
          <span className="graph-stat"><strong>{visLinks.length}</strong> edges</span>
        </div>

        <button className="btn" onClick={handleFit} disabled={!visNodes.length}>
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
            <h3>Failed to load graph</h3><p>{error}</p>
            <button className="btn" onClick={fetchGraph} style={{ marginTop: 8 }}>Retry</button>
          </div>
        )}

        {!error && !graphData.nodes.length && !loading && (
          <div className="empty-state" style={{ position: "absolute", inset: 0 }}>
            <i className="ti ti-share-2" />
            <h3>No graph data yet</h3>
            <p>Upload documents — entities connect via semantic relationships and are colored by community membership, not cluster type.</p>
          </div>
        )}

        {graphData.nodes.length > 0 && (
          <>
            <svg ref={svgRef} width={dims.w} height={dims.h} style={{ display: "block", background: "transparent" }} />

            <ClusterLegend clusters={allClusters} activeFilters={effectiveActiveFilters} onToggle={toggleFilter} counts={clusterCounts} />

            {tooltip && <Tooltip node={tooltip.node} x={tooltip.x} y={tooltip.y} links={graphData.links} />}

            {/* Community legend */}
            {graphData.numCommunities > 1 && (
              <div style={{ position: "absolute", top: 14, left: 14, background: "var(--bg-surface)", border: "1px solid var(--border)", borderRadius: 10, padding: "8px 14px", zIndex: 10, maxWidth: 180 }}>
                <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.6px", marginBottom: 6 }}>Communities</div>
                {Array.from({ length: Math.min(graphData.numCommunities, 8) }, (_, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                    <div style={{ width: 8, height: 8, borderRadius: "50%", background: communityColor(i), flexShrink: 0 }} />
                    <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>Community {i}</span>
                  </div>
                ))}
                {graphData.numCommunities > 8 && <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>+{graphData.numCommunities - 8} more</div>}
              </div>
            )}

            {/* Edge type legend */}
            <div style={{ position: "absolute", bottom: 12, left: 14, background: "var(--bg-surface)", border: "1px solid var(--border)", borderRadius: 10, padding: "8px 14px", display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap", zIndex: 10, maxWidth: "calc(100% - 40px)" }}>
              {[
                { label: "LLM semantic",  color: "#8b84ff", w: 2.2, count: llmEdges },
                { label: "SVO verb",      color: "#1fc791", w: 1.5, count: svoEdges },
                { label: "Co-occurrence", color: "#ffffff", w: 1.0, count: coocEdges },
                { label: "Bridge",        color: "#6b6b80", w: 0.5, count: bridgeEdges, dash: true },
              ].filter(e => e.count > 0).map((e, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <svg width="22" height="10">
                    <line x1="0" y1="5" x2="22" y2="5" stroke={e.color} strokeWidth={e.w} strokeOpacity={0.7} strokeDasharray={e.dash ? "3 3" : "none"} />
                  </svg>
                  <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
                    {e.label} <span style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>({e.count})</span>
                  </span>
                </div>
              ))}
              <span style={{ fontSize: 10, color: "var(--text-muted)", marginLeft: 4 }}>
                Node size = degree+centrality · ★ = hub connector · Color = community
              </span>
            </div>
          </>
        )}
      </div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}