import { useEffect, useState, useCallback, useRef } from "react";
import * as d3 from "d3";

const API = "http://127.0.0.1:8000";

const CLUSTER_CONFIG = {
  CENTER:   { color: "#8b84ff", bg: "#8b84ff18", icon: "◉", label: "Center", size: 22 },
  ENTITY:   { color: "#8b84ff", bg: "#8b84ff14", icon: "◉", label: "Entities", size: 12 },
  CONCEPT:  { color: "#1fc791", bg: "#1fc79114", icon: "◈", label: "Concepts", size: 12 },
  LOCATION: { color: "#e05252", bg: "#e0525214", icon: "◎", label: "Locations", size: 12 },
  EVENT:    { color: "#ff9f43", bg: "#ff9f4314", icon: "◆", label: "Events", size: 11 },
  DATE:     { color: "#00d2d3", bg: "#00d2d314", icon: "◇", label: "Dates", size: 10 },
  ACTION:   { color: "#c47aff", bg: "#c47aff14", icon: "▶", label: "Actions", size: 11 },
  QUANTITY: { color: "#54a0ff", bg: "#54a0ff14", icon: "▣", label: "Quantities", size: 10 },
  RELATION: { color: "#f5a623", bg: "#f5a62314", icon: "⟷", label: "Relations", size: 11 },
  TECH:     { color: "#47bfff", bg: "#47bfff14", icon: "⬡", label: "Technologies", size: 13 },
  MISC:     { color: "#6b6b80", bg: "#6b6b8014", icon: "·",  label: "Other", size: 9 },
  SKILL:    { color: "#1fc791", bg: "#1fc79114", icon: "⚡", label: "Skills", size: 11 },
  COMPANY:  { color: "#f5a623", bg: "#f5a62314", icon: "⬜", label: "Companies", size: 12 },
  EDUCATION:{ color: "#47bfff", bg: "#47bfff14", icon: "▲", label: "Education", size: 11 },
  ROLE:     { color: "#c47aff", bg: "#c47aff14", icon: "⬟", label: "Roles", size: 11 },
  PROJECT:  { color: "#ff9f43", bg: "#ff9f4314", icon: "◫", label: "Projects", size: 12 },
  CONTACT:  { color: "#54a0ff", bg: "#54a0ff14", icon: "✉",  label: "Contact", size: 10 },
};

const cfg = (cluster) => CLUSTER_CONFIG[cluster] || CLUSTER_CONFIG.MISC;

// ── Build multi-document graph with per-doc subgraphs ───────────────────────
function buildGraph(rawArray) {
  if (!rawArray?.length) return { nodes: [], links: [] };

  const nodeMap = new Map();
  const links = [];
  const seenLinks = new Set();

  // Multiple documents = multiple center nodes, each with their own subgraph
  const isMultiDoc = rawArray.length > 1;

  rawArray.forEach((seg, segIdx) => {
    if (!seg) return;
    const centerId = seg.center?.id || `Document ${segIdx + 1}`;

    // Add center
    if (!nodeMap.has(centerId)) {
      nodeMap.set(centerId, {
        id: centerId,
        cluster: "CENTER",
        r: isMultiDoc ? 18 : 22,
        segIdx,
        docCenter: true,
      });
    }

    // Add cluster nodes
    if (seg.clusters) {
      for (const [clusterName, nodes] of Object.entries(seg.clusters)) {
        for (const node of (nodes || [])) {
          const key = node.id;
          if (!nodeMap.has(key)) {
            const clCfg = cfg(clusterName);
            nodeMap.set(key, {
              id: key,
              cluster: clusterName,
              r: clCfg.size || 10,
              segIdx,
            });
          }
        }
      }
    }

    // Add relationships
    if (seg.relationships) {
      for (const rel of seg.relationships) {
        const lKey = `${rel.source}|${rel.target}`;
        if (seenLinks.has(lKey)) continue;
        if (!nodeMap.has(rel.source) || !nodeMap.has(rel.target)) continue;
        seenLinks.add(lKey);
        links.push({ source: rel.source, target: rel.target, label: rel.label || "→", segIdx });
      }
    }
  });

  return { nodes: [...nodeMap.values()], links };
}

// ── Tooltip ──────────────────────────────────────────────────────────────────
function Tooltip({ node, x, y, links }) {
  if (!node) return null;
  const c = cfg(node.cluster);
  const conns = links.filter(l => {
    const s = typeof l.source === "object" ? l.source.id : l.source;
    const t = typeof l.target === "object" ? l.target.id : l.target;
    return s === node.id || t === node.id;
  });
  const left = Math.min(x + 14, window.innerWidth - 300);
  const top = Math.max(y - 10, 8);

  return (
    <div style={{
      position: "fixed", left, top,
      background: "var(--bg-surface)", border: `1px solid ${c.color}55`,
      borderRadius: 12, padding: "14px 16px",
      minWidth: 200, maxWidth: 280, zIndex: 200, pointerEvents: "none",
      boxShadow: `0 16px 40px rgba(0,0,0,0.5), 0 0 0 1px ${c.color}20`,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 9 }}>
        <div style={{
          width: 32, height: 32, borderRadius: "50%", flexShrink: 0,
          background: c.bg, border: `1.5px solid ${c.color}`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 14,
        }}>
          {c.icon}
        </div>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: c.color, lineHeight: 1.3 }}>
            {node.id.length > 28 ? node.id.slice(0, 27) + "…" : node.id}
          </div>
          <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.7px" }}>
            {c.label} · {conns.length} link{conns.length !== 1 ? "s" : ""}
          </div>
        </div>
      </div>
      {conns.slice(0, 4).map((l, i) => {
        const s = typeof l.source === "object" ? l.source.id : l.source;
        const t = typeof l.target === "object" ? l.target.id : l.target;
        const other = s === node.id ? t : s;
        const arrow = s === node.id ? "→" : "←";
        return (
          <div key={i} style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 11, color: "var(--text-secondary)", marginBottom: 3 }}>
            <span style={{ color: c.color, fontWeight: 700 }}>{arrow}</span>
            <span style={{ background: "var(--bg-raised)", border: "1px solid var(--border)", borderRadius: 3, padding: "0 4px", fontSize: 10, color: "var(--text-muted)", flexShrink: 0 }}>{l.label}</span>
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{other.length > 20 ? other.slice(0, 19) + "…" : other}</span>
          </div>
        );
      })}
      {conns.length > 4 && <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 3 }}>+{conns.length - 4} more</div>}
    </div>
  );
}

// ── Legend ───────────────────────────────────────────────────────────────────
function Legend({ clusters, activeFilters, onToggle, counts }) {
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
  const [layoutMode, setLayoutMode] = useState("force"); // force | radial | cluster

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
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { fetchGraph(); }, [fetchGraph]);

  useEffect(() => {
    const g = buildGraph(rawGraph);
    setGraphData(g);
    const clusters = [...new Set(g.nodes.map(n => n.cluster))].filter(c => c !== "CENTER");
    setActiveFilters(new Set(clusters));
  }, [rawGraph]);

  // Visible subgraph
  const visible = useCallback(() => {
    let nodes = graphData.nodes.filter(n => n.cluster === "CENTER" || activeFilters.has(n.cluster));
    const ids = new Set(nodes.map(n => n.id));
    const links = graphData.links.filter(l => {
      const s = typeof l.source === "object" ? l.source.id : l.source;
      const t = typeof l.target === "object" ? l.target.id : l.target;
      return ids.has(s) && ids.has(t);
    });
    return { nodes, links };
  }, [graphData, activeFilters]);

  const matchIds = useCallback(() => {
    if (!searchQ.trim()) return new Set();
    const q = searchQ.toLowerCase();
    return new Set(graphData.nodes.filter(n => n.id.toLowerCase().includes(q)).map(n => n.id));
  }, [graphData, searchQ]);

  // ── D3 render ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!svgRef.current) return;
    const { nodes, links } = visible();
    if (nodes.length === 0) return;
    const { w, h } = dims;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const matches = matchIds();
    const centerNodes = nodes.filter(n => n.cluster === "CENTER");
    const isMultiDoc = centerNodes.length > 1;

    // Defs: arrow markers per cluster + glow filter
    const defs = svg.append("defs");

    // Glow filter
    const filter = defs.append("filter").attr("id", "glow").attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
    filter.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "blur");
    const merge = filter.append("feMerge");
    merge.append("feMergeNode").attr("in", "blur");
    merge.append("feMergeNode").attr("in", "SourceGraphic");

    // Arrow markers
    const allClusters = [...new Set(nodes.map(n => n.cluster))];
    allClusters.forEach(cluster => {
      const color = cfg(cluster).color;
      ["arr-" + cluster, "arr-hi-" + cluster].forEach((id, hi) => {
        defs.append("marker").attr("id", id)
          .attr("viewBox", "0 0 10 10").attr("refX", 28).attr("refY", 5)
          .attr("markerWidth", hi ? 6 : 4).attr("markerHeight", hi ? 6 : 4)
          .attr("orient", "auto-start-reverse")
          .append("path").attr("d", "M1 1L9 5L1 9Z")
          .attr("fill", color).attr("opacity", hi ? 1 : 0.5);
      });
    });

    const g = svg.append("g");
    const zoom = d3.zoom().scaleExtent([0.08, 8]).on("zoom", e => g.attr("transform", e.transform));
    svg.call(zoom);
    zoomRef.current = zoom;

    // ── Initial positions ──────────────────────────────────────────────────
    if (isMultiDoc) {
      // Arrange center nodes in a circle, their children nearby
      const angle = (2 * Math.PI) / centerNodes.length;
      const orbit = Math.min(w, h) * 0.28;
      centerNodes.forEach((cn, i) => {
        cn.x = w / 2 + Math.cos(angle * i - Math.PI / 2) * orbit;
        cn.y = h / 2 + Math.sin(angle * i - Math.PI / 2) * orbit;
        cn.fx = cn.x; cn.fy = cn.y; // pin initially, release after warmup
      });
      // Scatter children near their center
      nodes.filter(n => n.cluster !== "CENTER").forEach(n => {
        const center = centerNodes.find(c => c.segIdx === n.segIdx) || centerNodes[0];
        const a = Math.random() * Math.PI * 2;
        const d = 60 + Math.random() * 100;
        n.x = (center.x || w / 2) + Math.cos(a) * d;
        n.y = (center.y || h / 2) + Math.sin(a) * d;
      });
    } else {
      const center = centerNodes[0];
      if (center) { center.x = w / 2; center.y = h / 2; center.fx = w / 2; center.fy = h / 2; }
      // Radial initial placement by cluster
      const clusterGroups = {};
      nodes.filter(n => n.cluster !== "CENTER").forEach(n => {
        (clusterGroups[n.cluster] = clusterGroups[n.cluster] || []).push(n);
      });
      const clKeys = Object.keys(clusterGroups);
      const sectorAngle = (2 * Math.PI) / Math.max(clKeys.length, 1);
      clKeys.forEach((cl, ci) => {
        const baseAngle = ci * sectorAngle - Math.PI / 2;
        clusterGroups[cl].forEach((n, ni) => {
          const spread = (ni - (clusterGroups[cl].length - 1) / 2) * 0.35;
          const a = baseAngle + spread;
          const d = 100 + (ni % 3) * 55;
          n.x = w / 2 + Math.cos(a) * d;
          n.y = h / 2 + Math.sin(a) * d;
        });
      });
    }

    // ── Simulation ─────────────────────────────────────────────────────────
    if (simRef.current) simRef.current.stop();

    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id)
        .distance(d => {
          const t = typeof d.target === "object" ? d.target : nodes.find(n => n.id === d.target);
          const s = typeof d.source === "object" ? d.source : nodes.find(n => n.id === d.source);
          if (t?.cluster === "CENTER" || s?.cluster === "CENTER") return isMultiDoc ? 90 : 110;
          return 60;
        })
        .strength(d => {
          const t = typeof d.target === "object" ? d.target : nodes.find(n => n.id === d.target);
          return t?.cluster === "CENTER" ? 0.7 : 0.45;
        })
      )
      .force("charge", d3.forceManyBody()
        .strength(d => d.cluster === "CENTER" ? (isMultiDoc ? -600 : -900) : -180)
        .distanceMax(400)
      )
      .force("collide", d3.forceCollide().radius(d => d.r + (d.cluster === "CENTER" ? 28 : 16)).strength(0.85))
      .force("center", isMultiDoc ? null : d3.forceCenter(w / 2, h / 2).strength(0.04))
      .alphaDecay(0.018).velocityDecay(0.35);

    if (isMultiDoc) {
      // Release center pins after warmup to let them spread naturally
      setTimeout(() => {
        centerNodes.forEach(cn => { cn.fx = null; cn.fy = null; });
        sim.alpha(0.3).restart();
      }, 600);
    }

    simRef.current = sim;

    // ── Draw links ─────────────────────────────────────────────────────────
    const linkG = g.append("g").attr("class", "links");
    const linkEl = linkG.selectAll("line").data(links).join("line")
      .attr("stroke", d => {
        const t = typeof d.target === "object" ? d.target : nodes.find(n => n.id === d.target);
        return cfg(t?.cluster || "MISC").color;
      })
      .attr("stroke-width", d => {
        const t = typeof d.target === "object" ? d.target : nodes.find(n => n.id === d.target);
        return t?.cluster === "CENTER" ? 1.6 : 0.8;
      })
      .attr("stroke-opacity", d => {
        const t = typeof d.target === "object" ? d.target : nodes.find(n => n.id === d.target);
        return t?.cluster === "CENTER" ? 0.55 : 0.3;
      })
      .attr("marker-end", d => {
        const t = typeof d.target === "object" ? d.target : nodes.find(n => n.id === d.target);
        return `url(#arr-${t?.cluster || "MISC"})`;
      });

    // Link labels (only for center→child links)
    const linkLabelEl = linkG.selectAll("text").data(links.filter(l => {
      const t = typeof l.target === "object" ? l.target : nodes.find(n => n.id === l.target);
      return t?.cluster === "CENTER";
    })).join("text")
      .attr("text-anchor", "middle")
      .attr("font-size", 8)
      .attr("fill", "var(--text-muted)")
      .attr("pointer-events", "none")
      .text(d => d.label?.slice(0, 12) || "");

    // ── Draw nodes ─────────────────────────────────────────────────────────
    const nodeG = g.append("g").attr("class", "nodes");
    const nodeEl = nodeG.selectAll("g").data(nodes).join("g")
      .attr("cursor", "pointer")
      .call(
        d3.drag()
          .on("start", (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
          .on("end", (ev, d) => { if (!ev.active) sim.alphaTarget(0); if (d.cluster !== "CENTER") { d.fx = null; d.fy = null; } })
      )
      .on("mouseover", (ev, d) => {
        setTooltip({ node: d, x: ev.clientX, y: ev.clientY });
        // Highlight connected
        const connectedIds = new Set([d.id]);
        links.forEach(l => {
          const s = typeof l.source === "object" ? l.source.id : l.source;
          const t = typeof l.target === "object" ? l.target.id : l.target;
          if (s === d.id) connectedIds.add(t);
          if (t === d.id) connectedIds.add(s);
        });
        nodeEl.attr("opacity", n => connectedIds.has(n.id) ? 1 : 0.12);
        linkEl
          .attr("stroke-opacity", l => {
            const s = typeof l.source === "object" ? l.source.id : l.source;
            const t = typeof l.target === "object" ? l.target.id : l.target;
            return (s === d.id || t === d.id) ? 0.95 : 0.04;
          })
          .attr("stroke-width", l => {
            const s = typeof l.source === "object" ? l.source.id : l.source;
            const t = typeof l.target === "object" ? l.target.id : l.target;
            return (s === d.id || t === d.id) ? 2.5 : 0.5;
          });
      })
      .on("mousemove", ev => setTooltip(p => p ? { ...p, x: ev.clientX, y: ev.clientY } : p))
      .on("mouseout", () => {
        setTooltip(null);
        nodeEl.attr("opacity", 1);
        linkEl
          .attr("stroke-opacity", d => {
            const t = typeof d.target === "object" ? d.target : nodes.find(n => n.id === d.target);
            return t?.cluster === "CENTER" ? 0.55 : 0.3;
          })
          .attr("stroke-width", d => {
            const t = typeof d.target === "object" ? d.target : nodes.find(n => n.id === d.target);
            return t?.cluster === "CENTER" ? 1.6 : 0.8;
          });
      });

    // Search dimming
    if (matches.size > 0) nodeEl.attr("opacity", d => matches.has(d.id) || d.cluster === "CENTER" ? 1 : 0.1);

    // Pulse ring (center nodes)
    nodeEl.filter(d => d.cluster === "CENTER").append("circle")
      .attr("r", d => d.r + 14)
      .attr("fill", d => cfg(d.cluster).color + "08")
      .attr("stroke", d => cfg(d.cluster).color + "25")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "5 3");

    // Second pulse ring (larger, for multi-doc centers)
    if (isMultiDoc) {
      nodeEl.filter(d => d.cluster === "CENTER").append("circle")
        .attr("r", d => d.r + 26)
        .attr("fill", "none")
        .attr("stroke", d => cfg(d.cluster).color + "10")
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "2 8");
    }

    // Search ring
    if (matches.size > 0) {
      nodeEl.filter(d => matches.has(d.id)).append("circle")
        .attr("r", d => d.r + 8)
        .attr("fill", "none")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.5)
        .attr("stroke-dasharray", "3 3");
    }

    // Main circle
    nodeEl.append("circle")
      .attr("r", d => d.r)
      .attr("fill", d => cfg(d.cluster).bg)
      .attr("stroke", d => cfg(d.cluster).color)
      .attr("stroke-width", d => d.cluster === "CENTER" ? 2.2 : 1.4)
      .attr("filter", d => d.cluster === "CENTER" ? "url(#glow)" : null);

    // Icon
    nodeEl.append("text")
      .text(d => cfg(d.cluster).icon)
      .attr("text-anchor", "middle").attr("dominant-baseline", "central")
      .attr("font-size", d => d.cluster === "CENTER" ? 13 : 8)
      .attr("pointer-events", "none");

    // Label below node
    nodeEl.append("text")
      .text(d => {
        const l = d.id;
        const max = d.cluster === "CENTER" ? 22 : 18;
        return l.length > max ? l.slice(0, max - 1) + "…" : l;
      })
      .attr("text-anchor", "middle").attr("dominant-baseline", "hanging")
      .attr("y", d => d.r + 5)
      .attr("font-size", d => d.cluster === "CENTER" ? 12 : 9)
      .attr("font-weight", d => d.cluster === "CENTER" ? "600" : "400")
      .attr("fill", d => cfg(d.cluster).color)
      .attr("font-family", "var(--font, system-ui)")
      .attr("pointer-events", "none");

    // Label above center node
    nodeEl.filter(d => d.cluster === "CENTER").append("text")
      .text(d => d.id.length > 26 ? d.id.slice(0, 25) + "…" : d.id)
      .attr("text-anchor", "middle").attr("dominant-baseline", "auto")
      .attr("y", d => -(d.r + 10))
      .attr("font-size", 13).attr("font-weight", "700")
      .attr("fill", d => cfg(d.cluster).color)
      .attr("font-family", "var(--font, system-ui)")
      .attr("pointer-events", "none");

    // Tick
    sim.on("tick", () => {
      linkEl
        .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
      linkLabelEl
        .attr("x", d => ((d.source.x || 0) + (d.target.x || 0)) / 2)
        .attr("y", d => ((d.source.y || 0) + (d.target.y || 0)) / 2);
      nodeEl.attr("transform", d => `translate(${d.x || 0},${d.y || 0})`);
    });

    sim.on("end", () => {
      const b = g.node().getBBox();
      if (!b.width || !b.height) return;
      const scale = Math.min(0.9, Math.min(w / (b.width + 120), h / (b.height + 120)));
      const tx = (w - b.width * scale) / 2 - b.x * scale;
      const ty = (h - b.height * scale) / 2 - b.y * scale;
      svg.transition().duration(800).call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
    });

    return () => { simRef.current?.stop(); };
  }, [graphData, dims, activeFilters, searchQ]);

  const handleFit = () => {
    if (!svgRef.current || !zoomRef.current) return;
    const svg = d3.select(svgRef.current);
    const gEl = svg.select("g");
    if (gEl.empty()) return;
    const { w, h } = dims;
    const b = gEl.node().getBBox();
    if (!b.width) return;
    const scale = Math.min(0.9, Math.min(w / (b.width + 120), h / (b.height + 120)));
    const tx = (w - b.width * scale) / 2 - b.x * scale;
    const ty = (h - b.height * scale) / 2 - b.y * scale;
    svg.transition().duration(500).call(zoomRef.current.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
  };

  const { nodes, links } = visible();
  const allClusters = [...new Set(graphData.nodes.map(n => n.cluster))].filter(c => c !== "CENTER");
  const clusterCounts = Object.fromEntries(allClusters.map(c => [c, graphData.nodes.filter(n => n.cluster === c).length]));
  const matches = matchIds();
  const docCount = rawGraph.length;

  const toggleFilter = (c) => {
    setActiveFilters(prev => {
      const next = new Set(prev);
      if (next.has(c)) { if (next.size > 1) next.delete(c); } else next.add(c);
      return next;
    });
  };

  return (
    <div className="graph-wrap">
      <div className="graph-toolbar">
        <div>
          <div className="page-title">Knowledge Graph</div>
          <div className="page-subtitle">
            {docCount > 1 ? `${docCount} documents · multi-hub network` : "Semantic entity map"} — hover to explore
          </div>
        </div>

        {/* Search */}
        <div style={{ position: "relative" }}>
          <i className="ti ti-search" style={{ position: "absolute", left: 9, top: "50%", transform: "translateY(-50%)", fontSize: 12, color: "var(--text-muted)", pointerEvents: "none" }} />
          <input
            type="text"
            value={searchQ}
            onChange={e => setSearchQ(e.target.value)}
            placeholder="Find node…"
            style={{ background: "var(--bg-raised)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", color: "var(--text-primary)", fontFamily: "var(--font)", fontSize: 12, padding: "6px 28px 6px 28px", outline: "none", width: 150 }}
          />
          {searchQ && (
            <button onClick={() => setSearchQ("")} style={{ position: "absolute", right: 7, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", fontSize: 12, display: "flex", padding: 0 }}>
              <i className="ti ti-x" />
            </button>
          )}
        </div>
        {searchQ && <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{matches.size}/{graphData.nodes.length}</span>}

        <div className="graph-stats">
          <span className="graph-stat"><strong>{nodes.length}</strong> nodes</span>
          <span className="graph-stat"><strong>{links.length}</strong> edges</span>
          {docCount > 1 && <span className="graph-stat"><strong>{docCount}</strong> docs</span>}
        </div>

        <button className="btn" onClick={handleFit} disabled={!nodes.length} title="Zoom to fit">
          <i className="ti ti-focus-2" /> Fit
        </button>
        <button className="btn" onClick={fetchGraph} disabled={loading}>
          <i className={`ti ${loading ? "ti-loader-2" : "ti-refresh"}`} style={loading ? { animation: "spin 1s linear infinite" } : {}} />
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
            <p>Upload documents — entities, concepts, technologies and their connections will appear here as an interactive knowledge network.</p>
          </div>
        )}

        {graphData.nodes.length > 0 && (
          <>
            <svg ref={svgRef} width={dims.w} height={dims.h} style={{ display: "block", background: "transparent" }} />

            <Legend clusters={allClusters} activeFilters={activeFilters} onToggle={toggleFilter} counts={clusterCounts} />

            {tooltip && <Tooltip node={tooltip.node} x={tooltip.x} y={tooltip.y} links={graphData.links} />}

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

            {/* Bottom legend */}
            <div style={{
              position: "absolute", bottom: 12, left: 14,
              background: "var(--bg-surface)", border: "1px solid var(--border)",
              borderRadius: 10, padding: "7px 12px",
              display: "flex", flexWrap: "wrap", gap: "5px 12px",
              maxWidth: 480, zIndex: 10,
            }}>
              {allClusters.filter(c => activeFilters.has(c)).map(c => {
                const cl = cfg(c);
                return (
                  <div key={c} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                    <div style={{ width: 6, height: 6, borderRadius: "50%", background: cl.color }} />
                    <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{cl.label}</span>
                  </div>
                );
              })}
              <span style={{ fontSize: 10, color: "var(--text-muted)", marginLeft: "auto", alignSelf: "center" }}>
                Drag · Scroll to zoom
              </span>
            </div>
          </>
        )}
      </div>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}