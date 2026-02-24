import { useRef, useEffect, useState, useCallback } from 'react';
import type { ClusterScatterPoint } from '../../lib/api';

const TOP_COLORS = [
  '#6366f1', '#34d399', '#f87171', '#fbbf24', '#38bdf8',
  '#a78bfa', '#fb923c', '#e879f9', '#2dd4bf', '#f472b6',
  '#818cf8', '#4ade80', '#fb7185', '#facc15', '#22d3ee',
  '#c084fc', '#fdba74', '#f0abfc', '#5eead4', '#ec4899',
];
const OTHER_COLOR = '#3b3f51';

interface Props {
  data: ClusterScatterPoint[];
}

export default function ClusterScatter({ data }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number; y: number; point: ClusterScatterPoint; clusterCount: number;
  } | null>(null);
  const [size, setSize] = useState({ w: 800, h: 500 });

  // Compute cluster ranking + counts
  const clusterMeta = useRef<{
    ranked: number[]; counts: Map<number, number>; colorMap: Map<number, string>;
    keywordsMap: Map<number, string>;
  }>({ ranked: [], counts: new Map(), colorMap: new Map(), keywordsMap: new Map() });

  useEffect(() => {
    const counts = new Map<number, number>();
    const kwMap = new Map<number, string>();
    for (const p of data) {
      counts.set(p.cluster_id, (counts.get(p.cluster_id) || 0) + 1);
      if (!kwMap.has(p.cluster_id) && p.keywords) kwMap.set(p.cluster_id, p.keywords);
    }
    const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(e => e[0]);
    const colorMap = new Map<number, string>();
    ranked.forEach((cid, i) => {
      colorMap.set(cid, i < TOP_COLORS.length ? TOP_COLORS[i] : OTHER_COLOR);
    });
    clusterMeta.current = { ranked, counts, colorMap, keywordsMap: kwMap };
  }, [data]);

  // Resize observer
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      const { width } = entries[0].contentRect;
      if (width > 0) setSize({ w: width, h: Math.min(560, Math.max(400, width * 0.55)) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Precompute projected coords
  const projected = useRef<{ px: number; py: number; idx: number }[]>([]);

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.w * dpr;
    canvas.height = size.h * dpr;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, size.w, size.h);

    const pad = { top: 20, right: 20, bottom: 20, left: 20 };
    const plotW = size.w - pad.left - pad.right;
    const plotH = size.h - pad.top - pad.bottom;

    // Compute bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of data) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    const proj: { px: number; py: number; idx: number }[] = [];
    const { colorMap } = clusterMeta.current;

    // Sort so "other" (grey) points draw first, colored on top
    const indices = data.map((_, i) => i);
    indices.sort((a, b) => {
      const ca = colorMap.get(data[a].cluster_id) === OTHER_COLOR ? 0 : 1;
      const cb = colorMap.get(data[b].cluster_id) === OTHER_COLOR ? 0 : 1;
      return ca - cb;
    });

    for (const i of indices) {
      const p = data[i];
      const px = pad.left + ((p.x - minX) / rangeX) * plotW;
      const py = pad.top + ((p.y - minY) / rangeY) * plotH;
      proj.push({ px, py, idx: i });

      ctx.fillStyle = colorMap.get(p.cluster_id) || OTHER_COLOR;
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      ctx.arc(px, py, 1.8, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    projected.current = proj;
  }, [data, size]);

  // Hover handler
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || projected.current.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Find nearest point within 12px
    let bestDist = 144; // 12^2
    let bestIdx = -1;
    for (const { px, py, idx } of projected.current) {
      const d = (px - mx) ** 2 + (py - my) ** 2;
      if (d < bestDist) { bestDist = d; bestIdx = idx; }
    }

    if (bestIdx >= 0) {
      const point = data[bestIdx];
      const count = clusterMeta.current.counts.get(point.cluster_id) || 0;
      setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, point, clusterCount: count });
    } else {
      setTooltip(null);
    }
  }, [data]);

  const handleMouseLeave = useCallback(() => setTooltip(null), []);

  // Legend: top 10 clusters
  const legend = clusterMeta.current.ranked.slice(0, 10).map(cid => ({
    cid,
    color: clusterMeta.current.colorMap.get(cid) || OTHER_COLOR,
    keywords: clusterMeta.current.keywordsMap.get(cid) || `Cluster ${cid}`,
    count: clusterMeta.current.counts.get(cid) || 0,
  }));

  return (
    <div ref={containerRef} className="relative w-full">
      <canvas
        ref={canvasRef}
        style={{ width: size.w, height: size.h }}
        className="rounded-lg"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
      {tooltip && (
        <div
          className="absolute pointer-events-none bg-[#1a1d27] border border-[#2a2e3d] rounded-lg shadow-xl p-3 text-xs z-50"
          style={{
            left: tooltip.x + 14,
            top: tooltip.y - 10,
            maxWidth: 260,
          }}
        >
          <p className="font-semibold text-[#e8eaed] mb-1">
            Cluster {tooltip.point.cluster_id}
          </p>
          <p className="text-[#8b8fa3]">
            Keywords: <span className="text-[#e8eaed]">{tooltip.point.keywords || '—'}</span>
          </p>
          <p className="text-[#8b8fa3]">
            Subreddit: <span className="text-[#e8eaed]">r/{tooltip.point.subreddit}</span>
          </p>
          <p className="text-[#8b8fa3]">
            Points in cluster: <span className="text-[#e8eaed] font-mono">{tooltip.clusterCount.toLocaleString()}</span>
          </p>
        </div>
      )}
      {legend.length > 0 && (
        <div className="flex flex-wrap gap-x-4 gap-y-1.5 mt-3 text-[11px] text-[#8b8fa3]">
          {legend.map(l => (
            <span key={l.cid} className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: l.color }} />
              <span className="truncate max-w-[160px]">{l.keywords}</span>
              <span className="text-[#64748b] font-mono">({l.count.toLocaleString()})</span>
            </span>
          ))}
          {clusterMeta.current.ranked.length > 10 && (
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: OTHER_COLOR }} />
              Others
            </span>
          )}
        </div>
      )}
    </div>
  );
}
