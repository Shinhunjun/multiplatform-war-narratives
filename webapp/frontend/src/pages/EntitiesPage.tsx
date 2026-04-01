import { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import {
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  BarChart, Bar, Cell,
} from 'recharts';
import ForceGraph2D from 'react-force-graph-2d';
import { fetchEntityNetwork, fetchEntityRelationships, fetchEntityMonths } from '../lib/api';
import type { EntityNetwork, EntityRelationship, Platform } from '../lib/api';

const COMMUNITY_COLORS = [
  '#6366f1', '#34d399', '#f87171', '#fbbf24', '#38bdf8',
  '#a78bfa', '#fb923c', '#e879f9', '#2dd4bf', '#f472b6',
  '#818cf8', '#4ade80', '#ef4444', '#eab308', '#0ea5e9',
  '#c084fc', '#f97316', '#d946ef', '#14b8a6', '#ec4899',
  '#64748b', '#94a3b8',
];

const PLATFORM_COLORS: Record<string, string> = {
  reddit: '#6366f1',
  news: '#f59e0b',
  tiktok: '#ff0050',
};

const PLATFORM_LABELS: Record<string, string> = {
  reddit: 'Reddit',
  news: 'GDELT News',
  tiktok: 'TikTok',
};

function NetworkGraph({ nodes, edges, relMap }: { nodes: any[]; edges: any[]; relMap: Record<string, string> }) {
  const [selected, setSelected] = useState<string | null>(null);
  const fgRef = useRef<any>(null);

  // Configure forces after mount for better spread
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;
    // Very strong repulsion to push nodes far apart
    fg.d3Force('charge')?.strength(-2000).distanceMin(50);
    // Long link distances so connected nodes aren't glued
    fg.d3Force('link')?.distance(200).strength(0.08);
    // Very weak centering
    fg.d3Force('center')?.strength(0.01);
    // Remove x/y positioning forces if they exist
    fg.d3Force('x', null);
    fg.d3Force('y', null);
    // Reheat with high alpha
    fg.d3ReheatSimulation();
  }, [nodes, edges]);

  const neighborSet = new Set<string>();
  if (selected) {
    edges.forEach(e => {
      if (e.source === selected || e.target === selected) {
        neighborSet.add(e.source);
        neighborSet.add(e.target);
      }
    });
  }

  // Compute log-scaled radius with a cap so big nodes don't dominate
  const maxFreq = Math.max(...nodes.map(n => n.frequency), 1);
  const getRadius = (freq: number) => {
    const normalized = Math.log(freq + 1) / Math.log(maxFreq + 1); // 0..1
    return 4 + normalized * 16; // 4px to 20px
  };

  return (
    <ForceGraph2D
      ref={fgRef}
      graphData={{
        nodes: nodes.map(n => ({ ...n, val: getRadius(n.frequency) })),
        links: edges.map(e => ({
          source: e.source, target: e.target, value: e.weight,
          relation: relMap[`${e.source}→${e.target}`] || '',
        })),
      }}
      onNodeClick={(node: any) => setSelected(selected === node.id ? null : node.id)}
      nodeLabel={(node: any) => `${node.id} (${node.type}, freq: ${node.frequency})`}
      linkLabel={(link: any) => {
        const s = typeof link.source === 'object' ? link.source.id : link.source;
        const t = typeof link.target === 'object' ? link.target.id : link.target;
        return link.relation ? `${s} → ${link.relation} → ${t}` : `${s} ↔ ${t} (weight: ${link.value})`;
      }}
      linkWidth={(link: any) => {
        if (!selected) return Math.min(2.5, Math.sqrt(link.value) * 0.3);
        const s = typeof link.source === 'object' ? link.source.id : link.source;
        const t = typeof link.target === 'object' ? link.target.id : link.target;
        return (s === selected || t === selected) ? Math.min(4, Math.sqrt(link.value)) : 0.2;
      }}
      linkColor={(link: any) => {
        if (!selected) return link.relation ? 'rgba(163, 130, 250, 0.18)' : 'rgba(100, 116, 139, 0.06)';
        const s = typeof link.source === 'object' ? link.source.id : link.source;
        const t = typeof link.target === 'object' ? link.target.id : link.target;
        return (s === selected || t === selected) ? 'rgba(163, 130, 250, 0.8)' : 'rgba(100, 116, 139, 0.02)';
      }}
      linkDirectionalArrowLength={selected ? 4 : 0}
      linkDirectionalArrowRelPos={1}
      d3AlphaDecay={0.015}
      d3VelocityDecay={0.3}
      cooldownTicks={300}
      warmupTicks={200}
      nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
        const r = getRadius(node.frequency);
        const isHighlighted = !selected || node.id === selected || neighborSet.has(node.id);

        // Node circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
        const color = COMMUNITY_COLORS[node.community % COMMUNITY_COLORS.length];
        ctx.fillStyle = color + (isHighlighted ? 'cc' : '18');
        ctx.fill();

        // Border
        ctx.strokeStyle = isHighlighted ? color : 'rgba(100,116,139,0.1)';
        ctx.lineWidth = node.id === selected ? 2.5 / globalScale : 0.5 / globalScale;
        ctx.stroke();

        // Label — always show for top nodes, show all when zoomed
        const isTopNode = node.frequency > maxFreq * 0.1;
        const showLabel = isHighlighted && (globalScale > 0.8 || isTopNode || node.id === selected);
        if (showLabel) {
          const fontSize = Math.max(12 / globalScale, 3);
          ctx.font = `bold ${fontSize}px Inter, system-ui, sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          // Text shadow for readability
          ctx.fillStyle = 'rgba(15, 17, 23, 0.7)';
          ctx.fillText(node.id, node.x + 0.5, node.y + r + 2.5);
          ctx.fillStyle = isHighlighted ? '#e8eaed' : 'rgba(232, 234, 237, 0.2)';
          ctx.fillText(node.id, node.x, node.y + r + 2);
        }
      }}
      nodePointerAreaPaint={(node: any, color: string, ctx: CanvasRenderingContext2D) => {
        const r = getRadius(node.frequency);
        ctx.beginPath();
        ctx.arc(node.x, node.y, r + 3, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
      }}
      backgroundColor="#1a1d27"
      width={undefined}
      height={600}
    />
  );
}

// Preset crisis periods
const PRESETS = [
  { label: 'All Time', start: '', end: '' },
  { label: 'Maduro Inauguration (2013)', start: '2013-03', end: '2013-06' },
  { label: 'Venezuelan Protests (2014)', start: '2014-01', end: '2014-06' },
  { label: 'Trump Sanctions (2017)', start: '2017-06', end: '2017-12' },
  { label: 'Guaidó Crisis (2019)', start: '2019-01', end: '2019-06' },
  { label: '2024 Election Crisis', start: '2024-06', end: '2024-12' },
  { label: 'Maduro Captured (2026)', start: '2025-12', end: '2026-02' },
];

export default function EntitiesPage() {
  const [platform, setPlatform] = useState<Platform>('reddit');
  const [months, setMonths] = useState<string[]>([]);
  const [startMonth, setStartMonth] = useState('');
  const [endMonth, setEndMonth] = useState('');
  const [network, setNetwork] = useState<EntityNetwork | null>(null);
  const [relationships, setRelationships] = useState<EntityRelationship[]>([]);
  const [loading, setLoading] = useState(true);

  // Load available months when platform changes
  useEffect(() => {
    fetchEntityMonths(platform).then(setMonths).catch(() => setMonths([]));
  }, [platform]);

  // Fetch data when platform or period changes
  const fetchData = useCallback(() => {
    setLoading(true);
    const s = startMonth || undefined;
    const e = endMonth || undefined;
    Promise.all([
      fetchEntityNetwork(platform, s, e).catch(() => null),
      fetchEntityRelationships(platform, s, e).catch(() => []),
    ]).then(([n, r]) => {
      setNetwork(n);
      setRelationships(r);
    }).finally(() => setLoading(false));
  }, [platform, startMonth, endMonth]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const applyPreset = (preset: typeof PRESETS[0]) => {
    setStartMonth(preset.start);
    setEndMonth(preset.end);
  };

  const periodLabel = useMemo(() => {
    if (!startMonth && !endMonth) return 'All Time';
    if (startMonth && endMonth) return `${startMonth} to ${endMonth}`;
    if (startMonth) return `From ${startMonth}`;
    return `Until ${endMonth}`;
  }, [startMonth, endMonth]);

  // Entity type icons
  const typeIcon: Record<string, string> = { PERSON: '👤', ORG: '🏛', LOCATION: '📍', EVENT: '⚡', POLICY: '📜', UNKNOWN: '•' };

  const topEntities = useMemo(() => {
    if (!network?.nodes?.length) return [];
    return [...network.nodes]
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, 20)
      .map(n => ({ name: n.id, frequency: n.frequency, community: n.community, type: n.type }));
  }, [network]);

  const communities = network?.communities || [];
  const topRels = relationships.slice(0, 20);
  const maxCommunityFreq = Math.max(...communities.map(c => c.total_frequency), 1);

  // Build relationship lookup
  const relByPair: Record<string, string> = {};
  relationships.forEach(r => {
    relByPair[`${r.source}→${r.target}`] = r.relation;
    relByPair[`${r.target}→${r.source}`] = r.relation;
  });

  const accentColor = PLATFORM_COLORS[platform] || '#6366f1';

  return (
    <div className="px-6 py-8 space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-[#e8eaed] tracking-tight">Knowledge Graph</h2>
        <p className="text-[13px] text-[#8b8fa3] mt-1">
          Entity co-occurrence network — explore entities and relationships across platforms and time periods
        </p>
        <div className="h-[2px] w-10 mt-2 rounded-full" style={{ backgroundColor: accentColor }} />
      </div>

      {/* Controls */}
      <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-4 space-y-4">
        {/* Platform selector */}
        <div className="flex items-center gap-3">
          <span className="text-[12px] text-[#64748b] font-medium w-16">Platform</span>
          <div className="flex gap-2">
            {(['reddit', 'news', 'tiktok'] as Platform[]).map(p => (
              <button
                key={p}
                onClick={() => setPlatform(p)}
                className={`px-3 py-1.5 rounded-lg text-[12px] font-medium transition-all border ${
                  platform === p
                    ? 'text-white border-transparent'
                    : 'text-[#8b8fa3] border-[#2a2e3d] hover:border-[#3a3e4d]'
                }`}
                style={platform === p ? { backgroundColor: PLATFORM_COLORS[p] + 'cc' } : {}}
              >
                {PLATFORM_LABELS[p]}
              </button>
            ))}
          </div>
        </div>

        {/* Period presets */}
        <div className="flex items-center gap-3">
          <span className="text-[12px] text-[#64748b] font-medium w-16">Period</span>
          <div className="flex flex-wrap gap-1.5">
            {PRESETS.map(preset => {
              const isActive = startMonth === preset.start && endMonth === preset.end;
              return (
                <button
                  key={preset.label}
                  onClick={() => applyPreset(preset)}
                  className={`px-2.5 py-1 rounded text-[11px] transition-all border ${
                    isActive
                      ? 'text-white border-transparent bg-[#3a3e5d]'
                      : 'text-[#8b8fa3] border-[#2a2e3d] hover:border-[#3a3e4d] hover:text-[#c4c8d8]'
                  }`}
                >
                  {preset.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Custom date range */}
        <div className="flex items-center gap-3">
          <span className="text-[12px] text-[#64748b] font-medium w-16">Custom</span>
          <select
            value={startMonth}
            onChange={e => setStartMonth(e.target.value)}
            className="bg-[#0f1117] border border-[#2a2e3d] rounded px-2 py-1 text-[11px] text-[#e8eaed]"
          >
            <option value="">Start (earliest)</option>
            {months.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
          <span className="text-[#64748b] text-[11px]">→</span>
          <select
            value={endMonth}
            onChange={e => setEndMonth(e.target.value)}
            className="bg-[#0f1117] border border-[#2a2e3d] rounded px-2 py-1 text-[11px] text-[#e8eaed]"
          >
            <option value="">End (latest)</option>
            {months.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
      </div>

      {/* Loading / Empty states */}
      {loading && (
        <div className="flex items-center justify-center h-[40vh]">
          <p className="text-[#64748b]">Building entity network for {PLATFORM_LABELS[platform]} ({periodLabel})...</p>
        </div>
      )}

      {!loading && (!network || !network.nodes?.length) && (
        <div className="flex items-center justify-center h-[40vh]">
          <p className="text-[#64748b]">No entity data available for {PLATFORM_LABELS[platform]} in this period</p>
        </div>
      )}

      {!loading && network && network.nodes?.length > 0 && (
        <>
          {/* Stats bar */}
          <div className="flex gap-4">
            {[
              { label: 'Entities', value: network.nodes.length },
              { label: 'Connections', value: network.edges.length },
              { label: 'Communities', value: communities.length },
              { label: 'Period', value: periodLabel },
            ].map(s => (
              <div key={s.label} className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] px-4 py-2.5 flex-1">
                <p className="text-[10px] text-[#64748b] uppercase tracking-wide">{s.label}</p>
                <p className="text-[16px] font-bold text-[#e8eaed] mt-0.5">{typeof s.value === 'number' ? s.value.toLocaleString() : s.value}</p>
              </div>
            ))}
          </div>

          {/* Communities */}
          <div className="space-y-3">
            <h3 className="text-[13px] font-semibold text-[#e8eaed]">Entity Communities (Louvain Detection)</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              {communities.slice(0, 10).map((c, i) => {
                const color = COMMUNITY_COLORS[i % COMMUNITY_COLORS.length];
                const barWidth = (c.total_frequency / maxCommunityFreq) * 100;
                const commRels: string[] = [];
                const members = c.top_members;
                for (let a = 0; a < members.length && commRels.length < 2; a++) {
                  for (let b = a + 1; b < members.length && commRels.length < 2; b++) {
                    const rel = relByPair[`${members[a]}→${members[b]}`] || relByPair[`${members[b]}→${members[a]}`];
                    if (rel) commRels.push(`${members[a]} → ${rel} → ${members[b]}`);
                  }
                }
                const nodeMap = Object.fromEntries(network.nodes.map(n => [n.id, n]));

                return (
                  <div key={c.id} className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] overflow-hidden">
                    <div className="h-[3px]" style={{ backgroundColor: color }} />
                    <div className="p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: color }} />
                        <span className="text-[13px] font-semibold text-[#e8eaed]">Community {i + 1}</span>
                        <span className="text-[10px] text-[#64748b] ml-auto font-mono">{c.size} entities · freq {c.total_frequency.toLocaleString()}</span>
                      </div>

                      {(c as any).summary && (
                        <p className="text-[11px] text-[#8b8fa3] leading-relaxed mb-3">{(c as any).summary}</p>
                      )}

                      {commRels.length > 0 && (
                        <div className="mb-3 space-y-1">
                          {commRels.map((rel, ri) => (
                            <div key={ri} className="flex items-center gap-1.5 text-[10px]">
                              <span style={{ color: accentColor }}>⟶</span>
                              <span className="text-[#c4c8d8] italic">{rel}</span>
                            </div>
                          ))}
                        </div>
                      )}

                      <div className="flex flex-wrap gap-1.5 mb-3">
                        {c.top_members.slice(0, 8).map(m => {
                          const node = nodeMap[m];
                          const icon = node ? typeIcon[node.type] || '•' : '•';
                          return (
                            <span key={m} className="px-2 py-0.5 rounded-full text-[10px] border flex items-center gap-1"
                              style={{ borderColor: color + '30', color: color, backgroundColor: color + '08' }}>
                              <span className="text-[8px]">{icon}</span>{m}
                            </span>
                          );
                        })}
                        {c.size > 8 && <span className="text-[10px] text-[#64748b]">+{c.size - 8} more</span>}
                      </div>

                      <div className="h-1 bg-[#2a2e3d] rounded-full overflow-hidden">
                        <div className="h-full rounded-full transition-all" style={{ width: `${barWidth}%`, backgroundColor: color + '80' }} />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Force-directed Network Graph */}
          {(() => {
            const relMap: Record<string, string> = {};
            relationships.forEach(r => {
              relMap[`${r.source}→${r.target}`] = r.relation;
              relMap[`${r.target}→${r.source}`] = r.relation;
            });

            // Keep top 2 strongest edges per node so graph is readable
            const edgesByNode: Record<string, typeof network.edges> = {};
            network.edges.forEach(e => {
              if (!edgesByNode[e.source]) edgesByNode[e.source] = [];
              if (!edgesByNode[e.target]) edgesByNode[e.target] = [];
              edgesByNode[e.source].push(e);
              edgesByNode[e.target].push(e);
            });
            const keptEdges = new Set<string>();
            Object.values(edgesByNode).forEach(nodeEdges => {
              nodeEdges.sort((a, b) => b.weight - a.weight);
              nodeEdges.slice(0, 2).forEach(e => {
                keptEdges.add(`${e.source}→${e.target}`);
              });
            });
            // Cap total edges: ~2x node count for readable density
            const maxEdges = Math.max(40, network.nodes.length * 2);
            let filteredEdges = network.edges
              .filter(e => keptEdges.has(`${e.source}→${e.target}`))
              .sort((a, b) => b.weight - a.weight)
              .slice(0, maxEdges);

            const connectedNodes = new Set<string>();
            filteredEdges.forEach(e => { connectedNodes.add(e.source); connectedNodes.add(e.target); });
            const filteredNodes = network.nodes.filter(n => connectedNodes.has(n.id));

            return (
              <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
                <div className="flex items-center justify-between mb-1">
                  <h3 className="text-[13px] font-semibold text-[#e8eaed]">
                    Entity Co-occurrence Network — {PLATFORM_LABELS[platform]}
                  </h3>
                  <span className="text-[10px] text-[#64748b]">{filteredNodes.length} nodes, {filteredEdges.length} edges (top connections only)</span>
                </div>
                <p className="text-[11px] text-[#64748b] mb-4">Click a node to highlight its connections. Hover edges for relationships. Scroll to zoom.</p>
                <div style={{ height: 600 }}>
                  <NetworkGraph
                    nodes={filteredNodes}
                    edges={filteredEdges}
                    relMap={relMap}
                  />
                </div>
              </div>
            );
          })()}

          {/* Top Entities + Relationships side by side */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
              <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">Top 20 Entities by Frequency</h3>
              <ResponsiveContainer width="100%" height={500}>
                <BarChart data={topEntities} layout="vertical">
                  <CartesianGrid stroke="#1e2235" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 10, fill: '#64748b' }} axisLine={{ stroke: '#2a2e3d' }} tickLine={false} />
                  <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 10, fill: '#8b8fa3' }} axisLine={{ stroke: '#2a2e3d' }} tickLine={false} />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null;
                      const d = payload[0].payload;
                      return (
                        <div className="bg-[#1a1d27] border border-[#2a2e3d] rounded-lg p-3 text-xs">
                          <p className="font-semibold text-[#e8eaed]">{d.name}</p>
                          <p className="text-[#8b8fa3]">Type: <span className="text-[#e8eaed]">{d.type}</span></p>
                          <p className="text-[#8b8fa3]">Frequency: <span className="text-[#e8eaed] font-mono">{d.frequency}</span></p>
                        </div>
                      );
                    }}
                  />
                  <Bar dataKey="frequency" radius={[0, 3, 3, 0]}>
                    {topEntities.map((e, i) => (
                      <Cell key={i} fill={COMMUNITY_COLORS[e.community % COMMUNITY_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
              <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">Top Relationships</h3>
              <div className="max-h-[500px] overflow-y-auto space-y-2">
                {topRels.map((r, i) => (
                  <div key={i} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#0f1117] border border-[#2a2e3d]">
                    <span className="text-[12px] font-medium shrink-0" style={{ color: accentColor }}>{r.source}</span>
                    <span className="text-[10px] text-[#64748b] italic flex-1 text-center">{r.relation}</span>
                    <span className="text-[12px] text-[#38bdf8] font-medium shrink-0">{r.target}</span>
                    <span className="text-[10px] text-[#64748b] font-mono ml-2">{r.count}</span>
                  </div>
                ))}
                {topRels.length === 0 && (
                  <p className="text-[11px] text-[#64748b] text-center py-8">No relationships found for this period</p>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
