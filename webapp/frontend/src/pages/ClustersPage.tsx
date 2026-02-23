import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell,
} from 'recharts';
import { fetchClusterSummaries, fetchClusterScatter, fetchTemporalClusters } from '../lib/api';
import type { ClusterSummary, ClusterScatterPoint } from '../lib/api';
import { useTimeRange } from '../lib/TimeRangeContext';
import { DarkTooltip, COLORS as COLORS_CHART, chartGrid, chartTick, chartAxisLine } from '../components/charts/shared';
import ClusterScatter from '../components/charts/ClusterScatter';

function sentimentColor(val: number): string {
  if (val < -0.3) return '#f87171';
  if (val < -0.1) return '#fbbf24';
  if (val < 0.1) return '#64748b';
  return '#34d399';
}

function clusterLabel(c: ClusterSummary): string {
  if (c.theme && c.theme !== 'Error') return c.theme;
  if (c.keywords_short) return c.keywords_short;
  return `Cluster ${c.cluster_id}`;
}

export default function ClustersPage() {
  const { selected } = useTimeRange();
  const [clusters, setClusters] = useState<ClusterSummary[]>([]);
  const [scatter, setScatter] = useState<ClusterScatterPoint[]>([]);
  const [scatterLoading, setScatterLoading] = useState(true);
  const [temporal, setTemporal] = useState<{ year_month: string; cluster_id: number; count: number }[]>([]);

  useEffect(() => {
    const start = selected?.[0];
    const end = selected?.[1];

    fetchClusterSummaries(30, 20, start, end).then(setClusters);

    setScatterLoading(true);
    fetchClusterScatter(50, 30000, start, end)
      .then(setScatter)
      .finally(() => setScatterLoading(false));

    fetchTemporalClusters(10, start, end).then(setTemporal);
  }, [selected]);

  const topClusters = clusters.slice(0, 20).map(c => ({
    ...c,
    label: clusterLabel(c),
  }));

  const temporalMap: Record<string, any> = {};
  const clusterIds = [...new Set(temporal.map(t => t.cluster_id))];
  temporal.forEach(row => {
    if (!temporalMap[row.year_month]) temporalMap[row.year_month] = { year_month: row.year_month };
    temporalMap[row.year_month][`c${row.cluster_id}`] = row.count;
  });
  const temporalRows = Object.values(temporalMap).sort((a: any, b: any) => a.year_month.localeCompare(b.year_month));

  return (
    <div className="px-6 py-8 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-[#e8eaed] tracking-tight">Cluster Analysis (HDBSCAN)</h2>
        <p className="text-[13px] text-[#8b8fa3] mt-1">
          {clusters.length} clusters with 20+ documents
        </p>
        <div className="h-[2px] w-10 bg-[#6366f1] mt-2 rounded-full" />
      </div>

      {/* Scatter Plot */}
      <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
        <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">UMAP Cluster Embedding</h3>
        {scatterLoading ? (
          <div className="flex items-center justify-center h-[400px] text-[#8b8fa3] text-sm">
            Loading scatter data...
          </div>
        ) : scatter.length === 0 ? (
          <div className="flex items-center justify-center h-[400px] text-[#8b8fa3] text-sm">
            No scatter data available (cluster_assignments.parquet not found)
          </div>
        ) : (
          <ClusterScatter data={scatter} />
        )}
      </div>

      {/* Bar Chart */}
      <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
        <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">Top 20 Clusters by Size</h3>
        <ResponsiveContainer width="100%" height={500}>
          <BarChart data={topClusters} layout="vertical">
            <CartesianGrid stroke={chartGrid} horizontal={false} />
            <XAxis type="number" tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
            <YAxis type="category" dataKey="label" width={180} tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0].payload;
                return (
                  <div className="bg-[#1a1d27] border border-[#2a2e3d] rounded-lg shadow-xl p-3 text-xs">
                    <p className="font-semibold text-[#e8eaed]">{d.label}</p>
                    <p className="text-[#8b8fa3] mt-1">Documents: <span className="text-[#e8eaed] font-mono">{d.count.toLocaleString()}</span></p>
                    <p className="text-[#8b8fa3]">Top Subreddit: <span className="text-[#e8eaed]">r/{d.top_subreddit}</span></p>
                    <p className="text-[#8b8fa3]">Sentiment: <span className="font-mono" style={{ color: sentimentColor(d.sentiment_mean) }}>{d.sentiment_mean.toFixed(3)}</span></p>
                    <p className="text-[#8b8fa3]">Period: <span className="text-[#e8eaed] font-mono">{d.time_start?.slice(0, 10)} — {d.time_end?.slice(0, 10)}</span></p>
                  </div>
                );
              }}
            />
            <Bar dataKey="count" name="Documents" radius={[0, 3, 3, 0]}>
              {topClusters.map((c, i) => (
                <Cell key={i} fill={sentimentColor(c.sentiment_mean)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="flex gap-5 mt-3 text-[11px] text-[#8b8fa3]">
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: '#f87171' }} />Very Negative</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: '#fbbf24' }} />Negative</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: '#64748b' }} />Neutral</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: '#34d399' }} />Positive</span>
        </div>
      </div>

      {temporalRows.length > 0 && (
        <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
          <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">Cluster Volume Over Time (Top 10)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={temporalRows}>
              <CartesianGrid stroke={chartGrid} />
              <XAxis dataKey="year_month" tick={chartTick} axisLine={chartAxisLine} tickLine={false} interval={Math.max(1, Math.floor(temporalRows.length / 8))} />
              <YAxis tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
              <Tooltip content={<DarkTooltip />} />
              {clusterIds.slice(0, 10).map((cid, i) => (
                <Bar key={cid} dataKey={`c${cid}`} stackId="a" fill={COLORS_CHART[i % COLORS_CHART.length]} name={`Cluster ${cid}`} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Details Table */}
      <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
        <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">Cluster Details</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#2a2e3d] text-left text-[11px] text-[#8b8fa3] uppercase tracking-wider">
                <th className="py-2.5 w-12 font-medium">ID</th>
                <th className="py-2.5 font-medium">Keywords</th>
                <th className="py-2.5 w-24 font-medium">Docs</th>
                <th className="py-2.5 w-28 font-medium">Subreddit</th>
                <th className="py-2.5 w-24 font-medium">Sentiment</th>
                <th className="py-2.5 w-48 font-medium">Period</th>
              </tr>
            </thead>
            <tbody>
              {clusters.slice(0, 50).map(c => (
                <tr key={c.cluster_id} className="border-b border-[#2a2e3d]/50 hover:bg-[#242838] transition-colors">
                  <td className="py-2.5 font-mono text-[#64748b] text-xs">{c.cluster_id}</td>
                  <td className="py-2.5 text-[#e8eaed]">{clusterLabel(c)}</td>
                  <td className="py-2.5 text-[#8b8fa3] font-mono text-[13px]">{c.count.toLocaleString()}</td>
                  <td className="py-2.5 text-[#8b8fa3]">r/{c.top_subreddit}</td>
                  <td className="py-2.5 font-mono text-[13px]" style={{ color: sentimentColor(c.sentiment_mean) }}>
                    {c.sentiment_mean.toFixed(3)}
                  </td>
                  <td className="py-2.5 text-xs text-[#64748b] font-mono">
                    {c.time_start?.slice(0, 10)} — {c.time_end?.slice(0, 10)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
