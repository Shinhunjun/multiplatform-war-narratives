import { useEffect, useState, useRef, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';
import { fetchClusterSummaries, fetchClusterScatter, fetchTemporalClusters, fetchClustersMonthly, fetchClustersMonthlyMonths } from '../lib/api';
import type { ClusterSummary, ClusterScatterPoint, ClusterMonthly } from '../lib/api';
import { useTimeRange } from '../lib/TimeRangeContext';
import { DarkTooltip, COLORS as COLORS_CHART, chartGrid, chartTick, chartAxisLine } from '../components/charts/shared';
import ClusterScatter from '../components/charts/ClusterScatter';

function clusterLabel(c: ClusterSummary): string {
  if (c.theme && c.theme !== 'Error') return c.theme;
  if (c.keywords_short) return c.keywords_short;
  return `Cluster ${c.cluster_id}`;
}

function MonthlyClusterChart({ platform, title, color }: { platform?: string; title: string; color: string }) {
  const [months, setMonths] = useState<string[]>([]);
  const [sliderIdx, setSliderIdx] = useState(0);
  const [clusters, setClusters] = useState<ClusterMonthly[]>([]);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    fetchClustersMonthlyMonths(platform).then(m => {
      setMonths(m);
      if (m.length > 0) setSliderIdx(m.length - 1);
    }).catch(() => setMonths([]));
  }, [platform]);

  const fetchData = useCallback((month: string) => {
    setLoading(true);
    fetchClustersMonthly(month, 15, platform)
      .then(setClusters)
      .catch(() => setClusters([]))
      .finally(() => setLoading(false));
  }, [platform]);

  useEffect(() => {
    if (months.length === 0) return;
    fetchData(months[sliderIdx]);
  }, [months, sliderIdx, fetchData]);

  const handleSlider = (e: React.ChangeEvent<HTMLInputElement>) => {
    const idx = Number(e.target.value);
    setSliderIdx(idx);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      fetchData(months[idx]);
    }, 150);
  };

  if (months.length === 0) {
    return (
      <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
        <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">{title}</h3>
        <div className="flex items-center justify-center h-[350px] text-[#64748b] text-sm">No monthly data available</div>
      </div>
    );
  }

  const currentMonth = months[sliderIdx] || '';
  const label = currentMonth
    ? new Date(currentMonth + '-01').toLocaleDateString('en-US', { year: 'numeric', month: 'long' })
    : '';

  const bars = clusters.map(c => ({
    label: c.keywords || `Cluster ${c.cluster_id}`,
    count: c.count,
    cluster_id: c.cluster_id,
  }));

  return (
    <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
      <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">{title}</h3>
      <div className="mb-4 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[#8b8fa3] text-xs">{months[0]}</span>
          <span className="text-[#e8eaed] text-sm font-medium">{label}</span>
          <span className="text-[#8b8fa3] text-xs">{months[months.length - 1]}</span>
        </div>
        <input
          type="range"
          min={0}
          max={months.length - 1}
          value={sliderIdx}
          onChange={handleSlider}
          className="w-full h-1.5 rounded-lg appearance-none cursor-pointer accent-[#6366f1] bg-[#2a2e3d]"
        />
      </div>
      {loading ? (
        <div className="flex items-center justify-center h-[400px] text-[#64748b] text-sm">Loading...</div>
      ) : bars.length === 0 ? (
        <div className="flex items-center justify-center h-[400px] text-[#64748b] text-sm">No clusters for this month</div>
      ) : (
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={bars} layout="vertical">
            <CartesianGrid stroke={chartGrid} horizontal={false} />
            <XAxis type="number" tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
            <YAxis type="category" dataKey="label" width={180} tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
            <Tooltip content={<DarkTooltip />} />
            <Bar dataKey="count" fill={color} name="Documents" radius={[0, 3, 3, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

export default function ClustersPage() {
  const { selected } = useTimeRange();
  const [redditClusters, setRedditClusters] = useState<ClusterSummary[]>([]);
  const [newsClusters, setNewsClusters] = useState<ClusterSummary[]>([]);
  const [redditTemporal, setRedditTemporal] = useState<{ year_month: string; cluster_id: number; count: number }[]>([]);
  const [newsTemporal, setNewsTemporal] = useState<{ year_month: string; cluster_id: number; count: number }[]>([]);
  const [redditScatter, setRedditScatter] = useState<ClusterScatterPoint[]>([]);
  const [newsScatter, setNewsScatter] = useState<ClusterScatterPoint[]>([]);
  const [scatterLoading, setScatterLoading] = useState(true);

  useEffect(() => {
    const start = selected?.[0];
    const end = selected?.[1];

    fetchClusterSummaries(20, 10, start, end).then(setRedditClusters);
    fetchClusterSummaries(20, 10, start, end, 'news').then(setNewsClusters).catch(() => setNewsClusters([]));
    fetchTemporalClusters(10, start, end).then(setRedditTemporal);
    fetchTemporalClusters(10, start, end, 'news').then(setNewsTemporal).catch(() => setNewsTemporal([]));

    setScatterLoading(true);
    Promise.all([
      fetchClusterScatter(30, 15000, start, end).catch(() => []),
      fetchClusterScatter(30, 15000, start, end, 'news').catch(() => []),
    ]).then(([r, n]) => {
      setRedditScatter(r);
      setNewsScatter(n);
    }).finally(() => setScatterLoading(false));
  }, [selected]);

  const makeTopClusters = (clusters: ClusterSummary[]) => clusters.slice(0, 20).map(c => ({ ...c, label: clusterLabel(c) }));
  const makeTemporalRows = (temporal: { year_month: string; cluster_id: number; count: number }[]) => {
    const map: Record<string, any> = {};
    temporal.forEach(row => {
      if (!map[row.year_month]) map[row.year_month] = { year_month: row.year_month };
      map[row.year_month][`c${row.cluster_id}`] = row.count;
    });
    return Object.values(map).sort((a: any, b: any) => a.year_month.localeCompare(b.year_month));
  };

  return (
    <div className="px-6 py-8 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-[#e8eaed] tracking-tight">Cluster Analysis (HDBSCAN)</h2>
        <p className="text-[13px] text-[#8b8fa3] mt-1">Cross-platform clustering comparison</p>
        <div className="h-[2px] w-10 bg-[#6366f1] mt-2 rounded-full" />
      </div>

      {/* UMAP Cluster Scatter — per platform */}
      {scatterLoading ? (
        <div className="flex items-center justify-center h-[400px] text-[#8b8fa3] text-sm">Loading scatter data...</div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {redditScatter.length > 0 && (
            <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
              <div className="flex items-center gap-2 mb-4">
                <span className="w-2 h-2 rounded-full bg-[#6366f1]" />
                <h3 className="text-[13px] font-semibold text-[#e8eaed]">UMAP Clusters — Reddit</h3>
              </div>
              <ClusterScatter data={redditScatter} />
            </div>
          )}
          {newsScatter.length > 0 && (
            <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
              <div className="flex items-center gap-2 mb-4">
                <span className="w-2 h-2 rounded-full bg-[#f59e0b]" />
                <h3 className="text-[13px] font-semibold text-[#e8eaed]">UMAP Clusters — GDELT News</h3>
              </div>
              <ClusterScatter data={newsScatter} />
            </div>
          )}
        </div>
      )}

      {/* Top 20 Clusters — side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {[
          { label: 'Reddit', color: '#6366f1', data: makeTopClusters(redditClusters) },
          { label: 'GDELT News', color: '#f59e0b', data: makeTopClusters(newsClusters) },
        ].filter(p => p.data.length > 0).map(({ label, color, data }) => (
          <div key={label} className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
            <div className="flex items-center gap-2 mb-4">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
              <h3 className="text-[13px] font-semibold text-[#e8eaed]">Top 20 Clusters — {label}</h3>
            </div>
            <ResponsiveContainer width="100%" height={450}>
              <BarChart data={data} layout="vertical">
                <CartesianGrid stroke={chartGrid} horizontal={false} />
                <XAxis type="number" tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
                <YAxis type="category" dataKey="label" width={140} tick={{ fontSize: 10, fill: '#8b8fa3' }} axisLine={chartAxisLine} tickLine={false} />
                <Tooltip content={<DarkTooltip />} />
                <Bar dataKey="count" fill={color} name="Documents" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>

      {/* Cluster Volume Over Time — side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {[
          { label: 'Reddit', temporal: redditTemporal },
          { label: 'GDELT News', temporal: newsTemporal },
        ].filter(p => p.temporal.length > 0).map(({ label, temporal }) => {
          const rows = makeTemporalRows(temporal);
          const cids = [...new Set(temporal.map(t => t.cluster_id))];
          return (
            <div key={label} className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
              <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">Cluster Volume Over Time — {label}</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={rows}>
                  <CartesianGrid stroke={chartGrid} />
                  <XAxis dataKey="year_month" tick={chartTick} axisLine={chartAxisLine} tickLine={false} interval={Math.max(1, Math.floor(rows.length / 8))} />
                  <YAxis tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
                  <Tooltip content={<DarkTooltip />} />
                  {cids.slice(0, 10).map((cid, i) => (
                    <Bar key={cid} dataKey={`c${cid}`} stackId="a" fill={COLORS_CHART[i % COLORS_CHART.length]} name={`Cluster ${cid}`} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>

      {/* Monthly Cluster Slider */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MonthlyClusterChart platform="reddit" title="Monthly Clusters — Reddit" color="#6366f1" />
        <MonthlyClusterChart platform="news" title="Monthly Clusters — GDELT News" color="#f59e0b" />
      </div>
    </div>
  );
}
