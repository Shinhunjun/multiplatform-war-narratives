import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell,
} from 'recharts';
import { fetchClusterSummaries } from '../lib/api';
import type { ClusterSummary } from '../lib/api';

function sentimentColor(val: number): string {
  if (val < -0.3) return '#e63946';
  if (val < -0.1) return '#f4a261';
  if (val < 0.1) return '#a8dadc';
  return '#2a9d8f';
}

export default function ClustersPage() {
  const [clusters, setClusters] = useState<ClusterSummary[]>([]);

  useEffect(() => {
    fetchClusterSummaries(30, 20).then(setClusters);
  }, []);

  const topClusters = clusters.slice(0, 20).map(c => ({
    ...c,
    label: c.theme !== 'Error' ? c.theme : `Cluster ${c.cluster_id}`,
    keywords: c.summary !== 'Anthropic API key required' ? c.summary : '-',
  }));

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-xl font-bold text-slate-800">Cluster Analysis (HDBSCAN)</h2>
      <p className="text-sm text-slate-500">
        {clusters.length} clusters with 20+ documents (of 3,406 total)
      </p>

      {/* Top Clusters by Size */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Top 20 Clusters by Size</h3>
        <ResponsiveContainer width="100%" height={500}>
          <BarChart data={topClusters} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="label" width={120} tick={{ fontSize: 10 }} />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0].payload;
                return (
                  <div className="bg-white border rounded shadow p-2 text-xs max-w-xs">
                    <p className="font-bold">{d.label}</p>
                    <p>Documents: {d.count.toLocaleString()}</p>
                    <p>Top Subreddit: r/{d.top_subreddit}</p>
                    <p>Sentiment: {d.sentiment_mean.toFixed(3)}</p>
                    <p>Period: {d.time_start?.slice(0, 10)} ~ {d.time_end?.slice(0, 10)}</p>
                  </div>
                );
              }}
            />
            <Bar dataKey="count" name="Documents">
              {topClusters.map((c, i) => (
                <Cell key={i} fill={sentimentColor(c.sentiment_mean)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="flex gap-4 mt-2 text-xs text-slate-500">
          <span><span className="inline-block w-3 h-3 rounded mr-1" style={{ backgroundColor: '#e63946' }} />Very Negative</span>
          <span><span className="inline-block w-3 h-3 rounded mr-1" style={{ backgroundColor: '#f4a261' }} />Negative</span>
          <span><span className="inline-block w-3 h-3 rounded mr-1" style={{ backgroundColor: '#a8dadc' }} />Neutral</span>
          <span><span className="inline-block w-3 h-3 rounded mr-1" style={{ backgroundColor: '#2a9d8f' }} />Positive</span>
        </div>
      </div>

      {/* Cluster Detail Table */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Cluster Details</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-slate-500">
                <th className="py-2 w-12">ID</th>
                <th className="py-2">Theme</th>
                <th className="py-2 w-24">Docs</th>
                <th className="py-2 w-28">Subreddit</th>
                <th className="py-2 w-24">Sentiment</th>
                <th className="py-2 w-48">Period</th>
              </tr>
            </thead>
            <tbody>
              {clusters.slice(0, 50).map(c => (
                <tr key={c.cluster_id} className="border-b hover:bg-slate-50">
                  <td className="py-2 font-mono text-slate-400">{c.cluster_id}</td>
                  <td className="py-2">{c.theme !== 'Error' ? c.theme : '-'}</td>
                  <td className="py-2">{c.count.toLocaleString()}</td>
                  <td className="py-2 text-slate-600">r/{c.top_subreddit}</td>
                  <td className="py-2" style={{ color: sentimentColor(c.sentiment_mean) }}>
                    {c.sentiment_mean.toFixed(3)}
                  </td>
                  <td className="py-2 text-xs text-slate-400">
                    {c.time_start?.slice(0, 10)} ~ {c.time_end?.slice(0, 10)}
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
