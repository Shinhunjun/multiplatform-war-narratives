import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  LineChart, Line, Legend,
} from 'recharts';
import { fetchOverview, fetchSentimentByMonth, fetchSentimentBySubreddit } from '../lib/api';
import type { OverviewStats, SentimentMonth, SentimentSubreddit } from '../lib/api';

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <p className="text-xs text-slate-500 uppercase tracking-wide">{label}</p>
      <p className="text-2xl font-bold text-slate-800 mt-1">{value}</p>
      {sub && <p className="text-xs text-slate-400 mt-1">{sub}</p>}
    </div>
  );
}

export default function Dashboard() {
  const [stats, setStats] = useState<OverviewStats | null>(null);
  const [monthData, setMonthData] = useState<SentimentMonth[]>([]);
  const [subData, setSubData] = useState<SentimentSubreddit[]>([]);

  useEffect(() => {
    fetchOverview().then(setStats);
    fetchSentimentByMonth().then(setMonthData);
    fetchSentimentBySubreddit().then(setSubData);
  }, []);

  if (!stats) return <p className="p-8 text-slate-500">Loading...</p>;

  const subSorted = [...subData].sort((a, b) => a.mean_sentiment - b.mean_sentiment);

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-xl font-bold text-slate-800">Dashboard</h2>

      {/* Stat Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Total Documents" value={stats.total_documents.toLocaleString()} sub="Reddit posts & comments" />
        <StatCard label="Subreddits" value={stats.subreddits} sub={`${stats.date_range.start} ~ ${stats.date_range.end}`} />
        <StatCard label="Topics" value={stats.num_topics} sub="BERTopic clusters" />
        <StatCard label="Avg Sentiment" value={stats.avg_sentiment.toFixed(3)} sub="Mean across all subreddits" />
      </div>

      {/* Sentiment Timeline */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Sentiment Over Time (Monthly)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={monthData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="year_month"
              tick={{ fontSize: 10 }}
              interval={11}
            />
            <YAxis domain={[-0.8, 0.2]} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Line
              type="monotone"
              dataKey="mean_sentiment"
              stroke="#e63946"
              strokeWidth={1.5}
              dot={false}
              name="Mean Sentiment"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Sentiment by Subreddit */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Sentiment by Subreddit</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={subSorted} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[-0.6, 0.1]} tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="subreddit" width={90} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="mean_sentiment" fill="#1e3a5f" name="Mean Sentiment" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Volume Timeline */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Document Volume Over Time</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={monthData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year_month" tick={{ fontSize: 10 }} interval={11} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="positive_count" stackId="a" fill="#2a9d8f" name="Positive" />
            <Bar dataKey="neutral_count" stackId="a" fill="#a8dadc" name="Neutral" />
            <Bar dataKey="negative_count" stackId="a" fill="#e63946" name="Negative" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
