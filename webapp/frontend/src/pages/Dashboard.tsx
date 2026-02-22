import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  LineChart, Line, Legend,
} from 'recharts';
import { fetchOverview, fetchSentimentByMonth, fetchSentimentBoxplot } from '../lib/api';
import type { OverviewStats, SentimentMonth, BoxPlotStat } from '../lib/api';
import { useTimeRange } from '../lib/TimeRangeContext';
import BoxPlotChart from '../components/charts/BoxPlotChart';

const chartGrid = '#2a2e3d';
const chartTick = { fontSize: 11, fill: '#8b8fa3' };
const chartAxisLine = { stroke: '#2a2e3d' };

function DarkTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#1a1d27] border border-[#2a2e3d] rounded-lg px-3 py-2 shadow-xl">
      <p className="text-[11px] text-[#8b8fa3] mb-1 font-mono">{label}</p>
      {payload.map((p: any, i: number) => (
        <p key={i} className="text-[12px] font-medium" style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toLocaleString() : p.value}
        </p>
      ))}
    </div>
  );
}

function StatCard({ label, value, sub, accent }: { label: string; value: string | number; sub?: string; accent?: string }) {
  return (
    <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5 relative overflow-hidden">
      <div className="absolute top-0 left-0 right-0 h-[2px]" style={{ backgroundColor: accent || '#6366f1' }} />
      <p className="text-[11px] text-[#8b8fa3] uppercase tracking-wider font-medium">{label}</p>
      <p className="text-2xl font-semibold text-[#e8eaed] mt-1.5 font-mono">{value}</p>
      {sub && <p className="text-[11px] text-[#64748b] mt-1">{sub}</p>}
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="px-6 py-8 space-y-6">
      <div className="h-7 w-32 animate-pulse bg-[#1a1d27] rounded" />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="animate-pulse bg-[#1a1d27] rounded-lg h-24 border border-[#2a2e3d]" />
        ))}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="animate-pulse bg-[#1a1d27] rounded-lg h-[340px] border border-[#2a2e3d]" />
        <div className="animate-pulse bg-[#1a1d27] rounded-lg h-[340px] border border-[#2a2e3d]" />
      </div>
    </div>
  );
}

export default function Dashboard() {
  const { selected } = useTimeRange();
  const [stats, setStats] = useState<OverviewStats | null>(null);
  const [monthData, setMonthData] = useState<SentimentMonth[]>([]);
  const [boxplotData, setBoxplotData] = useState<BoxPlotStat[]>([]);

  useEffect(() => {
    fetchOverview().then(setStats);
  }, []);

  useEffect(() => {
    if (!selected) return;
    const [start, end] = selected;
    fetchSentimentByMonth(start, end).then(setMonthData);
    fetchSentimentBoxplot(start, end).then(setBoxplotData);
  }, [selected]);

  if (!stats) return <LoadingSkeleton />;

  const volumeData = monthData.map(d => ({
    year_month: d.year_month,
    Positive: Math.round(d.positive_ratio * d.total_count),
    Neutral: Math.round((1 - d.positive_ratio - d.negative_ratio) * d.total_count),
    Negative: Math.round(d.negative_ratio * d.total_count),
  }));

  return (
    <div className="px-6 py-8 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-[#e8eaed] tracking-tight">Overview</h2>
        <div className="h-[2px] w-10 bg-[#6366f1] mt-2 rounded-full" />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Total Documents" value={stats.total_documents.toLocaleString()} sub="Reddit posts & comments" accent="#6366f1" />
        <StatCard label="Subreddits" value={stats.subreddits} sub={`${stats.date_range.start} — ${stats.date_range.end}`} accent="#34d399" />
        <StatCard label="Topics" value={stats.num_topics} sub="BERTopic clusters" accent="#fbbf24" />
        <StatCard label="Avg Sentiment" value={stats.avg_sentiment.toFixed(3)} sub="Mean across all subreddits" accent="#f87171" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
          <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">Sentiment Over Time</h3>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={monthData}>
              <CartesianGrid stroke={chartGrid} />
              <XAxis dataKey="year_month" tick={chartTick} axisLine={chartAxisLine} tickLine={false} interval={Math.max(1, Math.floor(monthData.length / 8))} />
              <YAxis domain={[-0.8, 0.2]} tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
              <Tooltip content={<DarkTooltip />} />
              <Line
                type="monotone"
                dataKey="mean_sentiment"
                stroke="#6366f1"
                strokeWidth={2}
                dot={false}
                name="Mean Sentiment"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
          <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">Sentiment Distribution by Subreddit</h3>
          <BoxPlotChart data={boxplotData} />
        </div>
      </div>

      <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
        <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">Document Volume Over Time</h3>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={volumeData}>
            <CartesianGrid stroke={chartGrid} />
            <XAxis dataKey="year_month" tick={chartTick} axisLine={chartAxisLine} tickLine={false} interval={Math.max(1, Math.floor(volumeData.length / 8))} />
            <YAxis tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
            <Tooltip content={<DarkTooltip />} />
            <Legend wrapperStyle={{ color: '#8b8fa3', fontSize: 12 }} />
            <Bar dataKey="Positive" stackId="a" fill="#34d399" />
            <Bar dataKey="Neutral" stackId="a" fill="#64748b" />
            <Bar dataKey="Negative" stackId="a" fill="#f87171" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
