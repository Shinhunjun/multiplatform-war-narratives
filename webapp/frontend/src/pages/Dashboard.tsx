import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  LineChart, Line, Legend,
} from 'recharts';
import { fetchOverview, fetchSentimentByMonth, fetchSentimentBoxplot } from '../lib/api';
import type { OverviewStats, SentimentMonth, BoxPlotStat } from '../lib/api';
import { useTimeRange } from '../lib/TimeRangeContext';
import BoxPlotChart from '../components/charts/BoxPlotChart';
import { DarkTooltip, PlatformLabel, chartGrid, chartTick, chartAxisLine } from '../components/charts/shared';

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
  const [redditStats, setRedditStats] = useState<OverviewStats | null>(null);
  const [newsStats, setNewsStats] = useState<OverviewStats | null>(null);
  const [redditMonth, setRedditMonth] = useState<SentimentMonth[]>([]);
  const [newsMonth, setNewsMonth] = useState<SentimentMonth[]>([]);
  const [boxplotData, setBoxplotData] = useState<BoxPlotStat[]>([]);
  const [newsBoxplotData, setNewsBoxplotData] = useState<BoxPlotStat[]>([]);

  useEffect(() => {
    fetchOverview('reddit').then(setRedditStats);
    fetchOverview('news').then(d => setNewsStats(d?.total_documents != null ? d : null)).catch(() => setNewsStats(null));
  }, []);

  useEffect(() => {
    if (!selected) return;
    const [start, end] = selected;
    fetchSentimentByMonth(start, end, 'reddit').then(setRedditMonth);
    fetchSentimentByMonth(start, end, 'news').then(setNewsMonth).catch(() => setNewsMonth([]));
    fetchSentimentBoxplot(start, end).then(setBoxplotData);
    fetchSentimentBoxplot(start, end, 'news').then(setNewsBoxplotData).catch(() => setNewsBoxplotData([]));
  }, [selected]);

  if (!redditStats) return <LoadingSkeleton />;

  const redditVolume = redditMonth.map(d => ({
    year_month: d.year_month,
    Positive: Math.round(d.positive_ratio * d.total_count),
    Neutral: Math.round((1 - d.positive_ratio - d.negative_ratio) * d.total_count),
    Negative: Math.round(d.negative_ratio * d.total_count),
  }));

  const newsVolume = newsMonth.map(d => ({
    year_month: d.year_month,
    Positive: Math.round(d.positive_ratio * d.total_count),
    Neutral: Math.round((1 - d.positive_ratio - d.negative_ratio) * d.total_count),
    Negative: Math.round(d.negative_ratio * d.total_count),
  }));

  // Merge sentiment for comparison chart
  const mergedSentiment: Record<string, any> = {};
  redditMonth.forEach(d => {
    mergedSentiment[d.year_month] = { year_month: d.year_month, reddit: d.mean_sentiment };
  });
  newsMonth.forEach(d => {
    if (!mergedSentiment[d.year_month]) mergedSentiment[d.year_month] = { year_month: d.year_month };
    mergedSentiment[d.year_month].news = d.mean_sentiment;
  });
  const comparisonData = Object.values(mergedSentiment).sort((a: any, b: any) => a.year_month.localeCompare(b.year_month));

  const hasNews = newsStats !== null;

  return (
    <div className="px-6 py-8 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-[#e8eaed] tracking-tight">Overview</h2>
        <div className="h-[2px] w-10 bg-[#6366f1] mt-2 rounded-full" />
      </div>

      {/* Stats cards — side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Reddit stats */}
        <div className="space-y-3">
          <PlatformLabel platform="Reddit" color="#6366f1" />
          <div className="grid grid-cols-2 gap-3">
            <StatCard label="Documents" value={redditStats.total_documents.toLocaleString()} sub="Posts & comments" accent="#6366f1" />
            <StatCard label="Subreddits" value={redditStats.subreddits ?? 0} sub={`${redditStats.date_range.start} — ${redditStats.date_range.end}`} accent="#6366f1" />
            <StatCard label="Topics" value={redditStats.num_topics} sub="BERTopic clusters" accent="#6366f1" />
            <StatCard label="Avg Sentiment" value={redditStats.avg_sentiment.toFixed(3)} sub="Mean across subreddits" accent="#6366f1" />
          </div>
        </div>

        {/* News stats */}
        <div className="space-y-3">
          <PlatformLabel platform="GDELT News" color="#f59e0b" />
          {hasNews ? (
            <div className="grid grid-cols-2 gap-3">
              <StatCard label="Documents" value={newsStats.total_documents.toLocaleString()} sub="News articles" accent="#f59e0b" />
              <StatCard label="Sources" value={newsStats.sources ?? 0} sub={`${newsStats.date_range.start} — ${newsStats.date_range.end}`} accent="#f59e0b" />
              <StatCard label="Topics" value={newsStats.num_topics} sub="Mapped from Reddit" accent="#f59e0b" />
              <StatCard label="Avg Sentiment" value={newsStats.avg_sentiment.toFixed(3)} sub="Mean across sources" accent="#f59e0b" />
            </div>
          ) : (
            <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-8 text-center">
              <p className="text-[#8b8fa3] text-sm">News data not available yet.</p>
              <p className="text-[#64748b] text-xs mt-1">Run <code className="text-[#f59e0b]">python scripts/analyze_gdelt.py</code> to generate.</p>
            </div>
          )}
        </div>
      </div>

      {/* Sentiment comparison chart */}
      {hasNews && comparisonData.length > 0 && (
        <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
          <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">Sentiment Over Time — Reddit vs News</h3>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={comparisonData}>
              <CartesianGrid stroke={chartGrid} />
              <XAxis dataKey="year_month" tick={chartTick} axisLine={chartAxisLine} tickLine={false} interval={Math.max(1, Math.floor(comparisonData.length / 10))} />
              <YAxis domain={[-0.8, 0.4]} tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
              <Tooltip content={<DarkTooltip />} />
              <Legend wrapperStyle={{ color: '#8b8fa3', fontSize: 12 }} />
              <Line type="monotone" dataKey="reddit" stroke="#6366f1" strokeWidth={2} dot={false} name="Reddit" />
              <Line type="monotone" dataKey="news" stroke="#f59e0b" strokeWidth={2} dot={false} name="GDELT News" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Volume side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
          <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-1">Document Volume — Reddit</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={redditVolume}>
              <CartesianGrid stroke={chartGrid} />
              <XAxis dataKey="year_month" tick={chartTick} axisLine={chartAxisLine} tickLine={false} interval={Math.max(1, Math.floor(redditVolume.length / 6))} />
              <YAxis tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
              <Tooltip content={<DarkTooltip />} />
              <Legend wrapperStyle={{ color: '#8b8fa3', fontSize: 11 }} />
              <Bar dataKey="Positive" stackId="a" fill="#34d399" />
              <Bar dataKey="Neutral" stackId="a" fill="#64748b" />
              <Bar dataKey="Negative" stackId="a" fill="#f87171" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
          <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-1">Document Volume — News</h3>
          {newsVolume.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={newsVolume}>
                <CartesianGrid stroke={chartGrid} />
                <XAxis dataKey="year_month" tick={chartTick} axisLine={chartAxisLine} tickLine={false} interval={Math.max(1, Math.floor(newsVolume.length / 6))} />
                <YAxis tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
                <Tooltip content={<DarkTooltip />} />
                <Legend wrapperStyle={{ color: '#8b8fa3', fontSize: 11 }} />
                <Bar dataKey="Positive" stackId="a" fill="#34d399" />
                <Bar dataKey="Neutral" stackId="a" fill="#64748b" />
                <Bar dataKey="Negative" stackId="a" fill="#f87171" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[260px] text-[#64748b] text-sm">No news data available</div>
          )}
        </div>
      </div>

      {/* Box plots — Reddit & News side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
          <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">Sentiment Distribution — Reddit</h3>
          <BoxPlotChart data={boxplotData} />
        </div>
        <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
          <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">Sentiment Distribution — News Top 20</h3>
          {newsBoxplotData.length > 0 ? (
            <BoxPlotChart data={newsBoxplotData} labelPrefix="" />
          ) : (
            <div className="flex items-center justify-center h-[200px] text-[#64748b] text-sm">No news data available</div>
          )}
        </div>
      </div>
    </div>
  );
}
