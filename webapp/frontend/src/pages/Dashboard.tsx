import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  LineChart, Line, Legend,
} from 'recharts';
import { fetchOverview, fetchSentimentByMonth, fetchSentimentBoxplot } from '../lib/api';
import type { OverviewStats, SentimentMonth, BoxPlotStat } from '../lib/api';
import { useTimeRange } from '../lib/TimeRangeContext';
import BoxPlotChart from '../components/charts/BoxPlotChart';
import { DarkTooltip, chartGrid, chartTick, chartAxisLine } from '../components/charts/shared';
import { PlatformStats } from '../components/PlatformStats';

function LoadingSkeleton() {
  return (
    <div className="px-6 py-8 space-y-6">
      <div className="h-7 w-32 animate-pulse bg-background-primary rounded" />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="animate-pulse bg-background-primary rounded-lg h-24 border border-background-secondary" />
        ))}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="animate-pulse bg-background-primary rounded-lg h-[340px] border border-background-secondary" />
        <div className="animate-pulse bg-background-primary rounded-lg h-[340px] border border-background-secondary" />
      </div>
    </div>
  );
}

const CustomSentimentTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-background-primary border border-background-secondary rounded-lg px-3 py-2 shadow-xl">
        <p className="text-[11px] text-text-secondary mb-1 font-mono">{label}</p>
        {data.reddit != null && (
          <p className="text-[12px] font-medium text-accent-reddit">
            Reddit: {data.reddit.toFixed(3)}
          </p>
        )}
        {data.news != null && (
          <p className="text-[12px] font-medium text-accent-news">
            GDELT News: {data.news.toFixed(3)}
          </p>
        )}
        {data.tiktok != null && (
          <p className="text-[12px] font-medium text-accent-tiktok">
            TikTok: {data.tiktok.toFixed(3)}
          </p>
        )}
      </div>
    );
  }
  return null;
};

export default function Dashboard() {
  const { selected } = useTimeRange();
  const [redditStats, setRedditStats] = useState<OverviewStats | null>(null);
  const [newsStats, setNewsStats] = useState<OverviewStats | null>(null);
  const [tiktokStats, setTiktokStats] = useState<OverviewStats | null>(null);
  const [redditMonth, setRedditMonth] = useState<SentimentMonth[]>([]);
  const [newsMonth, setNewsMonth] = useState<SentimentMonth[]>([]);
  const [tiktokMonth, setTiktokMonth] = useState<SentimentMonth[]>([]);
  const [boxplotData, setBoxplotData] = useState<BoxPlotStat[]>([]);
  const [newsBoxplotData, setNewsBoxplotData] = useState<BoxPlotStat[]>([]);

  useEffect(() => {
    fetchOverview('reddit').then(setRedditStats);
    fetchOverview('news').then(d => setNewsStats(d?.total_documents != null ? d : null)).catch(() => setNewsStats(null));
    fetchOverview('tiktok').then(d => setTiktokStats(d?.total_documents != null ? d : null)).catch(() => setTiktokStats(null));
  }, []);

  useEffect(() => {
    if (!selected) return;
    const [start, end] = selected;
    fetchSentimentByMonth(start, end, 'reddit').then(setRedditMonth);
    fetchSentimentByMonth(start, end, 'news').then(setNewsMonth).catch(() => setNewsMonth([]));
    fetchSentimentByMonth(start, end, 'tiktok').then(setTiktokMonth).catch(() => setTiktokMonth([]));
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

  const tiktokVolume = tiktokMonth.map(d => ({
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
  tiktokMonth.forEach(d => {
    if (!mergedSentiment[d.year_month]) mergedSentiment[d.year_month] = { year_month: d.year_month };
    mergedSentiment[d.year_month].tiktok = d.mean_sentiment;
  });
  const comparisonData = Object.values(mergedSentiment).sort((a: any, b: any) => a.year_month.localeCompare(b.year_month));

  const hasNews = newsStats !== null;
  const hasTiktok = tiktokStats !== null;

  return (
    <div className="px-6 py-8 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-text-primary tracking-tight">Overview</h2>
        <div className="h-[2px] w-10 bg-accent-reddit mt-2 rounded-full" />
      </div>

      {/* Stats cards — 3 columns */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <PlatformStats platform="Reddit" stats={redditStats} />
        <PlatformStats platform="GDELT News" stats={newsStats} />
        <PlatformStats platform="TikTok" stats={tiktokStats} />
      </div>

      {/* Sentiment comparison chart */}
      {comparisonData.length > 0 && (
        <div className="bg-background-primary rounded-lg border border-background-secondary p-5">
          <h3 className="text-[13px] font-semibold text-text-primary mb-4">Sentiment Over Time — Cross-Platform</h3>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={comparisonData}>
              <CartesianGrid stroke={chartGrid} />
              <XAxis dataKey="year_month" tick={chartTick} axisLine={chartAxisLine} tickLine={false} interval={Math.max(1, Math.floor(comparisonData.length / 10))} />
              <YAxis domain={[-0.8, 0.4]} tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
              <Tooltip content={<CustomSentimentTooltip />} />
              <Legend wrapperStyle={{ color: '#8b8fa3', fontSize: 12 }} />
              <Line type="monotone" dataKey="reddit" stroke="#6366f1" strokeWidth={2} dot={false} name="Reddit" />
              {hasNews && <Line type="monotone" dataKey="news" stroke="#f59e0b" strokeWidth={2} dot={false} name="GDELT News" />}
              {hasTiktok && <Line type="monotone" dataKey="tiktok" stroke="#ff0050" strokeWidth={2} dot={false} name="TikTok" />}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Volume — dynamic columns based on data availability */}
      {(() => {
        const volumeCharts = [
          { label: 'Reddit', data: redditVolume },
          { label: 'News', data: newsVolume },
          { label: 'TikTok', data: tiktokVolume },
        ].filter(c => c.data.length > 0);
        const cols = volumeCharts.length === 1 ? 'lg:grid-cols-1' : volumeCharts.length === 2 ? 'md:grid-cols-2' : 'md:grid-cols-2 lg:grid-cols-3';
        return (
          <div className={`grid grid-cols-1 ${cols} gap-4`}>
            {volumeCharts.map(({ label, data }) => (
              <div key={label} className="bg-background-primary rounded-lg border border-background-secondary p-5">
                <h3 className="text-[13px] font-semibold text-text-primary mb-1">Document Volume — {label}</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={data}>
                    <CartesianGrid stroke={chartGrid} />
                    <XAxis dataKey="year_month" tick={chartTick} axisLine={chartAxisLine} tickLine={false} interval={Math.max(1, Math.floor(data.length / 6))} />
                    <YAxis tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
                    <Tooltip content={<DarkTooltip />} />
                    <Legend wrapperStyle={{ color: '#8b8fa3', fontSize: 11 }} />
                    <Bar dataKey="Positive" stackId="a" fill="#34d399" />
                    <Bar dataKey="Neutral" stackId="a" fill="#64748b" />
                    <Bar dataKey="Negative" stackId="a" fill="#f87171" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        );
      })()}

      {/* Box plots — Reddit & News side by side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-background-primary rounded-lg border border-background-secondary p-5">
          <h3 className="text-[13px] font-semibold text-text-primary mb-4">Sentiment Distribution — Reddit</h3>
          <BoxPlotChart data={boxplotData} />
        </div>
        <div className="bg-background-primary rounded-lg border border-background-secondary p-5">
          <h3 className="text-[13px] font-semibold text-text-primary mb-4">Sentiment Distribution — News Top 20</h3>
          {newsBoxplotData.length > 0 ? (
            <BoxPlotChart data={newsBoxplotData} labelPrefix="" />
          ) : (
            <div className="flex items-center justify-center h-[200px] text-text-muted text-sm">No news data available</div>
          )}
        </div>
      </div>
    </div>
  );
}
