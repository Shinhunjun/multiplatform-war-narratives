import { useEffect, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts';
import { fetchSentimentBySubredditMonth, fetchSentimentBySubreddit, fetchSentimentBoxplot, fetchSentimentByMonth } from '../lib/api';
import type { SentimentSubreddit, SentimentMonth, BoxPlotStat } from '../lib/api';
import { useTimeRange } from '../lib/TimeRangeContext';
import BoxPlotChart from '../components/charts/BoxPlotChart';
import { DarkTooltip, PlatformLabel, COLORS, chartGrid, chartTick, chartAxisLine } from '../components/charts/shared';

export default function SentimentPage() {
  const { selected } = useTimeRange();
  const [subreddits, setSubreddits] = useState<SentimentSubreddit[]>([]);
  const [sel, setSel] = useState<string[]>(['worldnews', 'politics', 'venezuela', 'vzla']);
  const [timelineData, setTimelineData] = useState<Record<string, any>[]>([]);
  const [boxplotData, setBoxplotData] = useState<BoxPlotStat[]>([]);
  const [newsBoxplotData, setNewsBoxplotData] = useState<BoxPlotStat[]>([]);

  // News state
  const [redditMonthly, setRedditMonthly] = useState<SentimentMonth[]>([]);
  const [newsMonthly, setNewsMonthly] = useState<SentimentMonth[]>([]);
  const [newsSources, setNewsSources] = useState<any[]>([]);

  useEffect(() => {
    fetchSentimentBySubreddit().then(setSubreddits);
    fetchSentimentBySubreddit('news').then(setNewsSources).catch(() => setNewsSources([]));
  }, []);

  useEffect(() => {
    if (!selected) return;
    const [start, end] = selected;
    fetchSentimentBoxplot(start, end).then(setBoxplotData);
    fetchSentimentBoxplot(start, end, 'news').then(setNewsBoxplotData).catch(() => setNewsBoxplotData([]));
    fetchSentimentByMonth(start, end, 'reddit').then(setRedditMonthly);
    fetchSentimentByMonth(start, end, 'news').then(setNewsMonthly).catch(() => setNewsMonthly([]));
  }, [selected]);

  useEffect(() => {
    if (sel.length === 0 || !selected) return;
    const [start, end] = selected;
    Promise.all(sel.map(sub => fetchSentimentBySubredditMonth(sub, start, end)))
      .then(results => {
        const merged: Record<string, any> = {};
        results.forEach((data, i) => {
          const sub = sel[i];
          data.forEach((row: any) => {
            if (!merged[row.year_month]) merged[row.year_month] = { year_month: row.year_month };
            merged[row.year_month][sub] = row.mean_sentiment;
          });
        });
        setTimelineData(Object.values(merged).sort((a, b) => a.year_month.localeCompare(b.year_month)));
      });
  }, [sel, selected]);

  const toggleSubreddit = (sub: string) => {
    setSel(prev =>
      prev.includes(sub) ? prev.filter(s => s !== sub) : [...prev, sub]
    );
  };

  // Merge Reddit vs News monthly sentiment
  const comparisonData: Record<string, any>[] = [];
  if (redditMonthly.length > 0 || newsMonthly.length > 0) {
    const merged: Record<string, any> = {};
    redditMonthly.forEach(d => {
      merged[d.year_month] = { year_month: d.year_month, reddit: d.mean_sentiment };
    });
    newsMonthly.forEach(d => {
      if (!merged[d.year_month]) merged[d.year_month] = { year_month: d.year_month };
      merged[d.year_month].news = d.mean_sentiment;
    });
    comparisonData.push(...Object.values(merged).sort((a, b) => a.year_month.localeCompare(b.year_month)));
  }

  // Top news sources for table
  const topNewsSources = [...newsSources]
    .sort((a: any, b: any) => b.total_count - a.total_count)
    .slice(0, 20);

  return (
    <div className="px-6 py-8 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-[#e8eaed] tracking-tight">Sentiment Analysis</h2>
        <div className="h-[2px] w-10 bg-[#6366f1] mt-2 rounded-full" />
      </div>

      {/* Reddit vs News comparison */}
      {comparisonData.length > 0 && (
        <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
          <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">
            Reddit vs News — Sentiment Over Time
          </h3>
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

      {/* Reddit subreddit selector */}
      <div>
        <PlatformLabel platform="Reddit" color="#6366f1" />
        <div className="flex flex-wrap gap-2 mt-2">
          {subreddits.map(s => (
            <button
              key={s.subreddit}
              onClick={() => toggleSubreddit(s.subreddit)}
              className={`px-3 py-1 text-[12px] rounded-full border transition-colors font-medium ${
                sel.includes(s.subreddit)
                  ? 'bg-[#6366f1]/15 text-[#6366f1] border-[#6366f1]/40'
                  : 'bg-[#1a1d27] text-[#8b8fa3] border-[#2a2e3d] hover:border-[#6366f1]/30 hover:text-[#e8eaed]'
              }`}
            >
              r/{s.subreddit}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
        <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">
          Subreddit Sentiment Comparison — {sel.length} subreddit{sel.length !== 1 && 's'}
        </h3>
        <ResponsiveContainer width="100%" height={360}>
          <LineChart data={timelineData}>
            <CartesianGrid stroke={chartGrid} />
            <XAxis dataKey="year_month" tick={chartTick} axisLine={chartAxisLine} tickLine={false} interval={Math.max(1, Math.floor(timelineData.length / 8))} />
            <YAxis domain={[-0.8, 0.3]} tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
            <Tooltip content={<DarkTooltip />} />
            <Legend wrapperStyle={{ color: '#8b8fa3', fontSize: 12 }} />
            {sel.map((sub, i) => (
              <Line
                key={sub}
                type="monotone"
                dataKey={sub}
                stroke={COLORS[i % COLORS.length]}
                strokeWidth={1.5}
                dot={false}
                name={`r/${sub}`}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

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

      {/* Side-by-side tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
          <div className="flex items-center gap-2 mb-4">
            <PlatformLabel platform="Reddit" color="#6366f1" />
            <h3 className="text-[13px] font-semibold text-[#e8eaed]">Subreddit Overview</h3>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#2a2e3d] text-left text-[11px] text-[#8b8fa3] uppercase tracking-wider">
                <th className="py-2.5 font-medium">Subreddit</th>
                <th className="py-2.5 font-medium">Sentiment</th>
                <th className="py-2.5 font-medium">Pos</th>
                <th className="py-2.5 font-medium">Neg</th>
                <th className="py-2.5 font-medium">Docs</th>
              </tr>
            </thead>
            <tbody>
              {[...subreddits].sort((a, b) => a.mean_sentiment - b.mean_sentiment).map(s => (
                <tr key={s.subreddit} className="border-b border-[#2a2e3d]/50 hover:bg-[#242838] transition-colors">
                  <td className="py-2 font-medium text-[#e8eaed] text-[12px]">r/{s.subreddit}</td>
                  <td className={`py-2 font-mono text-[12px] ${s.mean_sentiment < -0.3 ? 'text-[#f87171]' : s.mean_sentiment > 0 ? 'text-[#34d399]' : 'text-[#64748b]'}`}>
                    {s.mean_sentiment.toFixed(3)}
                  </td>
                  <td className="py-2 text-[#34d399] font-mono text-[12px]">{(s.positive_ratio * 100).toFixed(1)}%</td>
                  <td className="py-2 text-[#f87171] font-mono text-[12px]">{(s.negative_ratio * 100).toFixed(1)}%</td>
                  <td className="py-2 text-[#64748b] font-mono text-[12px]">{s.total_count.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
          <div className="flex items-center gap-2 mb-4">
            <PlatformLabel platform="GDELT News" color="#f59e0b" />
            <h3 className="text-[13px] font-semibold text-[#e8eaed]">Top Sources</h3>
          </div>
          {topNewsSources.length > 0 ? (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#2a2e3d] text-left text-[11px] text-[#8b8fa3] uppercase tracking-wider">
                  <th className="py-2.5 font-medium">Source</th>
                  <th className="py-2.5 font-medium">Sentiment</th>
                  <th className="py-2.5 font-medium">Pos</th>
                  <th className="py-2.5 font-medium">Neg</th>
                  <th className="py-2.5 font-medium">Docs</th>
                </tr>
              </thead>
              <tbody>
                {topNewsSources.map((s: any) => (
                  <tr key={s.source} className="border-b border-[#2a2e3d]/50 hover:bg-[#242838] transition-colors">
                    <td className="py-2 font-medium text-[#e8eaed] text-[12px] truncate max-w-[150px]">{s.source}</td>
                    <td className={`py-2 font-mono text-[12px] ${s.mean_sentiment < -0.3 ? 'text-[#f87171]' : s.mean_sentiment > 0 ? 'text-[#34d399]' : 'text-[#64748b]'}`}>
                      {s.mean_sentiment.toFixed(3)}
                    </td>
                    <td className="py-2 text-[#34d399] font-mono text-[12px]">{(s.positive_ratio * 100).toFixed(1)}%</td>
                    <td className="py-2 text-[#f87171] font-mono text-[12px]">{(s.negative_ratio * 100).toFixed(1)}%</td>
                    <td className="py-2 text-[#64748b] font-mono text-[12px]">{s.total_count.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="flex items-center justify-center h-32 text-[#64748b] text-sm">No news data available</div>
          )}
        </div>
      </div>
    </div>
  );
}
