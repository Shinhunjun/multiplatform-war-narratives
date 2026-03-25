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
  const [selNews, setSelNews] = useState<string[]>([]);
  const [timelineData, setTimelineData] = useState<Record<string, any>[]>([]);
  const [boxplotData, setBoxplotData] = useState<BoxPlotStat[]>([]);
  const [newsBoxplotData, setNewsBoxplotData] = useState<BoxPlotStat[]>([]);

  const [redditMonthly, setRedditMonthly] = useState<SentimentMonth[]>([]);
  const [newsMonthly, setNewsMonthly] = useState<SentimentMonth[]>([]);
  const [tiktokMonthly, setTiktokMonthly] = useState<SentimentMonth[]>([]);
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
    fetchSentimentByMonth(start, end, 'tiktok').then(setTiktokMonthly).catch(() => setTiktokMonthly([]));
  }, [selected]);

  useEffect(() => {
    if ((sel.length === 0 && selNews.length === 0) || !selected) return;
    const [start, end] = selected;
    const redditFetches = sel.map(sub => fetchSentimentBySubredditMonth(sub, start, end));
    const newsFetches = selNews.map(src => fetchSentimentBySubredditMonth(src, start, end, 'news'));
    Promise.all([...redditFetches, ...newsFetches])
      .then(results => {
        const merged: Record<string, any> = {};
        const allKeys = [...sel, ...selNews.map(s => `news:${s}`)];
        results.forEach((data, i) => {
          const key = allKeys[i];
          data.forEach((row: any) => {
            if (!merged[row.year_month]) merged[row.year_month] = { year_month: row.year_month };
            merged[row.year_month][key] = row.mean_sentiment;
          });
        });
        setTimelineData(Object.values(merged).sort((a, b) => a.year_month.localeCompare(b.year_month)));
      });
  }, [sel, selNews, selected]);

  const toggleSubreddit = (sub: string) => {
    setSel(prev =>
      prev.includes(sub) ? prev.filter(s => s !== sub) : [...prev, sub]
    );
  };

  const toggleNewsSource = (src: string) => {
    setSelNews(prev =>
      prev.includes(src) ? prev.filter(s => s !== src) : [...prev, src]
    );
  };

  // Top news sources for selector (by doc count)
  const topNewsForSelector = [...newsSources]
    .sort((a: any, b: any) => b.total_count - a.total_count)
    .slice(0, 10);

  // Merge Reddit vs News vs TikTok monthly sentiment
  const comparisonData: Record<string, any>[] = [];
  if (redditMonthly.length > 0 || newsMonthly.length > 0 || tiktokMonthly.length > 0) {
    const merged: Record<string, any> = {};
    redditMonthly.forEach(d => {
      merged[d.year_month] = { year_month: d.year_month, reddit: d.mean_sentiment };
    });
    newsMonthly.forEach(d => {
      if (!merged[d.year_month]) merged[d.year_month] = { year_month: d.year_month };
      merged[d.year_month].news = d.mean_sentiment;
    });
    tiktokMonthly.forEach(d => {
      if (!merged[d.year_month]) merged[d.year_month] = { year_month: d.year_month };
      merged[d.year_month].tiktok = d.mean_sentiment;
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
        <h2 className="text-xl font-bold text-text-primary tracking-tight">Sentiment Analysis</h2>
        <div className="h-[2px] w-10 bg-accent-reddit mt-2 rounded-full" />
      </div>

      {/* Reddit vs News comparison — with 3-month moving average */}
      {comparisonData.length > 0 && (() => {
        // Compute 3-month moving average for smoother trends
        const smoothed = comparisonData.map((d, i) => {
          const window = comparisonData.slice(Math.max(0, i - 1), i + 2);
          const rVals = window.map(w => w.reddit).filter((v: any) => v != null);
          const nVals = window.map(w => w.news).filter((v: any) => v != null);
          const tVals = window.map(w => w.tiktok).filter((v: any) => v != null);
          return {
            ...d,
            reddit_smooth: rVals.length ? +(rVals.reduce((a: number, b: number) => a + b, 0) / rVals.length).toFixed(4) : null,
            news_smooth: nVals.length ? +(nVals.reduce((a: number, b: number) => a + b, 0) / nVals.length).toFixed(4) : null,
            tiktok_smooth: tVals.length ? +(tVals.reduce((a: number, b: number) => a + b, 0) / tVals.length).toFixed(4) : null,
          };
        });
        return (
          <div className="bg-background-primary rounded-lg border border-background-secondary p-5">
            <h3 className="text-[13px] font-semibold text-text-primary mb-1">
              Cross-Platform Sentiment Over Time
            </h3>
            <p className="text-[11px] text-text-muted mb-4">Solid lines: 3-month moving average. Faded lines: raw monthly values.</p>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={smoothed}>
                <CartesianGrid stroke={chartGrid} />
                <XAxis dataKey="year_month" tick={chartTick} axisLine={chartAxisLine} tickLine={false} interval={Math.max(1, Math.floor(smoothed.length / 12))} />
                <YAxis domain={[-0.5, 0.4]} tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
                <Tooltip content={<DarkTooltip />} />
                <Legend wrapperStyle={{ color: '#8b8fa3', fontSize: 12 }} />
                {/* Raw (faded) */}
                <Line type="monotone" dataKey="reddit" stroke="#6366f1" strokeWidth={1} strokeOpacity={0.25} dot={false} name="Reddit (raw)" />
                <Line type="monotone" dataKey="news" stroke="#f59e0b" strokeWidth={1} strokeOpacity={0.25} dot={false} name="News (raw)" />
                <Line type="monotone" dataKey="tiktok" stroke="#ff0050" strokeWidth={1} strokeOpacity={0.25} dot={false} name="TikTok (raw)" />
                {/* Smoothed (bold) */}
                <Line type="monotone" dataKey="reddit_smooth" stroke="#6366f1" strokeWidth={2.5} dot={false} name="Reddit (3mo avg)" />
                <Line type="monotone" dataKey="news_smooth" stroke="#f59e0b" strokeWidth={2.5} dot={false} name="News (3mo avg)" />
                <Line type="monotone" dataKey="tiktok_smooth" stroke="#ff0050" strokeWidth={2.5} dot={false} name="TikTok (3mo avg)" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        );
      })()}

      {/* Source selector */}
      <div className="space-y-3">
        <div>
          <PlatformLabel platform="Reddit" />
          <div className="flex flex-wrap gap-2 mt-2">
            {subreddits.map(s => (
              <button
                key={s.subreddit}
                onClick={() => toggleSubreddit(s.subreddit)}
                className={`px-3 py-1 text-[12px] rounded-full border transition-colors font-medium ${
                  sel.includes(s.subreddit)
                    ? 'bg-accent-reddit/10 text-accent-reddit border-accent-reddit/25'
                    : 'bg-background-primary text-text-secondary border-background-secondary hover:border-accent-reddit/25 hover:text-text-primary'
                }`}
              >
                r/{s.subreddit}
              </button>
            ))}
          </div>
        </div>
        {topNewsForSelector.length > 0 && (
          <div>
            <PlatformLabel platform="GDELT News" />
            <div className="flex flex-wrap gap-2 mt-2">
              {topNewsForSelector.map((s: any) => (
                <button
                  key={s.source}
                  onClick={() => toggleNewsSource(s.source)}
                  className={`px-3 py-1 text-[12px] rounded-full border transition-colors font-medium ${
                    selNews.includes(s.source)
                      ? 'bg-accent-news/10 text-accent-news border-accent-news/25'
                      : 'bg-background-primary text-text-secondary border-background-secondary hover:border-accent-news/25 hover:text-text-primary'
                  }`}
                >
                  {s.source}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="bg-background-primary rounded-lg border border-background-secondary p-5">
        <h3 className="text-[13px] font-semibold text-text-primary mb-4">
          Sentiment Comparison — {sel.length + selNews.length} source{sel.length + selNews.length !== 1 ? 's' : ''}
        </h3>
        <ResponsiveContainer width="100%" height={420}>
          <LineChart data={timelineData}>
            <CartesianGrid stroke={chartGrid} />
            <XAxis dataKey="year_month" tick={chartTick} axisLine={chartAxisLine} tickLine={false} interval={Math.max(1, Math.floor(timelineData.length / 12))} />
            <YAxis domain={[-0.5, 0.4]} tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
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
            {selNews.map((src, i) => (
              <Line
                key={`news:${src}`}
                type="monotone"
                dataKey={`news:${src}`}
                stroke={COLORS[(sel.length + i) % COLORS.length]}
                strokeWidth={1.5}
                dot={false}
                strokeDasharray="5 3"
                name={src}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
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

      {/* Side-by-side tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-background-primary rounded-lg border border-background-secondary p-5">
          <div className="flex items-center gap-2 mb-4">
            <PlatformLabel platform="Reddit" />
            <h3 className="text-[13px] font-semibold text-text-primary">Subreddit Overview</h3>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-background-secondary text-left text-[11px] text-text-secondary uppercase tracking-wider">
                <th className="py-2.5 font-medium">Subreddit</th>
                <th className="py-2.5 font-medium">Sentiment</th>
                <th className="py-2.5 font-medium">Pos</th>
                <th className="py-2.5 font-medium">Neg</th>
                <th className="py-2.5 font-medium">Docs</th>
              </tr>
            </thead>
            <tbody>
              {[...subreddits].sort((a, b) => a.mean_sentiment - b.mean_sentiment).map(s => (
                <tr key={s.subreddit} className="border-b border-background-secondary/50 hover:bg-[#242838] transition-colors">
                  <td className="py-2 font-medium text-text-primary text-[12px]">r/{s.subreddit}</td>
                  <td className={`py-2 font-mono text-[12px] ${s.mean_sentiment < -0.3 ? 'text-sentiment-negative' : s.mean_sentiment > 0 ? 'text-sentiment-positive' : 'text-text-muted'}`}>
                    {s.mean_sentiment.toFixed(3)}
                  </td>
                  <td className="py-2 text-sentiment-positive font-mono text-[12px]">{(s.positive_ratio * 100).toFixed(1)}%</td>
                  <td className="py-2 text-sentiment-negative font-mono text-[12px]">{(s.negative_ratio * 100).toFixed(1)}%</td>
                  <td className="py-2 text-text-muted font-mono text-[12px]">{s.total_count.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="bg-background-primary rounded-lg border border-background-secondary p-5">
          <div className="flex items-center gap-2 mb-4">
            <PlatformLabel platform="GDELT News" />
            <h3 className="text-[13px] font-semibold text-text-primary">Top Sources</h3>
          </div>
          {topNewsSources.length > 0 ? (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-background-secondary text-left text-[11px] text-text-secondary uppercase tracking-wider">
                  <th className="py-2.5 font-medium">Source</th>
                  <th className="py-2.5 font-medium">Sentiment</th>
                  <th className="py-2.5 font-medium">Pos</th>
                  <th className="py-2.5 font-medium">Neg</th>
                  <th className="py-2.5 font-medium">Docs</th>
                </tr>
              </thead>
              <tbody>
                {topNewsSources.map((s: any) => (
                  <tr key={s.source} className="border-b border-background-secondary/50 hover:bg-[#242838] transition-colors">
                    <td className="py-2 font-medium text-text-primary text-[12px] truncate max-w-[150px]">{s.source}</td>
                    <td className={`py-2 font-mono text-[12px] ${s.mean_sentiment < -0.3 ? 'text-sentiment-negative' : s.mean_sentiment > 0 ? 'text-sentiment-positive' : 'text-text-muted'}`}>
                      {s.mean_sentiment.toFixed(3)}
                    </td>
                    <td className="py-2 text-sentiment-positive font-mono text-[12px]">{(s.positive_ratio * 100).toFixed(1)}%</td>
                    <td className="py-2 text-sentiment-negative font-mono text-[12px]">{(s.negative_ratio * 100).toFixed(1)}%</td>
                    <td className="py-2 text-text-muted font-mono text-[12px]">{s.total_count.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="flex items-center justify-center h-32 text-text-muted text-sm">No news data available</div>
          )}
        </div>
      </div>
    </div>
  );
}
