import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  AreaChart, Area, Legend, LineChart, Line,
} from 'recharts';
import {
  fetchTikTokHashtags, fetchTikTokHashtagsOverTime,
  fetchTikTokEngagement, fetchTikTokRegions,
  fetchSentimentByMonth, fetchTopicsMonthlyFitted, fetchTopicsMonthlyFittedMonths,
} from '../lib/api';
import type {
  HashtagTrend, HashtagOverTime, EngagementMetric, RegionDistribution,
  SentimentMonth, TopicMonthlyFitted,
} from '../lib/api';
import { useTimeRange } from '../lib/TimeRangeContext';
import { DarkTooltip, PlatformLabel, chartGrid, chartTick, chartAxisLine, COLORS } from '../components/charts/shared';

export default function TikTokPage() {
  const { selected, selectedMonth } = useTimeRange();
  const [hashtags, setHashtags] = useState<HashtagTrend[]>([]);
  const [hashtagTimeline, setHashtagTimeline] = useState<HashtagOverTime[]>([]);
  const [engagement, setEngagement] = useState<EngagementMetric[]>([]);
  const [regions, setRegions] = useState<RegionDistribution[]>([]);
  const [sentiment, setSentiment] = useState<SentimentMonth[]>([]);
  const [fittedTopics, setFittedTopics] = useState<TopicMonthlyFitted[]>([]);
  const [availableMonths, setAvailableMonths] = useState<string[]>([]);

  useEffect(() => {
    const [start, end] = selected || [];
    fetchTikTokHashtags(20, start, end).then(setHashtags).catch(() => {});
    fetchTikTokHashtagsOverTime(undefined, 8).then(setHashtagTimeline).catch(() => {});
    fetchTikTokEngagement(start, end).then(setEngagement).catch(() => {});
    fetchTikTokRegions(15, start, end).then(setRegions).catch(() => {});
    fetchSentimentByMonth(start, end, 'tiktok').then(setSentiment).catch(() => {});
    fetchTopicsMonthlyFittedMonths('tiktok').then(setAvailableMonths).catch(() => {});
  }, [selected]);

  useEffect(() => {
    if (selectedMonth && availableMonths.includes(selectedMonth)) {
      fetchTopicsMonthlyFitted(selectedMonth, 10, 'tiktok').then(setFittedTopics).catch(() => {});
    } else if (availableMonths.length > 0) {
      const lastMonth = availableMonths[availableMonths.length - 1];
      fetchTopicsMonthlyFitted(lastMonth, 10, 'tiktok').then(setFittedTopics).catch(() => {});
    }
  }, [selectedMonth, availableMonths]);

  // Pivot hashtag timeline for stacked area
  const htMonths = [...new Set(hashtagTimeline.map(d => d.year_month))].sort();
  const htNames = [...new Set(hashtagTimeline.map(d => d.hashtag))];
  const htAreaData = htMonths.map(m => {
    const row: Record<string, any> = { year_month: m };
    htNames.forEach(h => {
      const match = hashtagTimeline.find(d => d.year_month === m && d.hashtag === h);
      row[h] = match?.count ?? 0;
    });
    return row;
  });

  return (
    <div className="px-6 py-8 space-y-6">
      <div>
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-bold text-text-primary tracking-tight">TikTok Insights</h2>
          <PlatformLabel platform="TikTok" />
        </div>
        <div className="h-[2px] w-10 bg-accent-tiktok mt-2 rounded-full" />
      </div>

      {/* Sentiment + Engagement side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-background-primary rounded-lg border border-background-secondary p-5">
          <h3 className="text-[13px] font-semibold text-text-primary mb-4">Sentiment Over Time</h3>
          {sentiment.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={sentiment}>
                <CartesianGrid stroke={chartGrid} />
                <XAxis dataKey="year_month" tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
                <YAxis domain={[-1, 1]} tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
                <Tooltip content={<DarkTooltip />} />
                <Line type="monotone" dataKey="mean_sentiment" stroke="#ff0050" strokeWidth={2} dot name="Sentiment" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[280px] text-text-muted text-sm">No data</div>
          )}
        </div>

        <div className="bg-background-primary rounded-lg border border-background-secondary p-5">
          <h3 className="text-[13px] font-semibold text-text-primary mb-4">Engagement Metrics</h3>
          {engagement.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={engagement}>
                <CartesianGrid stroke={chartGrid} />
                <XAxis dataKey="year_month" tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
                <YAxis tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
                <Tooltip content={<DarkTooltip />} />
                <Legend wrapperStyle={{ color: '#8b8fa3', fontSize: 11 }} />
                <Bar dataKey="total_views" fill="#38bdf8" name="Views" />
                <Bar dataKey="total_likes" fill="#ff0050" name="Likes" />
                <Bar dataKey="total_shares" fill="#fbbf24" name="Shares" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[280px] text-text-muted text-sm">No data</div>
          )}
        </div>
      </div>

      {/* Hashtag Trends */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-background-primary rounded-lg border border-background-secondary p-5">
          <h3 className="text-[13px] font-semibold text-text-primary mb-4">Top Hashtags</h3>
          {hashtags.length > 0 ? (
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={hashtags} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid stroke={chartGrid} />
                <XAxis type="number" tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
                <YAxis type="category" dataKey="hashtag" tick={chartTick} axisLine={chartAxisLine} tickLine={false} width={75} />
                <Tooltip content={<DarkTooltip />} />
                <Bar dataKey="total_count" fill="#ff0050" name="Count" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[400px] text-text-muted text-sm">No data</div>
          )}
        </div>

        <div className="bg-background-primary rounded-lg border border-background-secondary p-5">
          <h3 className="text-[13px] font-semibold text-text-primary mb-4">Hashtag Trends Over Time</h3>
          {htAreaData.length > 0 ? (
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={htAreaData}>
                <CartesianGrid stroke={chartGrid} />
                <XAxis dataKey="year_month" tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
                <YAxis tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
                <Tooltip content={<DarkTooltip />} />
                <Legend wrapperStyle={{ color: '#8b8fa3', fontSize: 10 }} />
                {htNames.map((name, i) => (
                  <Area key={name} type="monotone" dataKey={name} stackId="1"
                    fill={COLORS[i % COLORS.length]} stroke={COLORS[i % COLORS.length]}
                    fillOpacity={0.6} />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[400px] text-text-muted text-sm">No data</div>
          )}
        </div>
      </div>

      {/* Region + Monthly Topics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-background-primary rounded-lg border border-background-secondary p-5">
          <h3 className="text-[13px] font-semibold text-text-primary mb-4">Video Distribution by Region</h3>
          {regions.length > 0 ? (
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={regions} layout="vertical" margin={{ left: 40 }}>
                <CartesianGrid stroke={chartGrid} />
                <XAxis type="number" tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
                <YAxis type="category" dataKey="region_code" tick={chartTick} axisLine={chartAxisLine} tickLine={false} width={35} />
                <Tooltip content={<DarkTooltip />} />
                <Bar dataKey="total_count" fill="#a78bfa" name="Videos" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[340px] text-text-muted text-sm">No data</div>
          )}
        </div>

        <div className="bg-background-primary rounded-lg border border-background-secondary p-5">
          <h3 className="text-[13px] font-semibold text-text-primary mb-4">
            Monthly Topics {selectedMonth && availableMonths.includes(selectedMonth) ? `— ${selectedMonth}` : availableMonths.length > 0 ? `— ${availableMonths[availableMonths.length - 1]}` : ''}
          </h3>
          {fittedTopics.length > 0 ? (
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={fittedTopics} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid stroke={chartGrid} />
                <XAxis type="number" tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
                <YAxis type="category" dataKey="keywords" tick={chartTick} axisLine={chartAxisLine} tickLine={false} width={115} />
                <Tooltip content={<DarkTooltip />} />
                <Bar dataKey="count" fill="#ff0050" name="Documents" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[340px] text-text-muted text-sm">No topic data for this month</div>
          )}
        </div>
      </div>
    </div>
  );
}
