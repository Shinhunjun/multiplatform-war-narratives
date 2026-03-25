import { useEffect, useState, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  AreaChart, Area, Legend,
} from 'recharts';
import {
  fetchTopicsMonthlyFitted,
  fetchTopicsOverTime, fetchTopicInfo,
} from '../lib/api';
import type { TopicMonthlyFitted, TopicOverTime, TopicInfo } from '../lib/api';
import { useTimeRange } from '../lib/TimeRangeContext';
import { DarkTooltip, PlatformLabel, COLORS, chartGrid, chartTick, chartAxisLine } from '../components/charts/shared';

function parseTopicLabel(name: string): string {
  const parts = name.split('_').slice(1, 3);
  const label = parts.join(', ');
  return label.length > 25 ? label.slice(0, 22) + '...' : label;
}

const PLATFORMS = [
  { key: 'reddit', label: 'Reddit' as const, color: '#6366f1' },
  { key: 'news', label: 'GDELT News' as const, color: '#f59e0b' },
  { key: 'tiktok', label: 'TikTok' as const, color: '#ff0050' },
];

function TopicEvolutionChart({ timeline, topics, selected, title }: { timeline: TopicOverTime[]; topics: TopicInfo[]; selected: [string, string] | null; title: string }) {
  const filteredTimeline = selected
    ? timeline.filter(row => {
        const ts = row.Timestamp.slice(0, 7);
        return ts >= selected[0] && ts <= selected[1];
      })
    : timeline;

  const timelineMap: Record<string, any> = {};
  filteredTimeline.forEach(row => {
    const ts = row.Timestamp.slice(0, 7);
    if (!timelineMap[ts]) timelineMap[ts] = { month: ts };
    timelineMap[ts][`t${row.Topic}`] = row.Frequency;
  });
  const timelineRows = Object.values(timelineMap).sort((a: any, b: any) => a.month.localeCompare(b.month));
  const topTopics = topics.filter(t => t.Topic >= 0).sort((a, b) => b.Count - a.Count).slice(0, 8);

  if (timelineRows.length === 0) {
    return (
      <div className="bg-background-primary rounded-lg border border-background-secondary p-5">
        <h3 className="text-[13px] font-semibold text-text-primary mb-4">{title}</h3>
        <div className="flex items-center justify-center h-[350px] text-text-muted text-sm">No data available</div>
      </div>
    );
  }

  return (
    <div className="bg-background-primary rounded-lg border border-background-secondary p-5">
      <h3 className="text-[13px] font-semibold text-text-primary mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={500}>
        <AreaChart data={timelineRows}>
          <CartesianGrid stroke={chartGrid} />
          <XAxis dataKey="month" tick={chartTick} axisLine={chartAxisLine} tickLine={false} interval={Math.max(1, Math.floor(timelineRows.length / 6))} />
          <YAxis tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
          <Tooltip content={<DarkTooltip />} />
          <Legend wrapperStyle={{ color: '#8b8fa3', fontSize: 10, lineHeight: '14px' }} iconSize={8} />
          {topTopics.map((t, i) => (
            <Area
              key={t.Topic}
              type="monotone"
              dataKey={`t${t.Topic}`}
              stackId="1"
              fill={COLORS[i % COLORS.length]}
              fillOpacity={0.6}
              stroke={COLORS[i % COLORS.length]}
              name={parseTopicLabel(t.Name)}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function MonthlyFittedBarChart({ topics, color, title }: { topics: TopicMonthlyFitted[]; color: string; title: string }) {
  const bars = topics.map(t => ({
    label: t.keywords || `Topic ${t.topic_id}`,
    count: t.count,
    topic_id: t.topic_id,
  }));

  if (bars.length === 0) {
    return (
      <div className="bg-background-primary rounded-lg border border-background-secondary p-5">
        <h3 className="text-[13px] font-semibold text-text-primary mb-4">{title}</h3>
        <div className="flex items-center justify-center h-[350px] text-text-muted text-sm">No topics for this month</div>
      </div>
    );
  }

  return (
    <div className="bg-background-primary rounded-lg border border-background-secondary p-5">
      <h3 className="text-[13px] font-semibold text-text-primary mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={350}>
        <BarChart data={bars} layout="vertical">
          <CartesianGrid stroke={chartGrid} horizontal={false} />
          <XAxis type="number" tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
          <YAxis type="category" dataKey="label" width={160} tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
          <Tooltip content={<DarkTooltip />} />
          <Bar dataKey="count" fill={color} name="Documents" radius={[0, 3, 3, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function TopicsPage() {
  const { range } = useTimeRange();
  const [selectedMonth, setSelectedMonth] = useState<string | null>(null);

  // Set default month when range loads
  useEffect(() => {
    if (range && !selectedMonth) setSelectedMonth(range.allMonths[range.allMonths.length - 1]);
  }, [range, selectedMonth]);

  const [loading, setLoading] = useState(false);
  const [redditFitted, setRedditFitted] = useState<TopicMonthlyFitted[]>([]);
  const [newsFitted, setNewsFitted] = useState<TopicMonthlyFitted[]>([]);
  const [tiktokFitted, setTiktokFitted] = useState<TopicMonthlyFitted[]>([]);

  // Evolution chart data (global model)
  const [redditTopics, setRedditTopics] = useState<TopicInfo[]>([]);
  const [redditTimeline, setRedditTimeline] = useState<TopicOverTime[]>([]);
  const [newsTopics, setNewsTopics] = useState<TopicInfo[]>([]);
  const [newsTimeline, setNewsTimeline] = useState<TopicOverTime[]>([]);
  const [tiktokTopics, setTiktokTopics] = useState<TopicInfo[]>([]);
  const [tiktokTimeline, setTiktokTimeline] = useState<TopicOverTime[]>([]);

  // Load evolution chart data (once)
  useEffect(() => {
    fetchTopicInfo('reddit').then(setRedditTopics);
    fetchTopicsOverTime(undefined, 'reddit').then(setRedditTimeline);
    fetchTopicInfo('news').then(setNewsTopics).catch(() => setNewsTopics([]));
    fetchTopicsOverTime(undefined, 'news').then(setNewsTimeline).catch(() => setNewsTimeline([]));
    fetchTopicInfo('tiktok').then(setTiktokTopics).catch(() => setTiktokTopics([]));
    fetchTopicsOverTime(undefined, 'tiktok').then(setTiktokTimeline).catch(() => setTiktokTimeline([]));
  }, []);

  // Fetch monthly fitted topics when month changes
  const fetchMonthData = useCallback((month: string) => {
    setLoading(true);
    Promise.all([
      fetchTopicsMonthlyFitted(month, 15, 'reddit').catch(() => []),
      fetchTopicsMonthlyFitted(month, 15, 'news').catch(() => []),
      fetchTopicsMonthlyFitted(month, 15, 'tiktok').catch(() => []),
    ]).then(([r, n, t]) => {
      setRedditFitted(r);
      setNewsFitted(n);
      setTiktokFitted(t);
    }).finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (selectedMonth) fetchMonthData(selectedMonth);
  }, [selectedMonth, fetchMonthData]);

  const monthLabel = selectedMonth
    ? new Date(selectedMonth + '-01').toLocaleDateString('en-US', { year: 'numeric', month: 'long' })
    : '';

  return (
    <div className="px-6 py-8 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-text-primary tracking-tight">Topic Modeling (BERTopic)</h2>
        <p className="text-[13px] text-text-secondary mt-1">Independent BERTopic model fitted per month</p>
        <div className="h-[2px] w-10 bg-accent-reddit mt-2 rounded-full" />
      </div>

      {/* Month slider */}
      {range && selectedMonth && (() => {
        const idx = range.allMonths.indexOf(selectedMonth);
        const total = range.allMonths.length - 1;
        return (
          <div className="bg-background-primary rounded-lg border border-background-secondary px-5 py-3">
            <div className="flex items-center gap-3">
              <span className="text-[11px] text-text-secondary font-medium uppercase tracking-wider shrink-0">Month</span>
              <button
                onClick={() => idx > 0 && setSelectedMonth(range.allMonths[idx - 1])}
                disabled={idx <= 0}
                className="w-7 h-7 flex items-center justify-center rounded-md bg-background-secondary text-text-secondary hover:text-text-primary disabled:opacity-30 disabled:cursor-not-allowed transition-colors shrink-0 text-sm font-medium"
              >‹</button>
              <input
                type="range"
                min={0}
                max={total}
                value={idx}
                onChange={e => setSelectedMonth(range.allMonths[Number(e.target.value)])}
                className="flex-1 h-1.5 rounded-lg appearance-none cursor-pointer accent-[#6366f1] bg-[#2a2e3d]"
              />
              <button
                onClick={() => idx < total && setSelectedMonth(range.allMonths[idx + 1])}
                disabled={idx >= total}
                className="w-7 h-7 flex items-center justify-center rounded-md bg-background-secondary text-text-secondary hover:text-text-primary disabled:opacity-30 disabled:cursor-not-allowed transition-colors shrink-0 text-sm font-medium"
              >›</button>
              <span className="text-[13px] text-text-primary font-medium shrink-0 font-mono tabular-nums bg-background-secondary px-3 py-1 rounded-md min-w-[100px] text-center">
                {monthLabel}
              </span>
            </div>
          </div>
        );
      })()}

      {/* Monthly Fitted Topics — only show platforms with data */}
      {loading ? (
        <div className="flex items-center justify-center h-[200px] text-text-muted text-sm">Loading...</div>
      ) : !selectedMonth ? (
        <div className="flex items-center justify-center h-[200px] text-text-muted text-sm">Loading...</div>
      ) : (() => {
        const allFitted = [
          { ...PLATFORMS[0], data: redditFitted },
          { ...PLATFORMS[1], data: newsFitted },
          { ...PLATFORMS[2], data: tiktokFitted },
        ];
        const activeFitted = allFitted.filter(p => p.data.length > 0);
        const cols = activeFitted.length === 1 ? 'lg:grid-cols-1' : activeFitted.length === 2 ? 'lg:grid-cols-2' : 'lg:grid-cols-3';
        return activeFitted.length > 0 ? (
          <div className={`grid grid-cols-1 ${cols} gap-4`}>
            {activeFitted.map(({ key, label, color, data }) => (
              <div key={key}>
                <div className="mb-2"><PlatformLabel platform={label} /></div>
                <MonthlyFittedBarChart topics={data} color={color} title={`Topics — ${label} (${monthLabel})`} />
              </div>
            ))}
          </div>
        ) : (
          <div className="flex items-center justify-center h-[200px] text-text-muted text-sm">No topic data for {monthLabel}</div>
        );
      })()}

      {/* Topic details tables — only show platforms with data */}
      {!loading && selectedMonth && (() => {
        const activeDetails = [
          { ...PLATFORMS[0], data: redditFitted },
          { ...PLATFORMS[1], data: newsFitted },
          { ...PLATFORMS[2], data: tiktokFitted },
        ].filter(p => p.data.length > 0);
        const cols = activeDetails.length === 1 ? 'lg:grid-cols-1' : activeDetails.length === 2 ? 'lg:grid-cols-2' : 'lg:grid-cols-3';
        return activeDetails.length > 0 ? (
        <div className={`grid grid-cols-1 ${cols} gap-4`}>
          {activeDetails.map(({ key, label, data }) => (
            <div key={key} className="bg-background-primary rounded-lg border border-background-secondary p-5">
              <div className="flex items-center gap-2 mb-4">
                <PlatformLabel platform={label} />
                <h3 className="text-[13px] font-semibold text-text-primary">Topic Details — {monthLabel}</h3>
              </div>
              <div className="max-h-[400px] overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-background-primary">
                    <tr className="border-b border-background-secondary text-left text-[11px] text-text-secondary uppercase tracking-wider">
                      <th className="py-2.5 w-12 font-medium">ID</th>
                      <th className="py-2.5 font-medium">Keywords</th>
                      <th className="py-2.5 w-20 font-medium">Docs</th>
                      <th className="py-2.5 w-20 font-medium">Share</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.map(t => (
                      <tr key={t.topic_id} className="border-b border-background-secondary/50 hover:bg-background-secondary/50 transition-colors">
                        <td className="py-2 font-mono text-text-muted text-xs">{t.topic_id}</td>
                        <td className="py-2 text-text-primary text-[12px]">{t.keywords}</td>
                        <td className="py-2 text-text-secondary font-mono text-[12px]">{t.count.toLocaleString()}</td>
                        <td className="py-2 text-text-secondary font-mono text-[12px]">{(t.proportion * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
        ) : null;
      })()}

      {/* Topic Evolution (global model, stacked area) — only show platforms with data */}
      {(() => {
        const activeEvolution = [
          { timeline: redditTimeline, topics: redditTopics, title: 'Topic Evolution (Top 8) — Reddit' },
          { timeline: newsTimeline, topics: newsTopics, title: 'Topic Evolution (Top 8) — News' },
          { timeline: tiktokTimeline, topics: tiktokTopics, title: 'Topic Evolution (Top 8) — TikTok' },
        ].filter(p => p.timeline.length > 0);
        const cols = activeEvolution.length === 1 ? 'lg:grid-cols-1' : 'lg:grid-cols-2';
        return activeEvolution.length > 0 ? (
          <div className={`grid grid-cols-1 ${cols} gap-4`}>
            {activeEvolution.map(({ timeline, topics, title }) => (
              <TopicEvolutionChart key={title} timeline={timeline} topics={topics} selected={null} title={title} />
            ))}
          </div>
        ) : null;
      })()}
    </div>
  );
}
