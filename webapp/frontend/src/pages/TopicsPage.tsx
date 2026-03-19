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
  const parts = name.split('_').slice(1, 4);
  return parts.join(', ');
}

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
      <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
        <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">{title}</h3>
        <div className="flex items-center justify-center h-[350px] text-[#64748b] text-sm">No data available</div>
      </div>
    );
  }

  return (
    <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
      <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={350}>
        <AreaChart data={timelineRows}>
          <CartesianGrid stroke={chartGrid} />
          <XAxis dataKey="month" tick={chartTick} axisLine={chartAxisLine} tickLine={false} interval={Math.max(1, Math.floor(timelineRows.length / 8))} />
          <YAxis tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
          <Tooltip content={<DarkTooltip />} />
          <Legend wrapperStyle={{ color: '#8b8fa3', fontSize: 12 }} />
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
      <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
        <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">{title}</h3>
        <div className="flex items-center justify-center h-[350px] text-[#64748b] text-sm">No topics for this month</div>
      </div>
    );
  }

  return (
    <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
      <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">{title}</h3>
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
  const { selectedMonth } = useTimeRange();

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
        <h2 className="text-xl font-bold text-[#e8eaed] tracking-tight">Topic Modeling (BERTopic)</h2>
        <p className="text-[13px] text-[#8b8fa3] mt-1">
          Independent BERTopic model fitted per month — use the slider above to explore
        </p>
        <div className="h-[2px] w-10 bg-[#6366f1] mt-2 rounded-full" />
      </div>

      {/* Monthly Fitted Topics — Reddit & News side by side */}
      {loading ? (
        <div className="flex items-center justify-center h-[200px] text-[#64748b] text-sm">Loading...</div>
      ) : !selectedMonth ? (
        <div className="flex items-center justify-center h-[200px] text-[#64748b] text-sm">Loading...</div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div>
            <div className="mb-2"><PlatformLabel platform="Reddit" color="#6366f1" /></div>
            <MonthlyFittedBarChart topics={redditFitted} color="#6366f1" title={`Topics — Reddit (${monthLabel})`} />
          </div>
          <div>
            <div className="mb-2"><PlatformLabel platform="GDELT News" color="#f59e0b" /></div>
            <MonthlyFittedBarChart topics={newsFitted} color="#f59e0b" title={`Topics — News (${monthLabel})`} />
          </div>
          <div>
            <div className="mb-2"><PlatformLabel platform="TikTok" color="#ff0050" /></div>
            <MonthlyFittedBarChart topics={tiktokFitted} color="#ff0050" title={`Topics — TikTok (${monthLabel})`} />
          </div>
        </div>
      )}

      {/* Topic details tables */}
      {!loading && selectedMonth && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {[
            { label: 'Reddit', color: '#6366f1', data: redditFitted },
            { label: 'GDELT News', color: '#f59e0b', data: newsFitted },
            { label: 'TikTok', color: '#ff0050', data: tiktokFitted },
          ].map(({ label, color, data }) => (
            <div key={label} className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
              <div className="flex items-center gap-2 mb-4">
                <PlatformLabel platform={label} color={color} />
                <h3 className="text-[13px] font-semibold text-[#e8eaed]">Topic Details — {monthLabel}</h3>
              </div>
              {data.length > 0 ? (
                <div className="max-h-[400px] overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 bg-[#1a1d27]">
                      <tr className="border-b border-[#2a2e3d] text-left text-[11px] text-[#8b8fa3] uppercase tracking-wider">
                        <th className="py-2.5 w-12 font-medium">ID</th>
                        <th className="py-2.5 font-medium">Keywords</th>
                        <th className="py-2.5 w-20 font-medium">Docs</th>
                        <th className="py-2.5 w-20 font-medium">Share</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.map(t => (
                        <tr key={t.topic_id} className="border-b border-[#2a2e3d]/50 hover:bg-[#242838] transition-colors">
                          <td className="py-2 font-mono text-[#64748b] text-xs">{t.topic_id}</td>
                          <td className="py-2 text-[#e8eaed] text-[12px]">{t.keywords}</td>
                          <td className="py-2 text-[#8b8fa3] font-mono text-[12px]">{t.count.toLocaleString()}</td>
                          <td className="py-2 text-[#8b8fa3] font-mono text-[12px]">{(t.proportion * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="flex items-center justify-center h-32 text-[#64748b] text-sm">No data for this month</div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Topic Evolution (global model, stacked area) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <TopicEvolutionChart timeline={redditTimeline} topics={redditTopics} selected={null} title="Topic Evolution (Top 8) — Reddit" />
        <TopicEvolutionChart timeline={newsTimeline} topics={newsTopics} selected={null} title="Topic Evolution (Top 8) — News" />
        <TopicEvolutionChart timeline={tiktokTimeline} topics={tiktokTopics} selected={null} title="Topic Evolution (Top 8) — TikTok" />
      </div>
    </div>
  );
}
