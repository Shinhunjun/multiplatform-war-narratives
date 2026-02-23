import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  AreaChart, Area, Legend,
} from 'recharts';
import { fetchTopicInfo, fetchTopicsOverTime } from '../lib/api';
import type { TopicInfo, TopicOverTime } from '../lib/api';
import { useTimeRange } from '../lib/TimeRangeContext';
import { DarkTooltip, PlatformLabel, COLORS, chartGrid, chartTick, chartAxisLine } from '../components/charts/shared';

function parseTopicLabel(name: string): string {
  const parts = name.split('_').slice(1, 4);
  return parts.join(', ');
}

function TopicDistributionChart({ topics, color, title }: { topics: TopicInfo[]; color: string; title: string }) {
  const topicBars = topics
    .filter(t => t.Topic >= 0)
    .sort((a, b) => b.Count - a.Count)
    .slice(0, 15)
    .map(t => ({
      topic: `Topic ${t.Topic}`,
      label: parseTopicLabel(t.Name),
      count: t.Count,
    }));

  if (topicBars.length === 0) {
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
        <BarChart data={topicBars} layout="vertical">
          <CartesianGrid stroke={chartGrid} horizontal={false} />
          <XAxis type="number" tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
          <YAxis type="category" dataKey="label" width={140} tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
          <Tooltip content={<DarkTooltip />} />
          <Bar dataKey="count" fill={color} name="Documents" radius={[0, 3, 3, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
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

export default function TopicsPage() {
  const { selected } = useTimeRange();
  const [redditTopics, setRedditTopics] = useState<TopicInfo[]>([]);
  const [redditTimeline, setRedditTimeline] = useState<TopicOverTime[]>([]);
  const [newsTopics, setNewsTopics] = useState<TopicInfo[]>([]);
  const [newsTimeline, setNewsTimeline] = useState<TopicOverTime[]>([]);

  useEffect(() => {
    fetchTopicInfo('reddit').then(setRedditTopics);
    fetchTopicsOverTime(undefined, 'reddit').then(setRedditTimeline);
    fetchTopicInfo('news').then(setNewsTopics).catch(() => setNewsTopics([]));
    fetchTopicsOverTime(undefined, 'news').then(setNewsTimeline).catch(() => setNewsTimeline([]));
  }, []);

  const hasNews = newsTopics.length > 0;

  return (
    <div className="px-6 py-8 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-[#e8eaed] tracking-tight">Topic Modeling (BERTopic)</h2>
        <div className="h-[2px] w-10 bg-[#6366f1] mt-2 rounded-full" />
      </div>

      {/* Filter banner */}
      <div className="bg-[#f59e0b]/10 border border-[#f59e0b]/20 rounded-lg px-4 py-2.5 flex items-center gap-2">
        <svg className="w-4 h-4 text-[#f59e0b] flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
        </svg>
        <p className="text-[12px] text-[#f59e0b]">
          Non-English topics (Spanish-dominated Topic 0) filtered from display for clarity.
        </p>
      </div>

      {/* Topic distribution side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div>
          <div className="mb-2"><PlatformLabel platform="Reddit" color="#6366f1" /></div>
          <TopicDistributionChart topics={redditTopics} color="#6366f1" title="Topic Distribution — Reddit" />
        </div>
        <div>
          <div className="mb-2"><PlatformLabel platform="GDELT News" color="#f59e0b" /></div>
          <TopicDistributionChart topics={newsTopics} color="#f59e0b" title="Topic Distribution — News" />
        </div>
      </div>

      {/* Topic evolution side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <TopicEvolutionChart timeline={redditTimeline} topics={redditTopics} selected={selected} title="Topic Evolution (Top 8) — Reddit" />
        {hasNews ? (
          <TopicEvolutionChart timeline={newsTimeline} topics={newsTopics} selected={selected} title="Topic Evolution (Top 8) — News" />
        ) : (
          <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
            <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">Topic Evolution — News</h3>
            <div className="flex items-center justify-center h-[350px] text-[#64748b] text-sm">No news data available</div>
          </div>
        )}
      </div>

      {/* Topic details tables side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
          <div className="flex items-center gap-2 mb-4">
            <PlatformLabel platform="Reddit" color="#6366f1" />
            <h3 className="text-[13px] font-semibold text-[#e8eaed]">Topic Details</h3>
          </div>
          <div className="max-h-[500px] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-[#1a1d27]">
                <tr className="border-b border-[#2a2e3d] text-left text-[11px] text-[#8b8fa3] uppercase tracking-wider">
                  <th className="py-2.5 w-12 font-medium">ID</th>
                  <th className="py-2.5 font-medium">Top Keywords</th>
                  <th className="py-2.5 w-20 font-medium">Docs</th>
                </tr>
              </thead>
              <tbody>
                {[...redditTopics].sort((a, b) => b.Count - a.Count).map(t => (
                  <tr key={t.Topic} className="border-b border-[#2a2e3d]/50 hover:bg-[#242838] transition-colors">
                    <td className="py-2 font-mono text-[#64748b] text-xs">{t.Topic}</td>
                    <td className="py-2 text-[#e8eaed] text-[12px]">{parseTopicLabel(t.Name)}</td>
                    <td className="py-2 text-[#8b8fa3] font-mono text-[12px]">{t.Count.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
          <div className="flex items-center gap-2 mb-4">
            <PlatformLabel platform="GDELT News" color="#f59e0b" />
            <h3 className="text-[13px] font-semibold text-[#e8eaed]">Topic Details</h3>
          </div>
          {newsTopics.length > 0 ? (
            <div className="max-h-[500px] overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-[#1a1d27]">
                  <tr className="border-b border-[#2a2e3d] text-left text-[11px] text-[#8b8fa3] uppercase tracking-wider">
                    <th className="py-2.5 w-12 font-medium">ID</th>
                    <th className="py-2.5 font-medium">Top Keywords</th>
                    <th className="py-2.5 w-20 font-medium">Docs</th>
                  </tr>
                </thead>
                <tbody>
                  {[...newsTopics].sort((a, b) => b.Count - a.Count).map(t => (
                    <tr key={t.Topic} className="border-b border-[#2a2e3d]/50 hover:bg-[#242838] transition-colors">
                      <td className="py-2 font-mono text-[#64748b] text-xs">{t.Topic}</td>
                      <td className="py-2 text-[#e8eaed] text-[12px]">{parseTopicLabel(t.Name)}</td>
                      <td className="py-2 text-[#8b8fa3] font-mono text-[12px]">{t.Count.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="flex items-center justify-center h-32 text-[#64748b] text-sm">No news data available</div>
          )}
        </div>
      </div>
    </div>
  );
}
