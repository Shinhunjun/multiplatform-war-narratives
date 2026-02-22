import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  AreaChart, Area, Legend,
} from 'recharts';
import { fetchTopicInfo, fetchTopicsOverTime } from '../lib/api';
import type { TopicInfo, TopicOverTime } from '../lib/api';
import { useTimeRange } from '../lib/TimeRangeContext';

const COLORS = [
  '#6366f1', '#34d399', '#f87171', '#fbbf24', '#38bdf8',
  '#a78bfa', '#fb923c', '#e879f9', '#2dd4bf', '#f472b6',
];

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

function parseTopicLabel(name: string): string {
  const parts = name.split('_').slice(1, 4);
  return parts.join(', ');
}

export default function TopicsPage() {
  const { selected } = useTimeRange();
  const [topics, setTopics] = useState<TopicInfo[]>([]);
  const [timeline, setTimeline] = useState<TopicOverTime[]>([]);

  useEffect(() => {
    fetchTopicInfo().then(setTopics);
  }, []);

  useEffect(() => {
    fetchTopicsOverTime().then(setTimeline);
  }, []);

  const topicBars = topics
    .filter(t => t.Topic >= 0)
    .sort((a, b) => b.Count - a.Count)
    .map(t => ({
      topic: `Topic ${t.Topic}`,
      label: parseTopicLabel(t.Name),
      count: t.Count,
    }));

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

  return (
    <div className="px-6 py-8 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-[#e8eaed] tracking-tight">Topic Modeling (BERTopic)</h2>
        <div className="h-[2px] w-10 bg-[#6366f1] mt-2 rounded-full" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
          <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">Topic Distribution</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={topicBars} layout="vertical">
              <CartesianGrid stroke={chartGrid} horizontal={false} />
              <XAxis type="number" tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
              <YAxis type="category" dataKey="label" width={150} tick={chartTick} axisLine={chartAxisLine} tickLine={false} />
              <Tooltip content={<DarkTooltip />} />
              <Bar dataKey="count" fill="#6366f1" name="Documents" radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
          <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">Topic Evolution (Top 8)</h3>
          <ResponsiveContainer width="100%" height={400}>
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
      </div>

      <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
        <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-4">Topic Details</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[#2a2e3d] text-left text-[11px] text-[#8b8fa3] uppercase tracking-wider">
              <th className="py-2.5 w-16 font-medium">ID</th>
              <th className="py-2.5 font-medium">Top Keywords</th>
              <th className="py-2.5 w-24 font-medium">Documents</th>
            </tr>
          </thead>
          <tbody>
            {topics.sort((a, b) => b.Count - a.Count).map(t => (
              <tr key={t.Topic} className="border-b border-[#2a2e3d]/50 hover:bg-[#242838] transition-colors">
                <td className="py-2.5 font-mono text-[#64748b] text-xs">{t.Topic}</td>
                <td className="py-2.5 text-[#e8eaed]">{parseTopicLabel(t.Name)}</td>
                <td className="py-2.5 text-[#8b8fa3] font-mono text-[13px]">{t.Count.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
