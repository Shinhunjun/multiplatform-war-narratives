import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  AreaChart, Area, Legend,
} from 'recharts';
import { fetchTopicInfo, fetchTopicsOverTime } from '../lib/api';
import type { TopicInfo, TopicOverTime } from '../lib/api';

const COLORS = [
  '#e63946', '#1e3a5f', '#2a9d8f', '#f4a261', '#264653',
  '#e76f51', '#457b9d', '#9467bd', '#bcbd22', '#17becf',
  '#d62728', '#ff7f0e', '#1f77b4', '#aec7e8',
];

function parseTopicLabel(name: string): string {
  // "0_que es_es que_que el_esto" -> "que es, es que, que el"
  const parts = name.split('_').slice(1, 4);
  return parts.join(', ');
}

export default function TopicsPage() {
  const [topics, setTopics] = useState<TopicInfo[]>([]);
  const [timeline, setTimeline] = useState<TopicOverTime[]>([]);

  useEffect(() => {
    fetchTopicInfo().then(setTopics);
    fetchTopicsOverTime().then(setTimeline);
  }, []);

  // Topic distribution bar chart
  const topicBars = topics
    .filter(t => t.Topic >= 0)
    .sort((a, b) => b.Count - a.Count)
    .map(t => ({
      topic: `Topic ${t.Topic}`,
      label: parseTopicLabel(t.Name),
      count: t.Count,
    }));

  // Topic timeline: pivot to {timestamp, topic0: freq, topic1: freq, ...}
  const timelineMap: Record<string, any> = {};
  timeline.forEach(row => {
    const ts = row.Timestamp.slice(0, 7); // YYYY-MM
    if (!timelineMap[ts]) timelineMap[ts] = { month: ts };
    timelineMap[ts][`t${row.Topic}`] = row.Frequency;
  });
  const timelineRows = Object.values(timelineMap).sort((a: any, b: any) => a.month.localeCompare(b.month));

  const topTopics = topics.filter(t => t.Topic >= 0).sort((a, b) => b.Count - a.Count).slice(0, 8);

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-xl font-bold text-slate-800">Topic Modeling (BERTopic)</h2>

      {/* Topic Distribution */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Topic Distribution</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={topicBars} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="label" width={150} tick={{ fontSize: 10 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#1e3a5f" name="Documents" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Topic Evolution Over Time */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Topic Evolution (Top 8)</h3>
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={timelineRows}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 10 }} interval={5} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend />
            {topTopics.map((t, i) => (
              <Area
                key={t.Topic}
                type="monotone"
                dataKey={`t${t.Topic}`}
                stackId="1"
                fill={COLORS[i % COLORS.length]}
                stroke={COLORS[i % COLORS.length]}
                name={parseTopicLabel(t.Name)}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Topic Details Table */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Topic Details</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-slate-500">
              <th className="py-2 w-16">ID</th>
              <th className="py-2">Top Keywords</th>
              <th className="py-2 w-24">Documents</th>
            </tr>
          </thead>
          <tbody>
            {topics.sort((a, b) => b.Count - a.Count).map(t => (
              <tr key={t.Topic} className="border-b hover:bg-slate-50">
                <td className="py-2 font-mono text-slate-500">{t.Topic}</td>
                <td className="py-2">{parseTopicLabel(t.Name)}</td>
                <td className="py-2 text-slate-600">{t.Count.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
