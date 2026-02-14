import { useEffect, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts';
import { fetchSentimentBySubredditMonth, fetchSentimentBySubreddit } from '../lib/api';
import type { SentimentSubreddit } from '../lib/api';

const COLORS = [
  '#e63946', '#1e3a5f', '#2a9d8f', '#f4a261', '#264653',
  '#e76f51', '#457b9d', '#9467bd', '#bcbd22', '#17becf', '#d62728',
];

export default function SentimentPage() {
  const [subreddits, setSubreddits] = useState<SentimentSubreddit[]>([]);
  const [selected, setSelected] = useState<string[]>(['worldnews', 'politics', 'venezuela', 'vzla']);
  const [timelineData, setTimelineData] = useState<Record<string, any>[]>([]);

  useEffect(() => {
    fetchSentimentBySubreddit().then(setSubreddits);
  }, []);

  useEffect(() => {
    if (selected.length === 0) return;
    Promise.all(selected.map(sub => fetchSentimentBySubredditMonth(sub)))
      .then(results => {
        // Merge into single timeline
        const merged: Record<string, any> = {};
        results.forEach((data, i) => {
          const sub = selected[i];
          data.forEach((row: any) => {
            if (!merged[row.year_month]) merged[row.year_month] = { year_month: row.year_month };
            merged[row.year_month][sub] = row.mean_sentiment;
          });
        });
        setTimelineData(Object.values(merged).sort((a, b) => a.year_month.localeCompare(b.year_month)));
      });
  }, [selected]);

  const toggleSubreddit = (sub: string) => {
    setSelected(prev =>
      prev.includes(sub) ? prev.filter(s => s !== sub) : [...prev, sub]
    );
  };

  return (
    <div className="p-6 space-y-6">
      <h2 className="text-xl font-bold text-slate-800">Sentiment Analysis</h2>

      {/* Subreddit Selector */}
      <div className="flex flex-wrap gap-2">
        {subreddits.map(s => (
          <button
            key={s.subreddit}
            onClick={() => toggleSubreddit(s.subreddit)}
            className={`px-3 py-1 text-xs rounded-full border transition ${
              selected.includes(s.subreddit)
                ? 'bg-slate-800 text-white border-slate-800'
                : 'bg-white text-slate-600 border-slate-300 hover:border-slate-500'
            }`}
          >
            r/{s.subreddit}
          </button>
        ))}
      </div>

      {/* Multi-Subreddit Sentiment Timeline */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">
          Sentiment Comparison ({selected.length} subreddits)
        </h3>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={timelineData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year_month" tick={{ fontSize: 10 }} interval={11} />
            <YAxis domain={[-0.8, 0.3]} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend />
            {selected.map((sub, i) => (
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

      {/* Sentiment Distribution Table */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">Subreddit Sentiment Overview</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-slate-500">
              <th className="py-2">Subreddit</th>
              <th className="py-2">Mean Sentiment</th>
              <th className="py-2">Positive %</th>
              <th className="py-2">Negative %</th>
              <th className="py-2">Documents</th>
            </tr>
          </thead>
          <tbody>
            {[...subreddits].sort((a, b) => a.mean_sentiment - b.mean_sentiment).map(s => (
              <tr key={s.subreddit} className="border-b hover:bg-slate-50">
                <td className="py-2 font-medium">r/{s.subreddit}</td>
                <td className={`py-2 ${s.mean_sentiment < -0.3 ? 'text-red-600' : s.mean_sentiment > 0 ? 'text-green-600' : 'text-slate-600'}`}>
                  {s.mean_sentiment.toFixed(3)}
                </td>
                <td className="py-2 text-green-600">{(s.positive_ratio * 100).toFixed(1)}%</td>
                <td className="py-2 text-red-600">{(s.negative_ratio * 100).toFixed(1)}%</td>
                <td className="py-2 text-slate-500">{s.total_count.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
