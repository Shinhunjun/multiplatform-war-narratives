import type { OverviewStats } from '../lib/api';
import { PlatformLabel } from './charts/shared';

const NewsIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="h-8 w-8 text-text-muted">
    <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" x2="8" y1="13" y2="13" />
    <line x1="16" x2="8" y1="17" y2="17" />
    <line x1="10" x2="8" y1="9" y2="9" />
  </svg>
);

const TikTokIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="h-8 w-8 text-text-muted">
    <path d="M21.6 8a2.6 2.6 0 0 1-2.8 3.3 2.8 2.8 0 0 1-2.6-2.4v7.4a4 4 0 1 1-4-4V8a2.6 2.6 0 0 1-2.8 3.3A2.8 2.8 0 0 1 7 8.9" />
    <path d="M12 18a4 4 0 1 0 4-4" />
  </svg>
);

function StatCard({ label, value, sub, accentClass }: { label: string; value: string | number; sub?: string; accentClass?: string }) {
  return (
    <div className="bg-background-primary rounded-lg border border-background-secondary p-5 relative overflow-hidden">
      <div className={`absolute top-0 left-0 right-0 h-[2px] ${accentClass || 'bg-accent-reddit'}`} />
      <p className="text-[11px] text-text-secondary uppercase tracking-wider font-medium">{label}</p>
      <p className="text-2xl font-semibold text-text-primary mt-1.5 font-mono">{value}</p>
      {sub && <p className="text-[11px] text-text-muted mt-1">{sub}</p>}
    </div>
  );
}

type PlatformName = 'Reddit' | 'GDELT News' | 'TikTok';

interface PlatformStatsProps {
  platform: PlatformName;
  stats: OverviewStats | null;
}

export function PlatformStats({ platform, stats }: PlatformStatsProps) {
  const platformConfig = {
    'Reddit': {
      accentClass: 'bg-accent-reddit',
      cards: [
        { label: "Documents", value: (s: OverviewStats) => s.total_documents.toLocaleString(), sub: "Posts & comments" },
        { label: "Subreddits", value: (s: OverviewStats) => s.subreddits ?? 0, sub: (s: OverviewStats) => `${s.date_range.start} — ${s.date_range.end}` },
        { label: "Topics", value: (s: OverviewStats) => s.num_topics, sub: "BERTopic clusters" },
        { label: "Avg Sentiment", value: (s: OverviewStats) => s.avg_sentiment.toFixed(3), sub: "Mean across subreddits" },
      ],
      icon: null,
    },
    'GDELT News': {
      accentClass: 'bg-accent-news',
      cards: [
        { label: "Documents", value: (s: OverviewStats) => s.total_documents.toLocaleString(), sub: "News articles" },
        { label: "Sources", value: (s: OverviewStats) => s.sources ?? 0, sub: (s: OverviewStats) => `${s.date_range.start} — ${s.date_range.end}` },
        { label: "Topics", value: (s: OverviewStats) => s.num_topics, sub: "Mapped from Reddit" },
        { label: "Avg Sentiment", value: (s: OverviewStats) => s.avg_sentiment.toFixed(3), sub: "Mean across sources" },
      ],
      icon: <NewsIcon />,
    },
    'TikTok': {
      accentClass: 'bg-accent-tiktok',
      cards: [
        { label: "Documents", value: (s: OverviewStats) => s.total_documents.toLocaleString(), sub: (s: OverviewStats) => `${s.total_videos ?? 0} videos, ${s.total_comments ?? 0} comments` },
        { label: "Creators", value: (s: OverviewStats) => s.num_sources ?? s.sources ?? 0, sub: (s: OverviewStats) => `${s.date_range.start} — ${s.date_range.end}` },
        { label: "Topics", value: (s: OverviewStats) => s.num_topics, sub: "BERTopic fitted" },
        { label: "Avg Sentiment", value: (s: OverviewStats) => s.avg_sentiment.toFixed(3), sub: "Mean across creators" },
      ],
      icon: <TikTokIcon />,
    },
  };

  const config = platformConfig[platform];

  return (
    <div className="space-y-3">
      <PlatformLabel platform={platform} />
      {stats ? (
        <div className="grid grid-cols-2 gap-3">
          {config.cards.map(card => (
            <StatCard
              key={card.label}
              label={card.label}
              value={card.value(stats)}
              sub={typeof card.sub === 'function' ? card.sub(stats) : card.sub}
              accentClass={config.accentClass}
            />
          ))}
        </div>
      ) : (
        <div className="bg-background-primary rounded-lg border border-background-secondary p-8 flex flex-col items-center justify-center text-center h-full">
          {config.icon}
          <p className="text-text-secondary text-sm mt-2">{platform} data not available yet.</p>
        </div>
      )}
    </div>
  );
}
