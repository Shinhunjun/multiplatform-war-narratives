type Platform = 'Reddit' | 'GDELT News' | 'TikTok';

export const COLORS = [
  '#6366f1', '#34d399', '#f87171', '#fbbf24', '#38bdf8',
  '#a78bfa', '#fb923c', '#e879f9', '#2dd4bf', '#f472b6',
];

export const chartGrid = '#2a2e3d';
export const chartTick = { fontSize: 11, fill: '#8b8fa3' };
export const chartAxisLine = { stroke: '#2a2e3d' };

export function DarkTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-background-primary border border-background-secondary rounded-lg px-3 py-2 shadow-xl">
      <p className="text-[11px] text-text-secondary mb-1 font-mono">{label}</p>
      {payload.map((p: any, i: number) => (
        <p key={i} className="text-[12px] font-medium" style={{ color: p.color || '#e8eaed' }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toLocaleString() : p.value}
        </p>
      ))}
    </div>
  );
}

const platformStyles: Record<Platform, { text: string; border: string; bg: string; dot: string }> = {
  Reddit: {
    text: 'text-accent-reddit',
    border: 'border-accent-reddit/25',
    bg: 'bg-accent-reddit/10',
    dot: 'bg-accent-reddit',
  },
  'GDELT News': {
    text: 'text-accent-news',
    border: 'border-accent-news/25',
    bg: 'bg-accent-news/10',
    dot: 'bg-accent-news',
  },
  TikTok: {
    text: 'text-accent-tiktok',
    border: 'border-accent-tiktok/25',
    bg: 'bg-accent-tiktok/10',
    dot: 'bg-accent-tiktok',
  },
};

export function PlatformLabel({ platform }: { platform: Platform }) {
  const styles = platformStyles[platform];

  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-medium border ${styles.text} ${styles.border} ${styles.bg}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${styles.dot}`} />
      {platform}
    </span>
  );
}
