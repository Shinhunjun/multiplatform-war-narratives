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
    <div className="bg-[#1a1d27] border border-[#2a2e3d] rounded-lg px-3 py-2 shadow-xl">
      <p className="text-[11px] text-[#8b8fa3] mb-1 font-mono">{label}</p>
      {payload.map((p: any, i: number) => (
        <p key={i} className="text-[12px] font-medium" style={{ color: p.color || '#e8eaed' }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toLocaleString() : p.value}
        </p>
      ))}
    </div>
  );
}

export function PlatformLabel({ platform, color }: { platform: string; color: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-medium border" style={{ color, borderColor: `${color}40`, backgroundColor: `${color}10` }}>
      <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }} />
      {platform}
    </span>
  );
}
