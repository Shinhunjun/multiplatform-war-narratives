import { useTimeRange } from '../lib/TimeRangeContext';

export default function TimeRangeBar() {
  const { range, selectedMonth, setSelectedMonth } = useTimeRange();

  if (!range || !selectedMonth) return null;

  const idx = range.allMonths.indexOf(selectedMonth);
  const total = range.allMonths.length - 1;

  const formatLabel = (ym: string) => {
    const [y, m] = ym.split('-');
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    return `${months[parseInt(m) - 1]} ${y}`;
  };

  return (
    <div className="bg-[#1a1d27] border-b border-[#2a2e3d] px-6 py-3">
      <div className="max-w-[1400px] mx-auto">
        <div className="flex items-center gap-4">
          <span className="text-[11px] text-[#8b8fa3] font-medium uppercase tracking-wider shrink-0">
            Month
          </span>
          <span className="text-[11px] text-[#64748b] shrink-0">{formatLabel(range.allMonths[0])}</span>
          <input
            type="range"
            min={0}
            max={total}
            value={idx}
            onChange={e => setSelectedMonth(range.allMonths[Number(e.target.value)])}
            className="flex-1 h-1.5 rounded-lg appearance-none cursor-pointer accent-[#6366f1] bg-[#2a2e3d]"
          />
          <span className="text-[11px] text-[#64748b] shrink-0">{formatLabel(range.allMonths[total])}</span>
          <span className="text-[12px] text-[#e8eaed] font-medium shrink-0 font-mono tabular-nums bg-[#2a2e3d] px-3 py-1 rounded-md">
            {formatLabel(selectedMonth)}
          </span>
        </div>
      </div>
    </div>
  );
}
