import { useRef } from 'react';
import { useTimeRange } from '../lib/TimeRangeContext';

export default function TimeRangeBar() {
  const { range, selected, setSelected } = useTimeRange();
  const trackRef = useRef<HTMLDivElement>(null);

  if (!range || !selected) return null;

  const total = range.allMonths.length - 1;
  const startIdx = Math.max(0, range.allMonths.indexOf(selected[0]));
  const endIdx = Math.max(0, range.allMonths.indexOf(selected[1]));

  const formatLabel = (ym: string) => {
    const [y, m] = ym.split('-');
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    return `${months[parseInt(m) - 1]} ${y}`;
  };

  const handleTrackClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!trackRef.current) return;
    const rect = trackRef.current.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    const idx = Math.round(pct * total);
    // Move whichever thumb is closer
    const distStart = Math.abs(idx - startIdx);
    const distEnd = Math.abs(idx - endIdx);
    if (distStart <= distEnd) {
      if (idx <= endIdx) setSelected([range.allMonths[idx], selected[1]]);
    } else {
      if (idx >= startIdx) setSelected([selected[0], range.allMonths[idx]]);
    }
  };

  const leftPct = (startIdx / total) * 100;
  const rightPct = (endIdx / total) * 100;

  return (
    <div className="bg-[#1a1d27] border-b border-[#2a2e3d] px-6 py-3">
      <div className="max-w-[1400px] mx-auto">
        <div className="flex items-center gap-3">
          <span className="text-[11px] text-[#8b8fa3] font-medium uppercase tracking-wider shrink-0">
            Range
          </span>
          <span className="text-[11px] text-[#64748b] shrink-0">{formatLabel(range.allMonths[0])}</span>

          {/* Custom dual range */}
          <div className="flex-1 relative h-8 flex items-center" ref={trackRef} onClick={handleTrackClick}>
            {/* Track background */}
            <div className="absolute w-full h-1.5 bg-[#2a2e3d] rounded-full" />
            {/* Selected range highlight */}
            <div
              className="absolute h-1.5 bg-[#6366f1]/40 rounded-full"
              style={{ left: `${leftPct}%`, width: `${rightPct - leftPct}%` }}
            />
            {/* Start thumb */}
            <input
              type="range"
              min={0}
              max={total}
              value={startIdx}
              onChange={e => {
                const v = Number(e.target.value);
                if (v <= endIdx) setSelected([range.allMonths[v], selected[1]]);
              }}
              className="absolute w-full h-1.5 appearance-none bg-transparent cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#6366f1] [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-[#e8eaed] [&::-webkit-slider-thumb]:cursor-grab [&::-webkit-slider-thumb]:relative [&::-webkit-slider-thumb]:z-10"
              style={{ pointerEvents: 'none' }}
              onPointerDown={e => { (e.target as HTMLElement).style.pointerEvents = 'auto'; }}
              onPointerUp={e => { (e.target as HTMLElement).style.pointerEvents = 'none'; }}
            />
            {/* End thumb */}
            <input
              type="range"
              min={0}
              max={total}
              value={endIdx}
              onChange={e => {
                const v = Number(e.target.value);
                if (v >= startIdx) setSelected([selected[0], range.allMonths[v]]);
              }}
              className="absolute w-full h-1.5 appearance-none bg-transparent cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[#34d399] [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-[#e8eaed] [&::-webkit-slider-thumb]:cursor-grab [&::-webkit-slider-thumb]:relative [&::-webkit-slider-thumb]:z-10"
              style={{ pointerEvents: 'none' }}
              onPointerDown={e => { (e.target as HTMLElement).style.pointerEvents = 'auto'; }}
              onPointerUp={e => { (e.target as HTMLElement).style.pointerEvents = 'none'; }}
            />
          </div>

          <span className="text-[11px] text-[#64748b] shrink-0">{formatLabel(range.allMonths[total])}</span>
          <span className="text-[12px] text-[#e8eaed] font-medium shrink-0 font-mono tabular-nums bg-[#2a2e3d] px-3 py-1 rounded-md">
            {formatLabel(selected[0])} — {formatLabel(selected[1])}
          </span>
          <button
            onClick={() => setSelected([range.allMonths[0], range.allMonths[total]])}
            className="text-[11px] text-[#64748b] hover:text-[#6366f1] transition-colors shrink-0"
          >Reset</button>
        </div>
      </div>
    </div>
  );
}
