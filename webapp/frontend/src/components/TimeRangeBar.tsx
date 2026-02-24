import { useCallback, useRef } from 'react';
import { useTimeRange } from '../lib/TimeRangeContext';

export default function TimeRangeBar() {
  const { range, selected, setSelected } = useTimeRange();
  const trackRef = useRef<HTMLDivElement>(null);

  const getIndex = useCallback((clientX: number) => {
    if (!trackRef.current || !range) return 0;
    const rect = trackRef.current.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    return Math.round(ratio * (range.allMonths.length - 1));
  }, [range]);

  const handleDrag = useCallback((which: 'start' | 'end') => (e: React.MouseEvent) => {
    if (!range || !selected) return;
    e.preventDefault();
    const onMove = (ev: MouseEvent) => {
      const idx = getIndex(ev.clientX);
      const startIdx = range.allMonths.indexOf(selected[0]);
      const endIdx = range.allMonths.indexOf(selected[1]);
      if (which === 'start' && idx < endIdx) {
        setSelected([range.allMonths[idx], selected[1]]);
      } else if (which === 'end' && idx > startIdx) {
        setSelected([selected[0], range.allMonths[idx]]);
      }
    };
    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, [range, selected, setSelected, getIndex]);

  if (!range || !selected) return null;

  const total = range.allMonths.length - 1;
  const startIdx = range.allMonths.indexOf(selected[0]);
  const endIdx = range.allMonths.indexOf(selected[1]);
  const leftPct = (startIdx / total) * 100;
  const widthPct = ((endIdx - startIdx) / total) * 100;

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
            Time Range
          </span>
          <div className="flex-1 relative" ref={trackRef}>
            <div className="h-1.5 bg-[#2a2e3d] rounded-full relative">
              <div
                className="absolute h-full bg-[#6366f1] rounded-full"
                style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
              />
              <div
                className="absolute top-1/2 -translate-y-1/2 w-3.5 h-3.5 bg-[#1a1d27] border-2 border-[#6366f1] rounded-full cursor-ew-resize hover:scale-110 transition-transform"
                style={{ left: `${leftPct}%`, marginLeft: '-7px' }}
                onMouseDown={handleDrag('start')}
              />
              <div
                className="absolute top-1/2 -translate-y-1/2 w-3.5 h-3.5 bg-[#1a1d27] border-2 border-[#6366f1] rounded-full cursor-ew-resize hover:scale-110 transition-transform"
                style={{ left: `${leftPct + widthPct}%`, marginLeft: '-7px' }}
                onMouseDown={handleDrag('end')}
              />
            </div>
          </div>
          <span className="text-[12px] text-[#e8eaed] font-medium shrink-0 font-mono tabular-nums">
            {formatLabel(selected[0])} — {formatLabel(selected[1])}
          </span>
        </div>
      </div>
    </div>
  );
}
