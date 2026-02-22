import { createContext, useContext, useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import { fetchOverview } from './api';

interface TimeRange {
  start: string;
  end: string;
  allMonths: string[];
}

interface TimeRangeContextValue {
  range: TimeRange | null;
  selected: [string, string] | null;
  setSelected: (range: [string, string]) => void;
}

const TimeRangeContext = createContext<TimeRangeContextValue>({
  range: null,
  selected: null,
  setSelected: () => {},
});

export function useTimeRange() {
  return useContext(TimeRangeContext);
}

function generateMonths(start: string, end: string): string[] {
  const months: string[] = [];
  const [sy, sm] = start.split('-').map(Number);
  const [ey, em] = end.split('-').map(Number);
  let y = sy, m = sm;
  while (y < ey || (y === ey && m <= em)) {
    months.push(`${y}-${String(m).padStart(2, '0')}`);
    m++;
    if (m > 12) { m = 1; y++; }
  }
  return months;
}

export function TimeRangeProvider({ children }: { children: ReactNode }) {
  const [range, setRange] = useState<TimeRange | null>(null);
  const [selected, setSelected] = useState<[string, string] | null>(null);

  useEffect(() => {
    fetchOverview().then(stats => {
      const allMonths = generateMonths(stats.date_range.start, stats.date_range.end);
      const r = { start: stats.date_range.start, end: stats.date_range.end, allMonths };
      setRange(r);
      setSelected([r.start, r.end]);
    });
  }, []);

  return (
    <TimeRangeContext.Provider value={{ range, selected, setSelected }}>
      {children}
    </TimeRangeContext.Provider>
  );
}
