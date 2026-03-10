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
  /** Currently selected single month (YYYY-MM) */
  selectedMonth: string | null;
  setSelectedMonth: (month: string) => void;
  /** [start, end] tuple for backward compat — both equal selectedMonth */
  selected: [string, string] | null;
  setSelected: (range: [string, string]) => void;
}

const TimeRangeContext = createContext<TimeRangeContextValue>({
  range: null,
  selectedMonth: null,
  setSelectedMonth: () => {},
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
  const [selectedMonth, setSelectedMonth] = useState<string | null>(null);

  useEffect(() => {
    fetchOverview().then(stats => {
      const allMonths = generateMonths(stats.date_range.start, stats.date_range.end);
      setRange({ start: stats.date_range.start, end: stats.date_range.end, allMonths });
      // Default to last month
      setSelectedMonth(allMonths[allMonths.length - 1]);
    });
  }, []);

  const selected: [string, string] | null = selectedMonth ? [selectedMonth, selectedMonth] : null;
  const setSelected = (r: [string, string]) => setSelectedMonth(r[1]);

  return (
    <TimeRangeContext.Provider value={{ range, selectedMonth, setSelectedMonth, selected, setSelected }}>
      {children}
    </TimeRangeContext.Provider>
  );
}
