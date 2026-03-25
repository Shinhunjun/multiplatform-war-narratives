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
  /** Currently selected single month (YYYY-MM) — used for monthly-fitted views */
  selectedMonth: string | null;
  setSelectedMonth: (month: string) => void;
  /** [start, end] range — used for sentiment, clusters, dashboard */
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
  const [selected, setSelected] = useState<[string, string] | null>(null);

  useEffect(() => {
    fetchOverview().then(stats => {
      const allMonths = generateMonths(stats.date_range.start, stats.date_range.end);
      setRange({ start: stats.date_range.start, end: stats.date_range.end, allMonths });
      // Default: last month for single-month views, full range for range views
      setSelectedMonth(allMonths[allMonths.length - 1]);
      setSelected([allMonths[0], allMonths[allMonths.length - 1]]);
    });
  }, []);

  // Keep selectedMonth in sync: when range changes, pick the end of range
  const handleSetSelected = (r: [string, string]) => {
    setSelected(r);
    setSelectedMonth(r[1]);
  };

  const handleSetMonth = (m: string) => {
    setSelectedMonth(m);
    // Don't change the range — single month selection is independent
  };

  return (
    <TimeRangeContext.Provider value={{
      range,
      selectedMonth,
      setSelectedMonth: handleSetMonth,
      selected,
      setSelected: handleSetSelected,
    }}>
      {children}
    </TimeRangeContext.Provider>
  );
}
