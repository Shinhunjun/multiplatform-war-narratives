import type { BoxPlotStat } from '../../lib/api';

const ROW_HEIGHT = 32;
const PADDING = { top: 20, right: 30, bottom: 30, left: 100 };

interface Props {
  data: BoxPlotStat[];
  labelPrefix?: string;
}

export default function BoxPlotChart({ data, labelPrefix = 'r/' }: Props) {
  if (data.length === 0) return <p className="text-sm text-[#8b8fa3]">No data</p>;

  const width = 600;
  const height = PADDING.top + PADDING.bottom + data.length * ROW_HEIGHT;

  const allVals = data.flatMap(d => [d.min, d.max]);
  const domainMin = Math.min(...allVals) - 0.05;
  const domainMax = Math.max(...allVals) + 0.05;

  const xScale = (v: number) =>
    PADDING.left + ((v - domainMin) / (domainMax - domainMin)) * (width - PADDING.left - PADDING.right);

  const yScale = (i: number) => PADDING.top + i * ROW_HEIGHT + ROW_HEIGHT / 2;

  const boxH = ROW_HEIGHT * 0.5;

  const tickCount = 6;
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) =>
    domainMin + (i / tickCount) * (domainMax - domainMin)
  );

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ maxHeight: Math.max(height, 200) }}>
      {/* Grid lines */}
      {ticks.map((t, i) => (
        <line
          key={i}
          x1={xScale(t)} x2={xScale(t)}
          y1={PADDING.top - 5} y2={height - PADDING.bottom}
          stroke="#2a2e3d" strokeWidth={1}
        />
      ))}

      {/* X-axis labels */}
      {ticks.map((t, i) => (
        <text
          key={i}
          x={xScale(t)} y={height - PADDING.bottom + 16}
          textAnchor="middle" fontSize={10} fill="#8b8fa3"
          fontFamily="'JetBrains Mono', monospace"
        >
          {t.toFixed(2)}
        </text>
      ))}

      {data.map((d, i) => {
        const cy = yScale(i);
        const x1 = xScale(d.min);
        const xQ1 = xScale(d.q1);
        const xMed = xScale(d.median);
        const xQ3 = xScale(d.q3);
        const x2 = xScale(d.max);
        const xMean = xScale(d.mean);

        return (
          <g key={d.subreddit}>
            {/* Row label */}
            <text
              x={PADDING.left - 8} y={cy + 4}
              textAnchor="end" fontSize={11} fill="#8b8fa3"
            >
              {labelPrefix}{d.subreddit}
            </text>

            {/* Whisker line */}
            <line x1={x1} x2={x2} y1={cy} y2={cy} stroke="#64748b" strokeWidth={1} />

            {/* Min tick */}
            <line x1={x1} x2={x1} y1={cy - boxH / 3} y2={cy + boxH / 3} stroke="#64748b" strokeWidth={1} />
            {/* Max tick */}
            <line x1={x2} x2={x2} y1={cy - boxH / 3} y2={cy + boxH / 3} stroke="#64748b" strokeWidth={1} />

            {/* IQR box */}
            <rect
              x={xQ1} y={cy - boxH / 2}
              width={xQ3 - xQ1} height={boxH}
              fill="#242838" stroke="#6366f1" strokeWidth={1}
              rx={2}
            />

            {/* Median line */}
            <line
              x1={xMed} x2={xMed}
              y1={cy - boxH / 2} y2={cy + boxH / 2}
              stroke="#e8eaed" strokeWidth={2}
            />

            {/* Mean dot */}
            <circle cx={xMean} cy={cy} r={3} fill="#f87171" />

            {/* Tooltip area */}
            <title>
              {`${labelPrefix}${d.subreddit}\nMean: ${d.mean.toFixed(4)} (±${d.std.toFixed(4)})\nMedian: ${d.median.toFixed(4)}\nQ1: ${d.q1.toFixed(4)}, Q3: ${d.q3.toFixed(4)}\nMin: ${d.min.toFixed(4)}, Max: ${d.max.toFixed(4)}\nDocs: ${d.count.toLocaleString()}`}
            </title>
          </g>
        );
      })}

      {/* Legend */}
      <g transform={`translate(${PADDING.left}, ${PADDING.top - 10})`}>
        <rect x={0} y={-4} width={8} height={8} fill="#242838" stroke="#6366f1" strokeWidth={0.5} rx={1} />
        <text x={12} y={3} fontSize={9} fill="#8b8fa3">IQR (Q1–Q3)</text>
        <line x1={70} x2={78} y1={0} y2={0} stroke="#e8eaed" strokeWidth={2} />
        <text x={82} y={3} fontSize={9} fill="#8b8fa3">Median</text>
        <circle cx={126} cy={0} r={3} fill="#f87171" />
        <text x={133} y={3} fontSize={9} fill="#8b8fa3">Mean</text>
      </g>
    </svg>
  );
}
