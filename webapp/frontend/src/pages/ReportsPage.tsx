import { useEffect, useState } from 'react';
import { fetchReport, fetchReportList } from '../lib/api';
import type { Report } from '../lib/api';
import { useTimeRange } from '../lib/TimeRangeContext';

// Platform colors
const PLATFORM_COLORS: Record<string, string> = {
  reddit: '#6366f1',
  news: '#f59e0b',
  tiktok: '#ff0050',
};
const PLATFORM_LABELS: Record<string, string> = {
  reddit: 'Reddit',
  news: 'GDELT News',
  tiktok: 'TikTok',
};

export default function ReportsPage() {
  const { selected } = useTimeRange();
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(false);
  const [cachedList, setCachedList] = useState<{ period: string; generated_at: string }[]>([]);
  const [startMonth, setStartMonth] = useState('');
  const [endMonth, setEndMonth] = useState('');
  const [initialized, setInitialized] = useState(false);

  useEffect(() => {
    fetchReportList().then(setCachedList).catch(() => {});
  }, []);

  // Only set defaults once on mount, don't sync with slider afterwards
  useEffect(() => {
    if (selected && !initialized) {
      setStartMonth(selected[0]);
      setEndMonth(selected[1]);
      setInitialized(true);
    }
  }, [selected, initialized]);

  const handleExportPDF = () => {
    const reportEl = document.getElementById('report-content');
    if (!reportEl) return;
    const printWindow = window.open('', '_blank');
    if (!printWindow) return;
    printWindow.document.write(`<!DOCTYPE html><html><head><title>Intelligence Report — ${report?.period || ''}</title>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 40px; color: #1a1a1a; max-width: 800px; margin: 0 auto; line-height: 1.6; }
        h1 { font-size: 22px; border-bottom: 2px solid #333; padding-bottom: 8px; }
        h2 { font-size: 18px; color: #333; margin-top: 24px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
        h3 { font-size: 15px; color: #555; margin-top: 16px; }
        p, li { font-size: 13px; }
        ul, ol { padding-left: 20px; }
        strong { color: #000; }
        .meta { color: #888; font-size: 11px; margin-bottom: 20px; }
        .stats { display: flex; gap: 20px; margin: 16px 0; }
        .stat-box { border: 1px solid #ddd; border-radius: 6px; padding: 12px; flex: 1; }
        .stat-label { font-size: 10px; color: #888; text-transform: uppercase; }
        .stat-value { font-size: 18px; font-weight: 600; }
        @media print { body { padding: 20px; } }
      </style></head><body>
      <h1>Venezuela-US Discourse — Intelligence Report</h1>
      <div class="meta">Period: ${report?.period || ''} &nbsp;|&nbsp; Generated: ${report?.generated_at ? new Date(report.generated_at).toLocaleString() : 'N/A'} &nbsp;|&nbsp; Powered by Gemini</div>
      ${report?.data_summary?.platforms ? '<div class="stats">' + Object.entries(report.data_summary.platforms).map(([k, d]: [string, any]) =>
        `<div class="stat-box"><div class="stat-label">${k}</div><div class="stat-value">${(d.total_docs||0).toLocaleString()}</div><div style="font-size:11px;color:#888">docs, sentiment: ${d.mean_sentiment?.toFixed(3)}</div></div>`
      ).join('') + '</div>' : ''}
      ${(report?.report || '').replace(/## /g, '<h2>').replace(/### /g, '<h3>').replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\n- /g, '\n<li>').replace(/\n/g, '<br/>')}
      </body></html>`);
    printWindow.document.close();
    setTimeout(() => { printWindow.print(); }, 500);
  };

  const handleGenerate = (force = false) => {
    if (!startMonth || !endMonth) return;
    setLoading(true);
    setReport(null);
    fetchReport(startMonth, endMonth, force)
      .then(r => {
        setReport(r);
        fetchReportList().then(setCachedList).catch(() => {});
      })
      .catch(e => setReport({ period: `${startMonth} to ${endMonth}`, error: String(e) }))
      .finally(() => setLoading(false));
  };

  const summary = report?.data_summary;
  const platforms = summary?.platforms || {};

  return (
    <div className="px-6 py-8 space-y-6 max-w-[1200px] mx-auto">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-[#e8eaed] tracking-tight">Intelligence Reports</h2>
        <p className="text-[12px] text-[#8b8fa3] mt-1">AI-generated cross-platform discourse analysis powered by Gemini</p>
        <div className="h-[2px] w-10 bg-[#34d399] mt-2 rounded-full" />
      </div>

      {/* Controls */}
      <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-5">
        <div className="flex items-end gap-4 flex-wrap">
          <div>
            <label className="block text-[11px] text-[#8b8fa3] uppercase tracking-wider mb-1.5 font-medium">Start Month</label>
            <input type="month" value={startMonth} onChange={e => setStartMonth(e.target.value)}
              className="bg-[#0f1117] border border-[#2a2e3d] rounded-lg px-3 py-2 text-[13px] text-[#e8eaed] focus:border-[#34d399] outline-none w-44" />
          </div>
          <div>
            <label className="block text-[11px] text-[#8b8fa3] uppercase tracking-wider mb-1.5 font-medium">End Month</label>
            <input type="month" value={endMonth} onChange={e => setEndMonth(e.target.value)}
              className="bg-[#0f1117] border border-[#2a2e3d] rounded-lg px-3 py-2 text-[13px] text-[#e8eaed] focus:border-[#34d399] outline-none w-44" />
          </div>
          <button onClick={() => handleGenerate(false)} disabled={loading || !startMonth || !endMonth}
            className="px-5 py-2 bg-[#34d399] text-[#0f1117] rounded-lg text-[13px] font-semibold hover:bg-[#2dd4bf] disabled:opacity-50 disabled:cursor-not-allowed transition-colors">
            {loading ? 'Generating...' : 'Generate Report'}
          </button>
          {report && !report.error && (
            <>
              <button onClick={() => handleGenerate(true)} disabled={loading}
                className="px-5 py-2 border border-[#2a2e3d] text-[#8b8fa3] rounded-lg text-[13px] hover:text-[#e8eaed] hover:border-[#64748b] transition-colors">
                Regenerate
              </button>
              <button onClick={handleExportPDF}
                className="px-5 py-2 border border-[#2a2e3d] text-[#8b8fa3] rounded-lg text-[13px] hover:text-[#e8eaed] hover:border-[#64748b] transition-colors flex items-center gap-1.5">
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                </svg>
                Export PDF
              </button>
            </>
          )}
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="bg-[#1a1d27] rounded-lg border border-[#2a2e3d] p-12 text-center">
          <div className="inline-block w-8 h-8 border-2 border-[#34d399] border-t-transparent rounded-full animate-spin mb-4" />
          <p className="text-[#e8eaed] text-sm font-medium">Generating Intelligence Report</p>
          <p className="text-[#64748b] text-xs mt-1">Analyzing cross-platform data with Gemini...</p>
        </div>
      )}

      {/* Error */}
      {report?.error && !loading && (
        <div className="bg-[#1a1d27] rounded-lg border border-[#f87171]/30 p-6">
          <div className="flex items-start gap-3">
            <span className="text-[#f87171] text-lg mt-0.5">!</span>
            <div>
              <p className="text-[#f87171] text-sm font-semibold">Report Generation Failed</p>
              <p className="text-[#8b8fa3] text-[13px] mt-1">{report.error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Report */}
      {report && !report.error && !loading && (
        <div className="space-y-5">
          {/* Report Header */}
          <div className="bg-gradient-to-r from-[#1a1d27] to-[#1e2235] rounded-xl border border-[#2a2e3d] p-6 relative overflow-hidden">
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-[#6366f1] via-[#f59e0b] to-[#ff0050]" />
            <div className="flex items-start justify-between">
              <div>
                <p className="text-[10px] text-[#64748b] uppercase tracking-[0.2em] font-medium">Venezuela-US Discourse</p>
                <h3 className="text-[20px] font-bold text-[#e8eaed] mt-1 tracking-tight">Intelligence Report</h3>
                <p className="text-[14px] text-[#8b8fa3] mt-1 font-mono">{report.period}</p>
              </div>
              <div className="text-right">
                {report.generated_at && (
                  <>
                    <p className="text-[10px] text-[#64748b] uppercase tracking-wider">Generated</p>
                    <p className="text-[12px] text-[#8b8fa3] font-mono mt-0.5">
                      {new Date(report.generated_at).toLocaleString()}
                    </p>
                  </>
                )}
                <p className="text-[10px] text-[#34d399] mt-2 font-medium">Powered by Gemini</p>
              </div>
            </div>
          </div>

          {/* Platform Stats Cards */}
          {Object.keys(platforms).length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {Object.entries(platforms).map(([key, data]: [string, any]) => (
                <div key={key} className="bg-[#1a1d27] rounded-xl border border-[#2a2e3d] p-5 relative overflow-hidden">
                  <div className="absolute top-0 left-0 right-0 h-[2px]" style={{ backgroundColor: PLATFORM_COLORS[key] || '#64748b' }} />
                  <div className="flex items-center gap-2 mb-3">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: PLATFORM_COLORS[key] || '#64748b' }} />
                    <span className="text-[12px] font-semibold text-[#e8eaed]">{PLATFORM_LABELS[key] || key}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <p className="text-[10px] text-[#64748b] uppercase tracking-wider">Documents</p>
                      <p className="text-[18px] font-bold text-[#e8eaed] font-mono">{(data.total_docs || 0).toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-[#64748b] uppercase tracking-wider">Sentiment</p>
                      <p className="text-[18px] font-bold font-mono" style={{
                        color: data.mean_sentiment > 0 ? '#34d399' : data.mean_sentiment < -0.1 ? '#f87171' : '#8b8fa3'
                      }}>
                        {data.mean_sentiment > 0 ? '+' : ''}{data.mean_sentiment?.toFixed(3)}
                      </p>
                    </div>
                    <div>
                      <p className="text-[10px] text-[#64748b] uppercase tracking-wider">Positive</p>
                      <p className="text-[13px] text-[#34d399] font-mono">{((data.positive_ratio || 0) * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-[#64748b] uppercase tracking-wider">Negative</p>
                      <p className="text-[13px] text-[#f87171] font-mono">{((data.negative_ratio || 0) * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                  {/* Top Topics */}
                  {data.top_topics && data.top_topics.length > 0 && (
                    <div className="mt-4 pt-3 border-t border-[#2a2e3d]">
                      <p className="text-[10px] text-[#64748b] uppercase tracking-wider mb-2">Top Topics</p>
                      <div className="space-y-1">
                        {data.top_topics.slice(0, 3).map((t: any, i: number) => (
                          <div key={i} className="flex items-center justify-between">
                            <span className="text-[11px] text-[#c4c8d8] truncate flex-1">{t.keywords}</span>
                            <span className="text-[11px] text-[#64748b] font-mono ml-2">{t.count}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {/* TikTok Hashtags */}
                  {data.top_hashtags && (
                    <div className="mt-4 pt-3 border-t border-[#2a2e3d]">
                      <p className="text-[10px] text-[#64748b] uppercase tracking-wider mb-2">Top Hashtags</p>
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(data.top_hashtags).slice(0, 6).map(([ht, count]: [string, any]) => (
                          <span key={ht} className="px-2 py-0.5 rounded-full bg-[#ff0050]/10 border border-[#ff0050]/20 text-[10px] text-[#ff6b8a]">
                            #{ht} <span className="text-[#64748b]">{count}</span>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {/* TikTok Engagement */}
                  {data.total_views != null && (
                    <div className="mt-3 flex gap-3 text-[10px]">
                      <span className="text-[#64748b]">Views: <span className="text-[#c4c8d8] font-mono">{data.total_views?.toLocaleString()}</span></span>
                      <span className="text-[#64748b]">Likes: <span className="text-[#c4c8d8] font-mono">{data.total_likes?.toLocaleString()}</span></span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Report Body */}
          <div className="bg-[#1a1d27] rounded-xl border border-[#2a2e3d] overflow-hidden">
            {/* Decorative top bar */}
            <div className="px-6 py-4 border-b border-[#2a2e3d] bg-[#161922]">
              <div className="flex items-center gap-2">
                <svg className="w-4 h-4 text-[#34d399]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                </svg>
                <span className="text-[13px] font-semibold text-[#e8eaed]">Analysis Report</span>
              </div>
            </div>
            <div className="px-8 py-6" dangerouslySetInnerHTML={{ __html: renderReport(report.report || '') }} />
          </div>
        </div>
      )}

      {/* Cached reports */}
      {cachedList.length > 0 && !report && !loading && (
        <div className="bg-[#1a1d27] rounded-xl border border-[#2a2e3d] p-5">
          <h3 className="text-[13px] font-semibold text-[#e8eaed] mb-3">Previous Reports</h3>
          <div className="space-y-1">
            {cachedList.map((r, i) => (
              <button key={i}
                onClick={() => {
                  const [s, e] = r.period.split(' to ');
                  setStartMonth(s);
                  setEndMonth(e);
                  setTimeout(() => handleGenerate(false), 50);
                }}
                className="flex items-center justify-between w-full text-left px-4 py-3 rounded-lg hover:bg-[#242838] transition-colors group">
                <div className="flex items-center gap-3">
                  <svg className="w-4 h-4 text-[#64748b] group-hover:text-[#34d399] transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                  </svg>
                  <span className="text-[13px] text-[#e8eaed] font-medium">{r.period}</span>
                </div>
                <span className="text-[11px] text-[#64748b]">{new Date(r.generated_at).toLocaleDateString()}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Render markdown report to styled HTML.
 * Handles ## H2, ### H3, **bold**, - bullet lists, numbered lists, and paragraphs.
 */
function renderReport(md: string): string {
  const lines = md.split('\n');
  const html: string[] = [];
  let inList = false;
  let listType: 'ul' | 'ol' | null = null;

  for (const line of lines) {
    const trimmed = line.trim();

    // Close list if needed
    if (inList && !trimmed.startsWith('- ') && !trimmed.startsWith('* ') && !/^\d+\.\s/.test(trimmed)) {
      html.push(listType === 'ol' ? '</ol>' : '</ul>');
      inList = false;
      listType = null;
    }

    if (!trimmed) {
      if (!inList) html.push('<div class="h-3"></div>');
      continue;
    }

    // H2 — section headers
    if (trimmed.startsWith('## ')) {
      const text = trimmed.slice(3);
      const isExecSummary = text.toLowerCase().includes('executive') || text.toLowerCase().includes('summary');
      const isCross = text.toLowerCase().includes('cross');
      const isSignal = text.toLowerCase().includes('signal') || text.toLowerCase().includes('notable');
      let iconColor = '#6366f1';
      let icon = 'M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6z';
      if (isExecSummary) { iconColor = '#34d399'; icon = 'M11.48 3.499a.562.562 0 011.04 0l2.125 5.111a.563.563 0 00.475.345l5.518.442c.499.04.701.663.321.988l-4.204 3.602a.563.563 0 00-.182.557l1.285 5.385a.562.562 0 01-.84.61l-4.725-2.885a.563.563 0 00-.586 0L6.982 20.54a.562.562 0 01-.84-.61l1.285-5.386a.562.562 0 00-.182-.557l-4.204-3.602a.563.563 0 01.321-.988l5.518-.442a.563.563 0 00.475-.345L11.48 3.5z'; }
      if (isCross) { iconColor = '#a78bfa'; icon = 'M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5'; }
      if (isSignal) { iconColor = '#fbbf24'; icon = 'M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z'; }

      html.push(`
        <div class="mt-8 mb-4 first:mt-0">
          <div class="flex items-center gap-2.5 mb-2">
            <div class="w-7 h-7 rounded-lg flex items-center justify-center" style="background: ${iconColor}15; border: 1px solid ${iconColor}30">
              <svg class="w-3.5 h-3.5" style="color: ${iconColor}" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
                <path stroke-linecap="round" stroke-linejoin="round" d="${icon}" />
              </svg>
            </div>
            <h2 class="text-[16px] font-bold text-[#e8eaed] tracking-tight">${escHtml(text)}</h2>
          </div>
          <div class="h-[1px] bg-gradient-to-r from-[#2a2e3d] to-transparent"></div>
        </div>
      `);
      continue;
    }

    // H3 — platform sub-headers
    if (trimmed.startsWith('### ')) {
      const text = trimmed.slice(4);
      let color = '#8b8fa3';
      if (text.toLowerCase().includes('reddit')) color = '#6366f1';
      else if (text.toLowerCase().includes('news') || text.toLowerCase().includes('gdelt')) color = '#f59e0b';
      else if (text.toLowerCase().includes('tiktok')) color = '#ff0050';

      html.push(`
        <div class="mt-5 mb-2 flex items-center gap-2">
          <span class="w-1.5 h-1.5 rounded-full" style="background: ${color}"></span>
          <h3 class="text-[14px] font-semibold" style="color: ${color}">${escHtml(text)}</h3>
        </div>
      `);
      continue;
    }

    // Bullet list
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      if (!inList || listType !== 'ul') {
        if (inList) html.push(listType === 'ol' ? '</ol>' : '</ul>');
        html.push('<ul class="space-y-1.5 my-2 ml-1">');
        inList = true;
        listType = 'ul';
      }
      const content = applyInline(trimmed.slice(2));
      html.push(`
        <li class="flex items-start gap-2 text-[13px] text-[#c4c8d8] leading-relaxed">
          <span class="w-1 h-1 rounded-full bg-[#64748b] mt-2 flex-shrink-0"></span>
          <span>${content}</span>
        </li>
      `);
      continue;
    }

    // Numbered list
    const numMatch = trimmed.match(/^(\d+)\.\s(.+)/);
    if (numMatch) {
      if (!inList || listType !== 'ol') {
        if (inList) html.push(listType === 'ol' ? '</ol>' : '</ul>');
        html.push('<ol class="space-y-1.5 my-2 ml-1">');
        inList = true;
        listType = 'ol';
      }
      const content = applyInline(numMatch[2]);
      html.push(`
        <li class="flex items-start gap-2.5 text-[13px] text-[#c4c8d8] leading-relaxed">
          <span class="text-[11px] text-[#64748b] font-mono mt-0.5 flex-shrink-0 w-4 text-right">${numMatch[1]}.</span>
          <span>${content}</span>
        </li>
      `);
      continue;
    }

    // Paragraph
    html.push(`<p class="text-[13px] text-[#c4c8d8] leading-[1.8] my-1.5">${applyInline(trimmed)}</p>`);
  }

  if (inList) html.push(listType === 'ol' ? '</ol>' : '</ul>');
  return html.join('\n');
}

function applyInline(text: string): string {
  return escHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong class="text-[#e8eaed] font-semibold">$1</strong>')
    .replace(/`(.+?)`/g, '<code class="text-[#34d399] bg-[#0f1117] px-1.5 py-0.5 rounded text-[12px] font-mono">$1</code>');
}

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
