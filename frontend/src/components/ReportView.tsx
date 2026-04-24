"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
  report: string;
  streaming?: boolean;
  totalSources: number;
  qualityScore?: number;
  sessionId: string;
}

export default function ReportView({
  report,
  streaming,
  totalSources,
  qualityScore,
  sessionId,
}: Props) {
  const handleCopy = () => {
    navigator.clipboard.writeText(report);
  };

  const handleDownload = () => {
    const blob = new Blob([report], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `research-${sessionId.slice(0, 8)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      {/* Report header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-4">
          <h2 className="text-sm font-semibold text-gray-300">Research Report</h2>
          <div className="flex items-center gap-3 text-xs text-gray-500">
            <span>{totalSources} sources</span>
            {qualityScore !== undefined && (
              <span className={
                qualityScore >= 0.7 ? "text-emerald-400" :
                qualityScore >= 0.5 ? "text-amber-400" : "text-red-400"
              }>
                Quality {Math.round(qualityScore * 100)}%
              </span>
            )}
          </div>
        </div>
        {!streaming && report && (
          <div className="flex gap-2">
            <button
              onClick={handleCopy}
              className="text-xs text-gray-500 hover:text-gray-300 px-2 py-1
                         rounded-lg hover:bg-slate-700 transition-colors"
            >
              Copy
            </button>
            <button
              onClick={handleDownload}
              className="text-xs text-gray-500 hover:text-gray-300 px-2 py-1
                         rounded-lg hover:bg-slate-700 transition-colors"
            >
              Download .md
            </button>
          </div>
        )}
      </div>

      {/* Report body */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
        {report ? (
          <div className={`prose prose-sm max-w-none ${streaming ? "cursor-blink" : ""}`}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {report}
            </ReactMarkdown>
          </div>
        ) : (
          <div className="flex items-center gap-3 text-gray-500 py-8 justify-center">
            <Spinner />
            <span className="text-sm">Writing the report...</span>
          </div>
        )}
      </div>
    </div>
  );
}

function Spinner() {
  return (
    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}
