"use client";

import type { ProgressStep, Source, ValidationResult, GapInfo } from "@/lib/types";

interface Props {
  steps: ProgressStep[];
  sources: Source[];
  totalSources: number;
  gapInfo: GapInfo | null;
  validation: ValidationResult | null;
}

export default function ProgressPanel({
  steps,
  sources,
  totalSources,
  gapInfo,
  validation,
}: Props) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-4xl mx-auto">
      {/* Progress steps */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
        <h3 className="text-sm font-semibold text-gray-300 mb-4">Progress</h3>
        <div className="space-y-3">
          {steps.map((step) => (
            <StepItem key={step.id} step={step} />
          ))}
        </div>

        {/* Gap detection results */}
        {gapInfo && gapInfo.gaps.length > 0 && (
          <div className="mt-4 p-3 bg-amber-900/20 border border-amber-700/40 rounded-lg">
            <p className="text-xs font-medium text-amber-400 mb-1">
              {gapInfo.gaps.length} knowledge gap(s) detected
            </p>
            <ul className="space-y-1">
              {gapInfo.gaps.slice(0, 3).map((g, i) => (
                <li key={i} className="text-xs text-amber-300/70">· {g}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Cross-validation results */}
        {validation && (
          <div className="mt-4 p-3 bg-slate-700/50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-gray-400">Cross-validation</span>
              <span className={`text-xs font-bold ${
                validation.quality_score >= 0.7 ? "text-emerald-400" :
                validation.quality_score >= 0.5 ? "text-amber-400" : "text-red-400"
              }`}>
                {Math.round(validation.quality_score * 100)}%
              </span>
            </div>
            <div className="flex gap-3 text-xs text-gray-500">
              <span>{validation.well_corroborated} corroborated</span>
              {validation.contradictions > 0 && (
                <span className="text-amber-500">{validation.contradictions} contradiction(s)</span>
              )}
            </div>
            {validation.summary && (
              <p className="text-xs text-gray-500 mt-1.5 line-clamp-2">{validation.summary}</p>
            )}
          </div>
        )}
      </div>

      {/* Collected sources */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-gray-300">Collected Sources</h3>
          <span className="text-xs text-indigo-400 font-medium">
            {totalSources}
          </span>
        </div>
        <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
          {sources.length === 0 ? (
            <p className="text-xs text-gray-600 text-center py-4">Searching...</p>
          ) : (
            sources.map((src, i) => <SourceCard key={i} source={src} />)
          )}
        </div>
      </div>
    </div>
  );
}

function StepItem({ step }: { step: ProgressStep }) {
  const icon = {
    pending: <PendingIcon />,
    active: <ActiveIcon />,
    done: <DoneIcon />,
    skipped: <SkippedIcon />,
  }[step.status];

  const labelClass = {
    pending: "text-gray-600",
    active: "text-gray-200 font-medium",
    done: "text-gray-400",
    skipped: "text-gray-700 line-through",
  }[step.status];

  return (
    <div className="flex items-start gap-3">
      <div className="mt-0.5 shrink-0">{icon}</div>
      <div>
        <p className={`text-sm ${labelClass}`}>{step.label}</p>
        {step.detail && step.status === "active" && (
          <p className="text-xs text-gray-600 mt-0.5">{step.detail}</p>
        )}
      </div>
    </div>
  );
}

function SourceCard({ source }: { source: Source }) {
  const trustColor = {
    high: "bg-emerald-900/40 text-emerald-400 border-emerald-700/40",
    medium: "bg-slate-700 text-gray-400 border-slate-600",
    low: "bg-amber-900/20 text-amber-500/70 border-amber-700/30",
  }[source.trust_level] || "bg-slate-700 text-gray-400 border-slate-600";

  const isLocal = source.source_type === "local";

  return (
    <div className="p-2.5 bg-slate-700/50 rounded-lg border border-slate-600/50 hover:border-slate-500 transition-colors">
      <div className="flex items-start justify-between gap-2">
        <a
          href={isLocal ? undefined : source.url}
          target={isLocal ? undefined : "_blank"}
          rel="noopener noreferrer"
          className={`text-xs text-gray-300 line-clamp-2 leading-snug
                     ${!isLocal ? "hover:text-indigo-400 cursor-pointer" : "cursor-default"}`}
        >
          {source.title || "Untitled"}
        </a>
        <div className="flex items-center gap-1 shrink-0">
          {isLocal && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-purple-900/40 text-purple-400 border border-purple-700/40">
              Local
            </span>
          )}
          <span className={`text-xs px-1.5 py-0.5 rounded border ${trustColor}`}>
            {Math.round(source.confidence * 100)}%
          </span>
        </div>
      </div>
    </div>
  );
}

// ── Icons ─────────────────────────────────────────────────────────────

function PendingIcon() {
  return <div className="w-4 h-4 rounded-full border border-slate-600" />;
}

function ActiveIcon() {
  return (
    <div className="relative w-4 h-4">
      <div className="w-4 h-4 rounded-full bg-indigo-500/20 border border-indigo-500 animate-pulse" />
      <div className="absolute inset-1 rounded-full bg-indigo-500" />
    </div>
  );
}

function DoneIcon() {
  return (
    <div className="w-4 h-4 rounded-full bg-emerald-500/20 border border-emerald-500 flex items-center justify-center">
      <svg className="w-2.5 h-2.5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
      </svg>
    </div>
  );
}

function SkippedIcon() {
  return <div className="w-4 h-4 rounded-full border border-slate-700 opacity-40" />;
}
