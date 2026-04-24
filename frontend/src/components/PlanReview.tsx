"use client";

import { useState } from "react";
import type { Plan, SubQuery } from "@/lib/types";

interface Props {
  plan: Plan;
  onApprove: (plan: Plan, reportLength: string) => void;
  onReject: () => void;
  loading: boolean;
}

const INTENT_LABELS: Record<string, string> = {
  factual: "Fact Check",
  analytical: "Analysis",
  comparative: "Comparison",
  predictive: "Prediction",
};

const DEPTH_OPTIONS = [
  { value: "fast",   label: "Fast",   desc: "3 sources/query",  color: "text-emerald-400 bg-emerald-900/30 border-emerald-800/50" },
  { value: "normal", label: "Normal", desc: "5 sources/query",  color: "text-indigo-400 bg-indigo-900/30 border-indigo-800/50" },
  { value: "deep",   label: "Deep",   desc: "8 sources/query",  color: "text-amber-400 bg-amber-900/30 border-amber-800/50" },
];

const DEPTH_COLORS: Record<string, string> = {
  fast: "text-emerald-400 bg-emerald-900/30",
  normal: "text-indigo-400 bg-indigo-900/30",
  deep: "text-amber-400 bg-amber-900/30",
};

const REPORT_LENGTH_OPTIONS = [
  {
    value: "brief",
    label: "Brief",
    desc: "1 call · ~500 chars",
    color: "text-sky-400 bg-sky-900/30 border-sky-800/50",
  },
  {
    value: "standard",
    label: "Standard",
    desc: "3 calls · ~3,000 chars",
    color: "text-violet-400 bg-violet-900/30 border-violet-800/50",
  },
  {
    value: "detailed",
    label: "Detailed",
    desc: "Full sections · ~15,000+ chars",
    color: "text-rose-400 bg-rose-900/30 border-rose-800/50",
  },
];

export default function PlanReview({ plan, onApprove, onReject, loading }: Props) {
  const [editedPlan, setEditedPlan] = useState<Plan>(plan);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");
  const [reportLength, setReportLength] = useState("detailed");

  const startEdit = (sq: SubQuery) => {
    setEditingId(sq.id);
    setEditText(sq.question);
  };

  const saveEdit = () => {
    if (!editingId) return;
    setEditedPlan((prev) => ({
      ...prev,
      sub_queries: prev.sub_queries.map((sq) =>
        sq.id === editingId ? { ...sq, question: editText } : sq
      ),
    }));
    setEditingId(null);
  };

  const removeSubQuery = (id: string) => {
    setEditedPlan((prev) => ({
      ...prev,
      sub_queries: prev.sub_queries.filter((sq) => sq.id !== id),
    }));
  };

  return (
    <div className="w-full max-w-2xl mx-auto space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-100">Research Plan Review</h2>
        <span className="text-xs text-gray-500">{editedPlan.estimated_time}</span>
      </div>

      {/* Research depth selection */}
      <div>
        <p className="text-xs text-gray-500 mb-2 uppercase tracking-wide font-medium">Research Depth</p>
        <div className="grid grid-cols-3 gap-2">
          {DEPTH_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setEditedPlan((p) => ({ ...p, depth: opt.value }))}
              className={`py-2 px-3 rounded-lg border text-xs font-medium transition-all ${
                editedPlan.depth === opt.value
                  ? opt.color
                  : "text-gray-500 bg-slate-800 border-slate-700 hover:border-slate-600"
              }`}
            >
              <span className="block font-semibold">{opt.label}</span>
              <span className="block opacity-70">{opt.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Report length selection */}
      <div>
        <p className="text-xs text-gray-500 mb-2 uppercase tracking-wide font-medium">Report Length</p>
        <div className="grid grid-cols-3 gap-2">
          {REPORT_LENGTH_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => setReportLength(opt.value)}
              className={`py-2 px-3 rounded-lg border text-xs font-medium transition-all ${
                reportLength === opt.value
                  ? opt.color
                  : "text-gray-500 bg-slate-800 border-slate-700 hover:border-slate-600"
              }`}
            >
              <span className="block font-semibold">{opt.label}</span>
              <span className="block opacity-70">{opt.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Interpretation */}
      <div className="p-4 bg-slate-800 rounded-xl border border-slate-700">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs text-indigo-400 font-medium uppercase tracking-wide">
            {INTENT_LABELS[editedPlan.intent] || editedPlan.intent}
          </span>
        </div>
        <p className="text-sm text-gray-300 leading-relaxed">
          {editedPlan.interpretation}
        </p>
      </div>

      {/* Sub-query list */}
      <div>
        <p className="text-xs text-gray-500 mb-3 uppercase tracking-wide font-medium">
          Search Plan ({editedPlan.sub_queries.length} sub-queries)
        </p>
        <div className="space-y-2">
          {editedPlan.sub_queries.map((sq, i) => (
            <div key={sq.id}
              className="flex items-start gap-3 p-3 bg-slate-800 rounded-lg border border-slate-700
                         hover:border-slate-600 transition-colors group">
              <span className="text-xs text-gray-600 font-mono mt-0.5 w-4 shrink-0">
                {i + 1}
              </span>

              {editingId === sq.id ? (
                <div className="flex-1 flex gap-2">
                  <input
                    autoFocus
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && saveEdit()}
                    className="flex-1 bg-slate-700 text-sm text-gray-100 px-2 py-1
                               rounded border border-indigo-500 focus:outline-none"
                  />
                  <button
                    onClick={saveEdit}
                    className="text-xs text-emerald-400 hover:text-emerald-300 px-2"
                  >
                    Save
                  </button>
                  <button
                    onClick={() => setEditingId(null)}
                    className="text-xs text-gray-500 hover:text-gray-400 px-1"
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-200">{sq.question}</p>
                    {sq.dimension && (
                      <span className="text-xs text-gray-600 mt-0.5 block">
                        [{sq.dimension}]
                      </span>
                    )}
                  </div>
                  <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                    <button
                      onClick={() => startEdit(sq)}
                      className="text-xs text-gray-500 hover:text-indigo-400 px-1.5 py-0.5 rounded"
                    >
                      Edit
                    </button>
                    <button
                      onClick={() => removeSubQuery(sq.id)}
                      className="text-xs text-gray-500 hover:text-red-400 px-1.5 py-0.5 rounded"
                    >
                      Delete
                    </button>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex gap-3 pt-1">
        <button
          onClick={() => onApprove(editedPlan, reportLength)}
          disabled={loading || editedPlan.sub_queries.length === 0}
          className="flex-1 py-2.5 px-4 rounded-xl font-semibold text-sm
                     bg-indigo-600 hover:bg-indigo-500 text-white
                     disabled:opacity-40 disabled:cursor-not-allowed
                     transition-all flex items-center justify-center gap-2"
        >
          {loading ? (
            <><Spinner /> Starting...</>
          ) : (
            <><CheckIcon /> Approve · Start Research</>
          )}
        </button>
        <button
          onClick={onReject}
          disabled={loading}
          className="py-2.5 px-4 rounded-xl font-semibold text-sm
                     bg-slate-700 hover:bg-slate-600 text-gray-300
                     disabled:opacity-40 transition-all"
        >
          Rewrite
        </button>
      </div>
    </div>
  );
}

function CheckIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
    </svg>
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
