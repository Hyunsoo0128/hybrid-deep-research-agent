"use client";

import { useState, useEffect } from "react";
import type { SessionSummary } from "@/lib/types";
import { getSessions, getReport } from "@/lib/api";

interface Props {
  onRestore: (sessionId: string, query: string, report: string, totalSources: number) => void;
}

const STATUS_LABEL: Record<string, { label: string; color: string }> = {
  complete:    { label: "Complete",    color: "text-emerald-400" },
  error:       { label: "Error",       color: "text-red-400" },
  plan_review: { label: "Reviewing",   color: "text-amber-400" },
  approved:    { label: "In Progress", color: "text-indigo-400" },
};

function formatDate(iso: string) {
  if (!iso) return "";
  const d = new Date(iso);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffDays = Math.floor(diffMs / 86400000);
  if (diffDays === 0) {
    return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  } else if (diffDays === 1) {
    return "Yesterday";
  } else if (diffDays < 7) {
    return `${diffDays} days ago`;
  }
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export default function SessionHistory({ onRestore }: Props) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [restoring, setRestoring] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    getSessions()
      .then((s) => {
        setSessions(s.filter((x) => x.query));
        if (s.filter((x) => x.query).length > 0) setOpen(true);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const completedSessions = sessions.filter((s) => s.status === "complete");

  const handleRestore = async (session: SessionSummary) => {
    if (restoring) return;
    setRestoring(session.session_id);
    try {
      const data = await getReport(session.session_id);
      if (data.final_report) {
        onRestore(
          session.session_id,
          session.query,
          data.final_report,
          data.citations?.length ?? 0,
        );
      }
    } catch {
      // Silently ignore restore failures
    } finally {
      setRestoring(null);
    }
  };

  if (loading || sessions.length === 0) return null;

  return (
    <div className="w-full max-w-2xl mx-auto">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 text-xs text-gray-500 hover:text-gray-300
                   transition-colors mb-3 w-full justify-between"
      >
        <span className="uppercase tracking-wide font-medium">
          Previous Research ({completedSessions.length})
        </span>
        <ChevronIcon open={open} />
      </button>

      {open && (
        <div className="space-y-1.5">
          {sessions.slice(0, 10).map((s) => {
            const st = STATUS_LABEL[s.status] ?? { label: s.status, color: "text-gray-500" };
            const isRestoring = restoring === s.session_id;
            return (
              <div
                key={s.session_id}
                className="flex items-center gap-3 px-3 py-2.5 rounded-lg bg-slate-800/60
                           border border-slate-700/50 hover:border-slate-600
                           hover:bg-slate-800 transition-all group"
              >
                <HistoryIcon />
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-300 truncate">{s.query}</p>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className={`text-xs font-medium ${st.color}`}>{st.label}</span>
                    {s.created_at && (
                      <span className="text-xs text-gray-600">{formatDate(s.created_at)}</span>
                    )}
                  </div>
                </div>
                {s.status === "complete" && (
                  <button
                    onClick={() => handleRestore(s)}
                    disabled={!!restoring}
                    className="text-xs text-gray-500 hover:text-indigo-400
                               px-2 py-1 rounded-md border border-slate-700
                               hover:border-indigo-600 transition-all
                               opacity-0 group-hover:opacity-100
                               disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
                  >
                    {isRestoring ? "Loading..." : "Load"}
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function HistoryIcon() {
  return (
    <svg className="w-3.5 h-3.5 text-gray-600 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      className={`w-3.5 h-3.5 transition-transform ${open ? "rotate-180" : ""}`}
      fill="none" viewBox="0 0 24 24" stroke="currentColor"
    >
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
    </svg>
  );
}
