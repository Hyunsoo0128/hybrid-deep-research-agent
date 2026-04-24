"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import type {
  AppState, Plan, Source, ProgressStep,
  ValidationResult, GapInfo, ChatMessage, LLMConfig,
  FeatureFlags,
} from "@/lib/types";
import { DEFAULT_FEATURE_FLAGS } from "@/lib/types";
import { startResearch, approvePlan, streamResearch, getHealth } from "@/lib/api";
import QueryInput from "@/components/QueryInput";
import PlanReview from "@/components/PlanReview";
import ProgressPanel from "@/components/ProgressPanel";
import ReportView from "@/components/ReportView";
import ChatInterface from "@/components/ChatInterface";
import FileIndexer from "@/components/FileIndexer";
import SettingsModal from "@/components/SettingsModal";
import SessionHistory from "@/components/SessionHistory";
import TechniquesPanel from "@/components/TechniquesPanel";

const INITIAL_STEPS: ProgressStep[] = [
  { id: "search",     label: "Parallel Sub-query Search",  status: "pending" },
  { id: "gap",        label: "Knowledge Gap Detection",    status: "pending" },
  { id: "validation", label: "Cross-source Validation",    status: "pending" },
  { id: "writing",    label: "Report Writing",             status: "pending" },
  { id: "critique",   label: "Quality Review",             status: "pending" },
];

function updateStep(
  steps: ProgressStep[],
  id: string,
  status: ProgressStep["status"],
  detail?: string
): ProgressStep[] {
  return steps.map((s) => (s.id === id ? { ...s, status, detail } : s));
}

function useElapsedTimer(active: boolean) {
  const [elapsed, setElapsed] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (active) {
      setElapsed(0);
      intervalRef.current = setInterval(() => setElapsed((n) => n + 1), 1000);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [active]);

  const mm = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const ss = String(elapsed % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

function LLMBadge({ config }: { config: LLMConfig | null }) {
  if (!config) return null;
  const label =
    config.provider === "ollama"
      ? `🖥️ ${config.model ?? "ollama"}`
      : config.provider === "claude"
      ? `🤖 ${config.model ?? "claude"}`
      : config.provider === "hybrid"
      ? `⚡ Hybrid`
      : `☁️ ${(config.model ?? "bedrock").split(".").pop()?.split(":")[0] ?? "bedrock"}`;

  return (
    <span className="text-xs text-gray-500 px-2 py-1 rounded-md bg-slate-800 border border-slate-700
                     hidden sm:inline-block truncate max-w-[180px]" title={config.model}>
      {label}
    </span>
  );
}

export default function Home() {
  const [appState, setAppState] = useState<AppState>("idle");
  const [sessionId, setSessionId] = useState<string>("");
  const [plan, setPlan] = useState<Plan | null>(null);
  const [steps, setSteps] = useState<ProgressStep[]>(INITIAL_STEPS);
  const [sources, setSources] = useState<Source[]>([]);
  const [totalSources, setTotalSources] = useState(0);
  const [gapInfo, setGapInfo] = useState<GapInfo | null>(null);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [report, setReport] = useState("");
  const [reportStreaming, setReportStreaming] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [approving, setApproving] = useState(false);

  const [llmConfig, setLlmConfig] = useState<LLMConfig | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [featureFlags, setFeatureFlags] = useState<FeatureFlags>(DEFAULT_FEATURE_FLAGS);

  const esRef = useRef<EventSource | null>(null);
  const elapsedTime = useElapsedTimer(appState === "streaming");

  // Initial LLM config load
  useEffect(() => {
    getHealth()
      .then((h) => setLlmConfig(h.llm))
      .catch(() => {});
  }, []);

  // ── Step 1: Start research ───────────────────────────────────────────────
  const handleQuerySubmit = async (query: string) => {
    setAppState("starting");
    setError(null);
    resetResearchState();
    try {
      const res = await startResearch(query, featureFlags);
      setSessionId(res.session_id);
      setPlan(res.plan);
      setAppState("plan_review");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start research");
      setAppState("error");
    }
  };

  // ── Step 2: Approve plan ───────────────────────────────────────────────
  const handleApprove = async (editedPlan: Plan, reportLength: string) => {
    if (!sessionId) return;
    setApproving(true);
    try {
      await approvePlan(sessionId, true, editedPlan, reportLength);
      setAppState("streaming");
      startStream(sessionId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to approve plan");
      setAppState("error");
    } finally {
      setApproving(false);
    }
  };

  const handleReject = () => {
    setAppState("idle");
    setPlan(null);
    setSessionId("");
  };

  // ── Cancel research ──────────────────────────────────────────────────────
  const handleCancel = () => {
    esRef.current?.close();
    esRef.current = null;
    setAppState("idle");
    resetResearchState();
    setPlan(null);
    setSessionId("");
    setError(null);
  };

  // ── Step 3: SSE Stream ──────────────────────────────────────────────
  const startStream = useCallback((sid: string) => {
    setSteps(INITIAL_STEPS);

    const es = streamResearch(sid, {
      onSearchStarted: () => {
        setSteps((prev) => updateStep(prev, "search", "active", "Searching in parallel..."));
      },
      onSourceFound: (src) => {
        setSources((prev) => [src, ...prev].slice(0, 50));
        setTotalSources((n) => n + 1);
      },
      onGapDetected: (info) => {
        setGapInfo(info);
        setSteps((prev) => {
          let s = updateStep(prev, "search", "done");
          s = updateStep(s, "gap", "active", `${info.gaps.length} gap(s) detected`);
          return s;
        });
      },
      onGapSearchStarted: () => {
        setSteps((prev) => updateStep(prev, "gap", "active", "Searching for more..."));
      },
      onValidationStarted: () => {
        setSteps((prev) => {
          let s = updateStep(prev, "search", "done");
          s = updateStep(s, "gap", "done");
          s = updateStep(s, "validation", "active", "Cross-validating...");
          return s;
        });
      },
      onValidationComplete: (result) => {
        setValidation(result);
        setSteps((prev) => updateStep(prev, "validation", "done"));
      },
      onSynthesisStarted: () => {
        setSteps((prev) => {
          let s = updateStep(prev, "search", "done");
          s = updateStep(s, "gap", "done");
          s = updateStep(s, "validation", "done");
          s = updateStep(s, "writing", "active", "Writing report...");
          return s;
        });
      },
      onCritiqueStarted: () => {
        setSteps((prev) => {
          let s = updateStep(prev, "writing", "done");
          s = updateStep(s, "critique", "active", "Reviewing quality...");
          return s;
        });
        setReportStreaming(false);
      },
      onReportChunk: (text) => {
        setReport((prev) => prev + text);
        setReportStreaming(true);
        setSteps((prev) => updateStep(prev, "writing", "active"));
      },
      onReportComplete: (_, total) => {
        setTotalSources(total);
        setReportStreaming(false);
        setSteps((prev) => {
          let s = updateStep(prev, "writing", "done");
          s = updateStep(s, "critique", "done");
          return s;
        });
        setAppState("complete");
        es.close();
      },
      onError: (msg) => {
        setError(msg);
        setAppState("error");
      },
    });

    esRef.current = es;
  }, []);

  const resetResearchState = () => {
    setSources([]);
    setTotalSources(0);
    setGapInfo(null);
    setValidation(null);
    setReport("");
    setReportStreaming(false);
    setChatMessages([]);
    setSteps(INITIAL_STEPS);
    esRef.current?.close();
  };

  const handleNewResearch = () => {
    setAppState("idle");
    resetResearchState();
    setPlan(null);
    setSessionId("");
    setError(null);
  };

  const handleChatMessage = (msg: ChatMessage) => {
    setChatMessages((prev) => [...prev, msg]);
  };

  // Restore previous session
  const handleRestore = (sid: string, _query: string, restoredReport: string, total: number) => {
    resetResearchState();
    setSessionId(sid);
    setReport(restoredReport);
    setTotalSources(total);
    setAppState("complete");
  };

  // ── Render ────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen flex flex-col">
      {/* Settings modal */}
      {showSettings && (
        <SettingsModal
          onClose={() => setShowSettings(false)}
          onSaved={(cfg) => setLlmConfig(cfg)}
        />
      )}

      {/* Header */}
      <header className="sticky top-0 z-10 border-b border-slate-800 bg-slate-950/80 backdrop-blur-sm">
        <div className="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
          <button
            onClick={handleNewResearch}
            className="flex items-center gap-2 text-gray-300 hover:text-white transition-colors"
          >
            <LogoIcon />
            <span className="font-semibold text-sm">Deep Research</span>
          </button>

          <div className="flex items-center gap-2">
            <LLMBadge config={llmConfig} />

            {/* Settings button */}
            <button
              onClick={() => setShowSettings(true)}
              className="p-1.5 rounded-lg text-gray-500 hover:text-gray-300
                         hover:bg-slate-800 transition-colors"
              title="LLM Settings"
            >
              <SettingsIcon />
            </button>

            {/* Cancel while streaming */}
            {appState === "streaming" && (
              <button
                onClick={handleCancel}
                className="text-xs text-gray-500 hover:text-red-400 px-3 py-1.5
                           rounded-lg border border-slate-700 hover:border-red-800
                           transition-colors"
              >
                Cancel
              </button>
            )}

            {(appState === "complete" || appState === "plan_review") && (
              <button
                onClick={handleNewResearch}
                className="text-xs text-gray-500 hover:text-gray-300 px-3 py-1.5
                           rounded-lg border border-slate-700 hover:border-slate-600
                           transition-colors"
              >
                New Research
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 max-w-5xl mx-auto w-full px-4 py-10 space-y-8">

        {/* idle / starting */}
        {(appState === "idle" || appState === "starting") && (
          <div className="flex flex-col items-center gap-10">
            <div className="text-center space-y-3">
              <h1 className="text-3xl font-bold text-gray-100 tracking-tight">
                Deep Research
              </h1>
              <p className="text-gray-500 text-base">
                AI browses the web and generates structured research reports
              </p>
            </div>
            <QueryInput
              onSubmit={handleQuerySubmit}
              loading={appState === "starting"}
            />
            <TechniquesPanel
              flags={featureFlags}
              onChange={setFeatureFlags}
              disabled={appState === "starting"}
            />
            <SessionHistory onRestore={handleRestore} />
            <FileIndexer />
          </div>
        )}

        {/* plan_review */}
        {appState === "plan_review" && plan && (
          <div className="flex flex-col items-center gap-4">
            <PlanReview
              plan={plan}
              onApprove={handleApprove}
              onReject={handleReject}
              loading={approving}
            />
          </div>
        )}

        {/* streaming */}
        {appState === "streaming" && (
          <>
            {/* Elapsed time */}
            <div className="flex items-center justify-between text-xs text-gray-600">
              <span>Research in progress...</span>
              <span className="font-mono">{elapsedTime}</span>
            </div>
            <ProgressPanel
              steps={steps}
              sources={sources}
              totalSources={totalSources}
              gapInfo={gapInfo}
              validation={validation}
            />
            {report && (
              <ReportView
                report={report}
                streaming={reportStreaming}
                totalSources={totalSources}
                qualityScore={validation?.quality_score}
                sessionId={sessionId}
              />
            )}
          </>
        )}

        {/* complete */}
        {appState === "complete" && (
          <>
            <ReportView
              report={report}
              streaming={false}
              totalSources={totalSources}
              qualityScore={validation?.quality_score}
              sessionId={sessionId}
            />
            <ChatInterface
              sessionId={sessionId}
              messages={chatMessages}
              onMessage={handleChatMessage}
            />
          </>
        )}

        {/* error */}
        {appState === "error" && (
          <div className="flex flex-col items-center gap-6 py-16">
            <div className="p-4 bg-red-900/20 border border-red-700/50 rounded-xl max-w-md text-center">
              <p className="text-sm text-red-400 font-medium mb-1">An error occurred</p>
              <p className="text-xs text-red-300/70">{error}</p>
            </div>
            <button
              onClick={handleNewResearch}
              className="text-sm text-gray-400 hover:text-gray-200 underline underline-offset-4"
            >
              Back to start
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

// ── Icons ─────────────────────────────────────────────────────────────

function LogoIcon() {
  return (
    <svg className="w-5 h-5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  );
}
