import type { Plan, Source, ValidationResult, GapInfo, FileIndexStatus, SessionSummary, LLMConfig, HealthInfo, FeatureFlags } from "./types";

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Research API ──────────────────────────────────────────────────────────

export async function startResearch(
  query: string,
  featureFlags?: FeatureFlags
): Promise<{ session_id: string; plan: Plan; message: string }> {
  const res = await fetch(`${API_URL}/research/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, feature_flags: featureFlags }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to start research");
  }
  return res.json();
}

export async function approvePlan(
  sessionId: string,
  approved: boolean,
  plan?: Plan,
  reportLength: string = "detailed"
): Promise<void> {
  const res = await fetch(`${API_URL}/research/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, approved, plan, report_length: reportLength }),
  });
  if (!res.ok) throw new Error("Failed to approve plan");
}

export async function getReport(sessionId: string): Promise<{
  session_id: string;
  status: string;
  final_report: string;
  citations: object[];
  plan: Plan;
}> {
  const res = await fetch(`${API_URL}/research/${sessionId}`);
  if (!res.ok) throw new Error("Failed to fetch report");
  return res.json();
}

// ── SSE Research Stream ────────────────────────────────────────────────────

export interface StreamCallbacks {
  onSearchStarted?: () => void;
  onSourceFound?: (source: Source) => void;
  onGapDetected?: (info: GapInfo) => void;
  onGapSearchStarted?: () => void;
  onValidationStarted?: () => void;
  onValidationComplete?: (result: ValidationResult) => void;
  onSynthesisStarted?: () => void;
  onCritiqueStarted?: () => void;
  onReportChunk?: (text: string) => void;
  onReportComplete?: (sessionId: string, totalSources: number) => void;
  onError?: (message: string) => void;
}

export function streamResearch(
  sessionId: string,
  callbacks: StreamCallbacks
): EventSource {
  const es = new EventSource(`${API_URL}/research/stream/${sessionId}`);

  const handle = (event: string, fn?: (data: unknown) => void) => {
    es.addEventListener(event, (e: MessageEvent) => {
      try {
        fn?.(JSON.parse(e.data));
      } catch {
        /* ignore */
      }
    });
  };

  handle("search_started", () => callbacks.onSearchStarted?.());
  handle("source_found", (d) => callbacks.onSourceFound?.(d as Source));
  handle("gap_detected", (d) => callbacks.onGapDetected?.(d as GapInfo));
  handle("gap_search_started", () => callbacks.onGapSearchStarted?.());
  handle("validation_started", () => callbacks.onValidationStarted?.());
  handle("validation_complete", (d) =>
    callbacks.onValidationComplete?.(d as ValidationResult)
  );
  handle("synthesis_started", () => callbacks.onSynthesisStarted?.());
  handle("critique_started", () => callbacks.onCritiqueStarted?.());
  handle("report_chunk", (d) =>
    callbacks.onReportChunk?.((d as { text: string }).text)
  );
  handle("report_complete", (d) => {
    const { session_id, total_sources } = d as {
      session_id: string;
      total_sources: number;
    };
    callbacks.onReportComplete?.(session_id, total_sources);
    es.close();
  });
  handle("error", (d) => {
    callbacks.onError?.((d as { message: string }).message);
    es.close();
  });

  es.onerror = () => callbacks.onError?.("SSE connection error");

  return es;
}

// ── SSE Chat (POST-based ReadableStream) ───────────────────────────────

export interface ChatCallbacks {
  onRouting?: (route: string) => void;
  onTargetedSearch?: () => void;
  onSourceFound?: (source: Source) => void;
  onChunk?: (text: string) => void;
  onComplete?: (route: string) => void;
  onError?: (message: string) => void;
}

export async function sendChatMessage(
  sessionId: string,
  message: string,
  callbacks: ChatCallbacks
): Promise<void> {
  const res = await fetch(`${API_URL}/research/${sessionId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  if (!res.ok || !res.body) {
    callbacks.onError?.("Chat request failed");
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalRoute = "memory";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Parse SSE events
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      const lines = part.trim().split("\n");
      let event = "";
      let data = "";
      for (const line of lines) {
        if (line.startsWith("event: ")) event = line.slice(7);
        if (line.startsWith("data: ")) data = line.slice(6);
      }
      if (!event || !data) continue;

      try {
        const parsed = JSON.parse(data);
        switch (event) {
          case "chat_routing":
            finalRoute = parsed.route;
            callbacks.onRouting?.(parsed.route);
            break;
          case "targeted_search":
            callbacks.onTargetedSearch?.();
            break;
          case "source_found":
            callbacks.onSourceFound?.(parsed as Source);
            break;
          case "chat_chunk":
            callbacks.onChunk?.(parsed.text);
            break;
          case "chat_complete":
            callbacks.onComplete?.(finalRoute);
            break;
          case "new_research":
            callbacks.onChunk?.(parsed.response || "");
            callbacks.onComplete?.("new_research");
            break;
          case "error":
            callbacks.onError?.(parsed.message);
            break;
        }
      } catch {
        /* ignore parse errors */
      }
    }
  }
}

// ── File Indexing API ────────────────────────────────────────────────────

export async function indexFiles(
  path: string,
  recursive = true,
  extensions?: string[]
): Promise<{ indexed_files: number; total_chunks: number; skipped: number; errors: string[]; local_search_enabled: boolean }> {
  const res = await fetch(`${API_URL}/files/index`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, recursive, extensions }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to index files");
  }
  return res.json();
}

export async function getFileStatus(): Promise<FileIndexStatus> {
  const res = await fetch(`${API_URL}/files/status`);
  if (!res.ok) throw new Error("Failed to fetch file status");
  return res.json();
}

export async function deleteFileIndex(): Promise<void> {
  const res = await fetch(`${API_URL}/files/delete`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete index");
}

// ── Session List ──────────────────────────────────────────────────────

export async function getSessions(): Promise<SessionSummary[]> {
  const res = await fetch(`${API_URL}/sessions`);
  if (!res.ok) throw new Error("Failed to fetch session list");
  const data = await res.json();
  return data.sessions as SessionSummary[];
}

// ── LLM Settings ───────────────────────────────────────────────────────

export async function getHealth(): Promise<HealthInfo> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function getSettings(): Promise<LLMConfig> {
  const res = await fetch(`${API_URL}/settings`);
  if (!res.ok) throw new Error("Failed to fetch settings");
  return res.json();
}

export async function updateSettings(config: LLMConfig): Promise<LLMConfig> {
  const res = await fetch(`${API_URL}/settings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to update settings");
  }
  const data = await res.json();
  return data.config as LLMConfig;
}
