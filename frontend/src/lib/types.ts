export type AppState =
  | "idle"
  | "starting"
  | "plan_review"
  | "streaming"
  | "complete"
  | "error";

export interface SubQuery {
  id: string;
  question: string;
  dimension?: string;
  status?: string;
}

export interface Plan {
  intent: string;
  interpretation: string;
  sub_queries: SubQuery[];
  depth: string;
  estimated_time: string;
  local_files?: string[];
}

export interface Source {
  title: string;
  url: string;
  trust_level: "high" | "medium" | "low";
  confidence: number;
  source_type: "web" | "local";
}

export interface ValidationResult {
  quality_score: number;
  well_corroborated: number;
  contradictions: number;
  summary: string;
}

export type ProgressStatus = "pending" | "active" | "done" | "skipped";

export interface ProgressStep {
  id: string;
  label: string;
  status: ProgressStatus;
  detail?: string;
}

export interface GapInfo {
  gaps: string[];
  additional_queries: string[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  route?: string;
  sources?: Source[];
}

export interface FileIndexStatus {
  total_chunks: number;
  collection: string;
  local_search_enabled: boolean;
}

export interface SessionSummary {
  session_id: string;
  query: string;
  status: "plan_review" | "approved" | "complete" | "error" | string;
  created_at: string;
}

export interface LLMConfig {
  provider: "bedrock" | "claude" | "ollama" | "hybrid" | string;
  model?: string;
  embed_model?: string;
  host?: string;
  region?: string;
  // hybrid only
  cloud_provider?: string;
  cloud_model?: string;
  local_model?: string;
}

export interface HealthInfo {
  status: string;
  local_search_enabled: boolean;
  llm: LLMConfig;
}

export interface FeatureFlags {
  // Stage 1: Search strategy
  query_decomp: boolean;
  crag: boolean;
  stride: boolean;
  // Stage 2: Evidence building
  mass_rag: boolean;
  rhinoinsight: boolean;
  // Stage 3: Verification & alignment
  alignrag: boolean;
  spec_rag_critic: boolean;
  // Stage 4: Quality enhancement
  construct: boolean;
  proclaim: boolean;
  navirag: boolean;
  // Infrastructure
  dsap: boolean;
  sdp: boolean;
  // Settings
  privacy_mode: boolean;
}

export const DEFAULT_FEATURE_FLAGS: FeatureFlags = {
  // Stage 1
  query_decomp:    true,
  crag:            true,
  stride:          false,
  // Stage 2
  mass_rag:        false,
  rhinoinsight:    false,
  // Stage 3
  alignrag:        true,
  spec_rag_critic: false,
  // Stage 4
  construct:       false,
  proclaim:        false,
  navirag:         false,
  // Infrastructure
  dsap:            true,
  sdp:             false,
  // Settings
  privacy_mode:    false,
};
