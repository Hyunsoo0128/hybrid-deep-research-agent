"use client";

import { useState } from "react";
import type { FeatureFlags } from "@/lib/types";

// ── Depth presets ────────────────────────────────────────────────────────────

type Depth = "fast" | "normal" | "deep";

const DEPTH_PRESETS: Record<Depth, Partial<FeatureFlags>> = {
  fast: {
    query_decomp: true,  crag: false, stride: false,
    mass_rag:     true,  rhinoinsight: false,
    alignrag:     false, spec_rag_critic: false,
    construct:    false, proclaim: false, navirag: false,
  },
  normal: {
    query_decomp: true,  crag: true,  stride: true,
    mass_rag:     true,  rhinoinsight: true,
    alignrag:     true,  spec_rag_critic: false,
    construct:    false, proclaim: false, navirag: false,
  },
  deep: {
    query_decomp: true,  crag: true,  stride: true,
    mass_rag:     true,  rhinoinsight: true,
    alignrag:     true,  spec_rag_critic: true,
    construct:    true,  proclaim: false, navirag: false,
  },
};

function detectDepth(flags: FeatureFlags): Depth | null {
  for (const [depth, preset] of Object.entries(DEPTH_PRESETS) as [Depth, Partial<FeatureFlags>][]) {
    if (Object.entries(preset).every(([k, v]) => flags[k as keyof FeatureFlags] === v)) {
      return depth;
    }
  }
  return null; // custom
}

// ── Technique metadata ────────────────────────────────────────────────────────

interface TechniqueMeta {
  key: keyof FeatureFlags;
  label: string;
  paper?: string;
  arxiv?: string;
  description: string;
}

const STAGES: { id: string; label: string; color: string; techniques: TechniqueMeta[] }[] = [
  {
    id: "stage1",
    label: "Stage 1 — Search Strategy",
    color: "blue",
    techniques: [
      {
        key: "query_decomp",
        label: "Query Decomposition",
        paper: "2507.00355",
        arxiv: "https://arxiv.org/abs/2507.00355",
        description: "Decomposes the query into 5 semantic dimensions to maximize search coverage.",
      },
      {
        key: "crag",
        label: "CRAG — Corrective RAG",
        paper: "2401.15884",
        arxiv: "https://arxiv.org/abs/2401.15884",
        description: "Evaluates and filters search results in 3 tiers: relevant / partial / irrelevant.",
      },
      {
        key: "stride",
        label: "STRIDE — Supervisor Routing",
        paper: "2604.17405",
        arxiv: "https://arxiv.org/abs/2604.17405",
        description: "Meta-Planner first establishes an abstract strategy (Sq), then Supervisor decides retrieve/rewrite/answer per sub-query.",
      },
    ],
  },
  {
    id: "stage2",
    label: "Stage 2 — Evidence Building",
    color: "teal",
    techniques: [
      {
        key: "mass_rag",
        label: "MASS-RAG — Multi-agent Synthesis",
        paper: "2604.18509",
        arxiv: "https://arxiv.org/abs/2604.18509",
        description: "Three agents — Summarizer, Extractor, and Reasoner — synthesize sources in parallel.",
      },
      {
        key: "rhinoinsight",
        label: "RhinoInsight — VCM + EAM",
        paper: "2511.18743",
        arxiv: "https://arxiv.org/abs/2511.18743",
        description: "Validates research sub-goals with a VCM checklist and normalizes evidence with EAM.",
      },
    ],
  },
  {
    id: "stage3",
    label: "Stage 3 — Verification & Alignment",
    color: "indigo",
    techniques: [
      {
        key: "alignrag",
        label: "AlignRAG — Factual Alignment",
        paper: "2504.14858",
        arxiv: "https://arxiv.org/abs/2504.14858",
        description: "Validates drafts in 3 phases: Phase1 (topic deviation) / Phase2 (manipulated citations) / Phase3 (numerical contradictions).",
      },
      {
        key: "spec_rag_critic",
        label: "Spec RAG Critic",
        paper: "2407.08223",
        arxiv: "https://arxiv.org/abs/2407.08223",
        description: "Eliminates self-preference bias via Drafter(local) → Verifier(cloud) → Refiner(local) separation. HybridProvider only.",
      },
    ],
  },
  {
    id: "stage4",
    label: "Stage 4 — Quality Enhancement",
    color: "violet",
    techniques: [
      {
        key: "construct",
        label: "CONSTRUCT — Knowledge Graph",
        paper: "2603.18014",
        arxiv: "https://arxiv.org/abs/2603.18014",
        description: "Builds an entity-relationship graph across citations to improve report consistency.",
      },
    ],
  },
];

// ── Props ─────────────────────────────────────────────────────────────────────

interface Props {
  flags: FeatureFlags;
  onChange: (flags: FeatureFlags) => void;
  disabled?: boolean;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function TechniquesPanel({ flags, onChange, disabled }: Props) {
  const [expanded, setExpanded] = useState(false);

  const activeCount = STAGES.flatMap((s) => s.techniques).filter((t) => flags[t.key]).length;
  const totalCount  = STAGES.flatMap((s) => s.techniques).length;
  const currentDepth = detectDepth(flags);

  const applyDepth = (depth: Depth) => {
    if (disabled) return;
    onChange({ ...flags, ...DEPTH_PRESETS[depth] });
  };

  const toggle = (key: keyof FeatureFlags) => {
    if (disabled) return;
    onChange({ ...flags, [key]: !flags[key] });
  };

  const togglePrivacy = () => {
    if (disabled) return;
    onChange({ ...flags, privacy_mode: !flags.privacy_mode });
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* Header toggle */}
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-2 rounded-lg
                   text-gray-500 hover:text-gray-300 hover:bg-slate-800
                   transition-colors duration-150 text-sm"
      >
        <span className="flex items-center gap-2">
          <FlaskIcon />
          Research Techniques
          <span className="text-xs px-1.5 py-0.5 rounded-full bg-indigo-900/50 text-indigo-400 border border-indigo-800/50">
            {activeCount} / {totalCount} active
          </span>
          {currentDepth && (
            <span className={`text-xs px-1.5 py-0.5 rounded-full border
              ${currentDepth === "fast"   ? "bg-sky-900/40 text-sky-400 border-sky-800/50" :
                currentDepth === "normal" ? "bg-indigo-900/40 text-indigo-400 border-indigo-800/50" :
                                            "bg-violet-900/40 text-violet-400 border-violet-800/50"}`}>
              {currentDepth}
            </span>
          )}
        </span>
        <ChevronIcon expanded={expanded} />
      </button>

      {expanded && (
        <div className="mt-2 rounded-xl border border-slate-700/50 bg-slate-900/50 overflow-hidden">

          {/* Depth presets */}
          <div className="px-4 py-3 border-b border-slate-800/50 bg-slate-800/20">
            <p className="text-xs text-gray-500 mb-2 font-medium uppercase tracking-wider">Depth Preset</p>
            <div className="flex gap-2">
              {(["fast", "normal", "deep"] as Depth[]).map((d) => (
                <button
                  key={d}
                  type="button"
                  onClick={() => applyDepth(d)}
                  disabled={disabled}
                  className={`flex-1 py-1.5 rounded-lg text-xs font-medium border transition-colors
                    ${currentDepth === d
                      ? d === "fast"   ? "bg-sky-900/60 text-sky-300 border-sky-700"
                      : d === "normal" ? "bg-indigo-900/60 text-indigo-300 border-indigo-700"
                      :                  "bg-violet-900/60 text-violet-300 border-violet-700"
                      : "bg-slate-800/50 text-gray-500 border-slate-700 hover:text-gray-300 hover:border-slate-600"
                    }`}
                >
                  {d === "fast" ? "⚡ Fast" : d === "normal" ? "◉ Normal" : "◈ Deep"}
                </button>
              ))}
            </div>
            <div className="mt-2 grid grid-cols-3 gap-1 text-xs text-gray-600">
              <span>Minimal search</span>
              <span className="text-center">Balanced</span>
              <span className="text-right">Highest quality</span>
            </div>
          </div>

          {/* Stage sections */}
          {STAGES.map((stage) => {
            const stageActive = stage.techniques.filter((t) => flags[t.key]).length;
            return (
              <div key={stage.id}>
                <StageHeader
                  label={stage.label}
                  color={stage.color}
                  activeCount={stageActive}
                  total={stage.techniques.length}
                />
                <div className="divide-y divide-slate-800/50">
                  {stage.techniques.map((t) => (
                    <TechniqueRow
                      key={t.key}
                      meta={t}
                      color={stage.color}
                      enabled={!!flags[t.key]}
                      onToggle={() => toggle(t.key)}
                      disabled={disabled}
                    />
                  ))}
                </div>
              </div>
            );
          })}

          {/* Settings section */}
          <div>
            <StageHeader label="Settings" color="gray" activeCount={flags.privacy_mode ? 1 : 0} total={1} />
            <div
              className={`px-4 py-3 flex items-start gap-3
                          ${disabled ? "opacity-50" : "hover:bg-slate-800/30 cursor-pointer"}`}
              onClick={togglePrivacy}
            >
              <ToggleSwitch enabled={flags.privacy_mode} color="gray" disabled={disabled} onToggle={togglePrivacy} />
              <div className="flex-1 min-w-0">
                <span className={`text-sm font-medium ${flags.privacy_mode ? "text-gray-200" : "text-gray-500"}`}>
                  Privacy Mode
                </span>
                <p className="text-xs text-gray-600 mt-0.5 leading-relaxed">
                  Blocks raw local file content from being sent to cloud LLMs. Requires mass_rag to be enabled.
                </p>
              </div>
            </div>
          </div>

        </div>
      )}
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

const COLOR_MAP: Record<string, { header: string; accent: string; toggle: string; badge: string }> = {
  blue:   { header: "bg-blue-900/10",   accent: "text-blue-400",   toggle: "bg-blue-600",   badge: "bg-blue-900/30 text-blue-400 border-blue-800/50" },
  teal:   { header: "bg-teal-900/10",   accent: "text-teal-400",   toggle: "bg-teal-600",   badge: "bg-teal-900/30 text-teal-400 border-teal-800/50" },
  indigo: { header: "bg-indigo-900/10", accent: "text-indigo-400", toggle: "bg-indigo-600", badge: "bg-indigo-900/30 text-indigo-400 border-indigo-800/50" },
  violet: { header: "bg-violet-900/10", accent: "text-violet-400", toggle: "bg-violet-600", badge: "bg-violet-900/30 text-violet-400 border-violet-800/50" },
  gray:   { header: "bg-slate-800/20",  accent: "text-gray-400",   toggle: "bg-slate-600",  badge: "bg-slate-700/30 text-gray-400 border-slate-700/50" },
};

function StageHeader({ label, color, activeCount, total }: {
  label: string; color: string; activeCount: number; total: number;
}) {
  const c = COLOR_MAP[color] ?? COLOR_MAP.gray;
  return (
    <div className={`px-4 py-2 flex items-center gap-2 border-b border-t border-slate-800/50 ${c.header}`}>
      <span className={`text-xs font-semibold uppercase tracking-wider ${c.accent}`}>{label}</span>
      <span className="text-xs text-gray-600">{activeCount}/{total} on</span>
    </div>
  );
}

function ToggleSwitch({ enabled, color, disabled, onToggle }: {
  enabled: boolean; color: string; disabled?: boolean; onToggle: () => void;
}) {
  const c = COLOR_MAP[color] ?? COLOR_MAP.gray;
  return (
    <button
      type="button"
      role="switch"
      aria-checked={enabled}
      onClick={(e) => { e.stopPropagation(); onToggle(); }}
      disabled={disabled}
      className={`relative mt-0.5 flex-shrink-0 w-9 h-5 rounded-full transition-colors duration-200
                  focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-1
                  focus:ring-offset-slate-900
                  ${enabled ? c.toggle : "bg-slate-700"}`}
    >
      <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow
                        transition-transform duration-200 ${enabled ? "translate-x-4" : "translate-x-0"}`} />
    </button>
  );
}

function TechniqueRow({ meta, color, enabled, onToggle, disabled }: {
  meta: TechniqueMeta; color: string; enabled: boolean; onToggle: () => void; disabled?: boolean;
}) {
  const c = COLOR_MAP[color] ?? COLOR_MAP.gray;
  return (
    <div
      className={`px-4 py-3 flex items-start gap-3
                  ${disabled ? "opacity-50" : "hover:bg-slate-800/30 cursor-pointer"}`}
      onClick={onToggle}
    >
      <ToggleSwitch enabled={enabled} color={color} disabled={disabled} onToggle={onToggle} />

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`text-sm font-medium ${enabled ? "text-gray-200" : "text-gray-500"}`}>
            {meta.label}
          </span>
          {meta.paper && meta.arxiv && (
            <a
              href={meta.arxiv}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className={`text-xs px-1.5 py-0.5 rounded border font-mono transition-colors
                          ${enabled ? `${c.badge} hover:opacity-80` : "bg-slate-700/30 text-gray-600 border-slate-700/50 hover:text-gray-400"}`}
            >
              {meta.paper}
            </a>
          )}

        </div>
        <p className="text-xs text-gray-600 mt-0.5 leading-relaxed">{meta.description}</p>
      </div>
    </div>
  );
}

function FlaskIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
        d="M9 3h6m-6 0v6l-4 9a1 1 0 001 1h12a1 1 0 001-1l-4-9V3m-6 0h6" />
    </svg>
  );
}

function ChevronIcon({ expanded }: { expanded: boolean }) {
  return (
    <svg
      className={`w-4 h-4 transition-transform duration-200 ${expanded ? "rotate-180" : ""}`}
      fill="none" viewBox="0 0 24 24" stroke="currentColor"
    >
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
    </svg>
  );
}
