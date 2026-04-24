"use client";

import { useState, useEffect } from "react";
import type { LLMConfig } from "@/lib/types";
import { getSettings, updateSettings } from "@/lib/api";

interface Props {
  onClose: () => void;
  onSaved: (config: LLMConfig) => void;
}

const DEFAULT_MODELS: Record<string, string> = {
  bedrock: "us.anthropic.claude-sonnet-4-6",
  claude: "claude-sonnet-4-6",
  ollama: "qwen3:8b",
  hybrid: "",
};

export default function SettingsModal({ onClose, onSaved }: Props) {
  const [config, setConfig] = useState<LLMConfig>({ provider: "ollama", model: "qwen3:8b" });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getSettings()
      .then(setConfig)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const handleProviderChange = (provider: string) => {
    if (provider === "hybrid") {
      setConfig({
        provider: "hybrid",
        cloud_provider: "bedrock",
        cloud_model: "us.anthropic.claude-sonnet-4-6",
        local_model: "qwen3:8b",
        embed_model: "nomic-embed-text",
        host: "http://localhost:11434",
        region: "us-west-2",
      });
    } else {
      setConfig({
        provider,
        model: DEFAULT_MODELS[provider] ?? "",
        embed_model: provider === "ollama" ? "nomic-embed-text" : undefined,
        host: provider === "ollama" ? "http://localhost:11434" : undefined,
        region: provider === "bedrock" ? "us-west-2" : undefined,
      });
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const saved = await updateSettings(config);
      onSaved(saved);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-md bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl p-6 mx-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-base font-semibold text-gray-100">LLM Settings</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-300 transition-colors"
          >
            <CloseIcon />
          </button>
        </div>

        {loading ? (
          <div className="py-8 flex justify-center">
            <Spinner />
          </div>
        ) : (
          <div className="space-y-5">
            {/* Provider selection */}
            <div>
              <label className="text-xs text-gray-500 uppercase tracking-wide font-medium mb-2 block">
                Provider
              </label>
              <div className="grid grid-cols-2 gap-2">
                {(["bedrock", "claude", "ollama", "hybrid"] as const).map((p) => (
                  <button
                    key={p}
                    onClick={() => handleProviderChange(p)}
                    className={`py-2.5 px-2 rounded-lg text-xs font-medium border transition-all ${
                      config.provider === p
                        ? "bg-indigo-600 border-indigo-500 text-white"
                        : "bg-slate-800 border-slate-700 text-gray-400 hover:border-slate-600"
                    }`}
                  >
                    {p === "bedrock" && "☁️ Bedrock"}
                    {p === "claude" && "🤖 Claude API"}
                    {p === "ollama" && "🖥️ Local LLM"}
                    {p === "hybrid" && "⚡ Hybrid"}
                  </button>
                ))}
              </div>
              {config.provider === "hybrid" && (
                <p className="text-xs text-gray-600 mt-2">
                  Cloud(Bedrock) — generation node · Local(Ollama) — evaluation node + Spec RAG
                </p>
              )}
            </div>

            {/* Single-provider model */}
            {config.provider !== "hybrid" && (
              <div>
                <label className="text-xs text-gray-500 uppercase tracking-wide font-medium mb-2 block">
                  Model
                </label>
                <input
                  type="text"
                  value={config.model ?? ""}
                  onChange={(e) => setConfig((c) => ({ ...c, model: e.target.value }))}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2
                             text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  placeholder={DEFAULT_MODELS[config.provider] ?? "Model name"}
                />
                {config.provider === "ollama" && (
                  <p className="text-xs text-gray-600 mt-1">e.g. qwen3:8b · exaone3.5:7.8b</p>
                )}
              </div>
            )}

            {/* Hybrid-specific settings */}
            {config.provider === "hybrid" && (
              <>
                <div>
                  <label className="text-xs text-gray-500 uppercase tracking-wide font-medium mb-2 block">
                    Cloud Model (Bedrock)
                  </label>
                  <input
                    type="text"
                    value={config.cloud_model ?? "us.anthropic.claude-sonnet-4-6"}
                    onChange={(e) => setConfig((c) => ({ ...c, cloud_model: e.target.value }))}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2
                               text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500 uppercase tracking-wide font-medium mb-2 block">
                    Local Model (Ollama)
                  </label>
                  <input
                    type="text"
                    value={config.local_model ?? "qwen3:8b"}
                    onChange={(e) => setConfig((c) => ({ ...c, local_model: e.target.value }))}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2
                               text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500 uppercase tracking-wide font-medium mb-2 block">
                    AWS Region
                  </label>
                  <input
                    type="text"
                    value={config.region ?? "us-west-2"}
                    onChange={(e) => setConfig((c) => ({ ...c, region: e.target.value }))}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2
                               text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>
              </>
            )}

            {/* Ollama-specific settings (non-hybrid) */}
            {config.provider === "ollama" && (
              <>
                <div>
                  <label className="text-xs text-gray-500 uppercase tracking-wide font-medium mb-2 block">
                    Embedding Model
                  </label>
                  <input
                    type="text"
                    value={config.embed_model ?? "nomic-embed-text"}
                    onChange={(e) => setConfig((c) => ({ ...c, embed_model: e.target.value }))}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2
                               text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500 uppercase tracking-wide font-medium mb-2 block">
                    Ollama Server
                  </label>
                  <input
                    type="text"
                    value={config.host ?? "http://localhost:11434"}
                    onChange={(e) => setConfig((c) => ({ ...c, host: e.target.value }))}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2
                               text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>
              </>
            )}

            {/* Bedrock-specific settings (non-hybrid) */}
            {config.provider === "bedrock" && (
              <div>
                <label className="text-xs text-gray-500 uppercase tracking-wide font-medium mb-2 block">
                  AWS Region
                </label>
                <input
                  type="text"
                  value={config.region ?? "us-west-2"}
                  onChange={(e) => setConfig((c) => ({ ...c, region: e.target.value }))}
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2
                             text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
            )}

            {error && (
              <p className="text-xs text-red-400 bg-red-900/20 border border-red-800/40 rounded-lg px-3 py-2">
                {error}
              </p>
            )}

            {/* Buttons */}
            <div className="flex gap-3 pt-1">
              <button
                onClick={handleSave}
                disabled={saving}
                className="flex-1 py-2.5 rounded-xl text-sm font-semibold
                           bg-indigo-600 hover:bg-indigo-500 text-white
                           disabled:opacity-40 transition-all flex items-center justify-center gap-2"
              >
                {saving ? <><Spinner /> Applying...</> : "Apply"}
              </button>
              <button
                onClick={onClose}
                className="py-2.5 px-4 rounded-xl text-sm font-semibold
                           bg-slate-700 hover:bg-slate-600 text-gray-300 transition-all"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function CloseIcon() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
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
