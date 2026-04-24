"use client";

import { useState, useRef } from "react";

interface Props {
  onSubmit: (query: string) => void;
  loading: boolean;
}

const EXAMPLE_QUERIES = [
  "Current trends and practical challenges in quantum computing",
  "Impact of generative AI on software development careers",
  "Comparative analysis of lithium battery alternative technologies",
];

export default function QueryInput({ onSubmit, loading }: Props) {
  const [query, setQuery] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !loading) onSubmit(query.trim());
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleSubmit(e);
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="relative">
          <textarea
            ref={textareaRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="What would you like to research?"
            rows={3}
            disabled={loading}
            className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-xl
                       text-gray-100 placeholder-gray-500 resize-none
                       focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent
                       disabled:opacity-50 disabled:cursor-not-allowed
                       text-base leading-relaxed"
          />
          <span className="absolute bottom-3 right-3 text-xs text-gray-600">
            ⌘↵ Run
          </span>
        </div>

        <button
          type="submit"
          disabled={!query.trim() || loading}
          className="w-full py-3 px-6 rounded-xl font-semibold text-sm
                     bg-indigo-600 hover:bg-indigo-500 text-white
                     disabled:opacity-40 disabled:cursor-not-allowed
                     transition-all duration-150 flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Spinner />
              Generating plan...
            </>
          ) : (
            <>
              <SearchIcon />
              Start Research
            </>
          )}
        </button>
      </form>

      {/* Example queries */}
      <div className="mt-6">
        <p className="text-xs text-gray-600 mb-2">Example queries</p>
        <div className="flex flex-col gap-2">
          {EXAMPLE_QUERIES.map((q) => (
            <button
              key={q}
              onClick={() => setQuery(q)}
              disabled={loading}
              className="text-left text-sm text-gray-400 hover:text-indigo-400
                         px-3 py-2 rounded-lg hover:bg-slate-800
                         transition-colors duration-100 disabled:opacity-50"
            >
              → {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function SearchIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
    </svg>
  );
}

function Spinner() {
  return (
    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10"
        stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}
