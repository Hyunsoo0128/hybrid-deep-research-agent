"use client";

import { useState, useEffect } from "react";
import { indexFiles, getFileStatus, deleteFileIndex } from "@/lib/api";
import type { FileIndexStatus } from "@/lib/types";

export default function FileIndexer() {
  const [open, setOpen] = useState(false);
  const [path, setPath] = useState("");
  const [recursive, setRecursive] = useState(true);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<FileIndexStatus | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) loadStatus();
  }, [open]);

  const loadStatus = async () => {
    try {
      const s = await getFileStatus();
      setStatus(s);
    } catch {
      // Ignore if file server is unavailable
    }
  };

  const handleIndex = async () => {
    if (!path.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const res = await indexFiles(path.trim(), recursive);
      setResult(`${res.total_chunks} chunks indexed from ${res.indexed_files} files`);
      await loadStatus();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Indexing failed");
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!confirm("Delete the local file index?")) return;
    setLoading(true);
    try {
      await deleteFileIndex();
      setStatus(null);
      setResult("Index deleted successfully");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Deletion failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-300
                   transition-colors w-full py-2"
      >
        <FolderIcon />
        <span>Local File Search Settings</span>
        <ChevronIcon open={open} />
        {status?.local_search_enabled && (
          <span className="ml-auto text-xs text-emerald-400 bg-emerald-900/30 px-2 py-0.5 rounded-full">
            {status.total_chunks} chunks indexed
          </span>
        )}
      </button>

      {open && (
        <div className="mt-2 p-4 bg-slate-800 rounded-xl border border-slate-700 space-y-4">
          <p className="text-xs text-gray-500">
            Indexing PDF, DOCX, TXT, MD, and code files enables searching local files during research.
          </p>

          <div className="space-y-3">
            <div>
              <label className="text-xs text-gray-500 block mb-1">
                Directory path to index — enter an absolute path, then click <strong>Start Indexing</strong>
              </label>
              <input
                value={path}
                onChange={(e) => setPath(e.target.value)}
                placeholder="/Users/me/documents/papers"
                disabled={loading}
                className="w-full bg-slate-700 text-sm text-gray-100 px-3 py-2 rounded-lg
                           border border-slate-600 focus:outline-none focus:ring-1 focus:ring-indigo-500
                           placeholder-gray-600 disabled:opacity-50"
              />
            </div>

            <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
              <input
                type="checkbox"
                checked={recursive}
                onChange={(e) => setRecursive(e.target.checked)}
                className="rounded border-slate-600 bg-slate-700 text-indigo-500
                           focus:ring-indigo-500"
              />
              Include subfolders
            </label>
          </div>

          <div className="flex gap-2">
            <button
              onClick={handleIndex}
              disabled={loading}
              className="flex-1 py-2 px-4 rounded-lg text-sm font-medium
                         bg-indigo-600 hover:bg-indigo-500 text-white
                         disabled:opacity-40 disabled:cursor-not-allowed
                         transition-colors flex items-center justify-center gap-2"
            >
              {loading ? <><Spinner /> Indexing...</> : "Start Indexing"}
            </button>
            {status?.local_search_enabled && (
              <button
                onClick={handleDelete}
                disabled={loading}
                className="py-2 px-4 rounded-lg text-sm
                           bg-red-900/30 hover:bg-red-900/50 text-red-400
                           border border-red-800/50 disabled:opacity-40 transition-colors"
              >
                Delete
              </button>
            )}
          </div>

          {result && (
            <p className="text-xs text-emerald-400 bg-emerald-900/20 px-3 py-2 rounded-lg">
              ✓ {result}
            </p>
          )}
          {error && (
            <p className="text-xs text-red-400 bg-red-900/20 px-3 py-2 rounded-lg">
              ✗ {error}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function FolderIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V7z" />
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

function Spinner() {
  return (
    <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}
