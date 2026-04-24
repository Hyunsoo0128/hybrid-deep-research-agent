"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage, Source } from "@/lib/types";
import { sendChatMessage } from "@/lib/api";

interface Props {
  sessionId: string;
  messages: ChatMessage[];
  onMessage: (msg: ChatMessage) => void;
}

const ROUTE_LABELS: Record<string, { label: string; color: string }> = {
  memory: { label: "Report-based", color: "text-emerald-400 bg-emerald-900/20" },
  targeted: { label: "Additional Search", color: "text-indigo-400 bg-indigo-900/20" },
  new_research: { label: "New Research Required", color: "text-amber-400 bg-amber-900/20" },
};

export default function ChatInterface({ sessionId, messages, onMessage }: Props) {
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [currentRoute, setCurrentRoute] = useState<string | null>(null);
  const [searchingSources, setSearchingSources] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  const handleSend = async () => {
    const msg = input.trim();
    if (!msg || sending) return;

    setInput("");
    setSending(true);
    setStreamingText("");
    setCurrentRoute(null);
    setSearchingSources(false);

    onMessage({ role: "user", content: msg });

    let accumulated = "";
    const newSources: Source[] = [];
    let finalRoute = "memory";

    try {
      await sendChatMessage(sessionId, msg, {
        onRouting: (route) => {
          setCurrentRoute(route);
          finalRoute = route;
        },
        onTargetedSearch: () => setSearchingSources(true),
        onSourceFound: (src) => {
          newSources.push(src);
          setSearchingSources(false);
        },
        onChunk: (text) => {
          accumulated += text;
          setStreamingText(accumulated);
        },
        onComplete: (route) => {
          finalRoute = route;
          onMessage({
            role: "assistant",
            content: accumulated,
            route: finalRoute,
            sources: newSources.length > 0 ? newSources : undefined,
          });
          setStreamingText("");
          setCurrentRoute(null);
        },
        onError: (err) => {
          onMessage({ role: "assistant", content: `Error: ${err}`, route: "error" });
          setStreamingText("");
        },
      });
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        {/* Header */}
        <div className="px-4 py-3 border-b border-slate-700 flex items-center gap-2">
          <ChatIcon />
          <h3 className="text-sm font-semibold text-gray-300">Follow-up Questions</h3>
          <span className="text-xs text-gray-600">Ask additional questions about the report</span>
        </div>

        {/* Message list */}
        <div className="p-4 space-y-4 max-h-96 overflow-y-auto">
          {messages.length === 0 && !streamingText && (
            <p className="text-xs text-gray-600 text-center py-4">
              Ask anything you want to know about the report.
            </p>
          )}

          {messages.map((msg, i) => (
            <MessageBubble key={i} message={msg} />
          ))}

          {/* Live streaming assistant response */}
          {streamingText && (
            <div className="flex flex-col gap-1.5">
              {currentRoute && (
                <RouteBadge route={currentRoute} />
              )}
              <div className="bg-slate-700/50 rounded-xl rounded-tl-sm px-4 py-3 text-sm text-gray-200">
                <div className="prose prose-sm max-w-none cursor-blink">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {streamingText}
                  </ReactMarkdown>
                </div>
              </div>
            </div>
          )}

          {/* Searching indicator */}
          {searchingSources && !streamingText && (
            <div className="flex items-center gap-2 text-xs text-gray-500 pl-1">
              <Spinner />
              Searching for additional information...
            </div>
          )}

          {/* Routing indicator (before response) */}
          {currentRoute && !streamingText && !searchingSources && (
            <div className="flex items-center gap-2 text-xs text-gray-500 pl-1">
              <Spinner />
              <RouteBadge route={currentRoute} />
              <span>Generating answer...</span>
            </div>
          )}

          <div ref={endRef} />
        </div>

        {/* Input area */}
        <div className="px-4 py-3 border-t border-slate-700 flex gap-3 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter a follow-up question... (Enter to send, Shift+Enter for new line)"
            rows={2}
            disabled={sending}
            className="flex-1 bg-slate-700 text-sm text-gray-100 px-3 py-2 rounded-lg
                       placeholder-gray-600 resize-none border border-slate-600
                       focus:outline-none focus:ring-1 focus:ring-indigo-500
                       disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || sending}
            className="shrink-0 p-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500
                       text-white disabled:opacity-40 disabled:cursor-not-allowed
                       transition-colors"
          >
            {sending ? <Spinner /> : <SendIcon />}
          </button>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="bg-indigo-600/30 border border-indigo-500/30 rounded-xl rounded-tr-sm
                        px-4 py-2.5 max-w-[80%] text-sm text-gray-200">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1.5">
      {message.route && <RouteBadge route={message.route} />}
      <div className="bg-slate-700/50 rounded-xl rounded-tl-sm px-4 py-3 text-sm text-gray-200 max-w-[90%]">
        <div className="prose prose-sm max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        </div>
        {message.sources && message.sources.length > 0 && (
          <div className="mt-3 pt-3 border-t border-slate-600 flex flex-wrap gap-2">
            {message.sources.map((src, i) => (
              <a
                key={i}
                href={src.source_type === "local" ? undefined : src.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-indigo-400 hover:text-indigo-300 underline underline-offset-2 line-clamp-1 max-w-[200px]"
              >
                {src.title || src.url}
              </a>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function RouteBadge({ route }: { route: string }) {
  const info = ROUTE_LABELS[route] || { label: route, color: "text-gray-400 bg-slate-700" };
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-medium w-fit ${info.color}`}>
      {info.label}
    </span>
  );
}

function ChatIcon() {
  return (
    <svg className="w-4 h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
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
