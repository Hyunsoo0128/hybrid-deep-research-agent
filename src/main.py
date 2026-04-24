"""
FastAPI + SSE Streaming Server

Flow:
  1. POST /research/start       → Generate plan, wait at interrupt
  2. GET  /research/stream/{id} → Subscribe to SSE with approval payload
                                   → Approval + execution + streaming handled simultaneously
  3. GET  /research/{id}        → Retrieve report after completion
"""

from __future__ import annotations
import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel
from dotenv import load_dotenv

from .graph import build_graph
from .chat_graph import build_chat_graph
from .tools.search import SearchTool
from .tools.local_file_search import LocalFileSearch
from .session_store import SessionStore
from .state import initial_state

load_dotenv()

# ── App Initialization ──────────────────────────────────────────────────────

def _create_llm_provider(config: dict | None = None):
    """Create an LLM provider. Uses environment variables if no config is provided."""
    if config is None:
        config = {
            "provider": os.getenv("LLM_PROVIDER", "bedrock"),
            "model": None,
        }
    provider = config.get("provider", "bedrock").lower()
    if provider == "ollama":
        from .providers.ollama import OllamaProvider
        return OllamaProvider(
            model=config.get("model") or os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            embed_model=config.get("embed_model") or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            host=config.get("host") or os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
    elif provider == "claude":
        from .providers.claude import ClaudeProvider
        return ClaudeProvider(
            model=config.get("model") or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        )
    elif provider == "hybrid":
        return _create_hybrid_provider(config)
    else:  # bedrock (default)
        from .providers.bedrock import BedrockProvider
        return BedrockProvider(
            model=config.get("model") or os.getenv("BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-6"),
            region=config.get("region") or os.getenv("AWS_REGION", "us-west-2"),
        )


def _create_hybrid_provider(config: dict):
    """Build a HybridProvider from HYBRID_CLOUD_PROVIDER / HYBRID_LOCAL_PROVIDER env vars.

    Supported combinations:
      HYBRID_CLOUD_PROVIDER=bedrock  (default) + HYBRID_LOCAL_PROVIDER=ollama (default)
      HYBRID_CLOUD_PROVIDER=claude             + HYBRID_LOCAL_PROVIDER=ollama

    Override models independently:
      HYBRID_CLOUD_MODEL=us.anthropic.claude-sonnet-4-6
      HYBRID_LOCAL_MODEL=qwen3:8b
    """
    from .providers.hybrid import HybridProvider

    cloud_backend = (
        config.get("cloud_provider")
        or os.getenv("HYBRID_CLOUD_PROVIDER", "bedrock")
    ).lower()
    local_backend = (
        config.get("local_provider")
        or os.getenv("HYBRID_LOCAL_PROVIDER", "ollama")
    ).lower()

    # ── Cloud provider ──────────────────────────────────────────────────────
    if cloud_backend == "claude":
        from .providers.claude import ClaudeProvider
        cloud = ClaudeProvider(
            model=config.get("cloud_model") or os.getenv("HYBRID_CLOUD_MODEL")
                  or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        )
    else:  # bedrock (default)
        from .providers.bedrock import BedrockProvider
        cloud = BedrockProvider(
            model=config.get("cloud_model") or os.getenv("HYBRID_CLOUD_MODEL")
                  or os.getenv("BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-6"),
            region=config.get("region") or os.getenv("AWS_REGION", "us-west-2"),
        )

    # ── Local provider ──────────────────────────────────────────────────────
    if local_backend == "ollama":
        from .providers.ollama import OllamaProvider
        local = OllamaProvider(
            model=config.get("local_model") or os.getenv("HYBRID_LOCAL_MODEL")
                  or os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            embed_model=config.get("embed_model") or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            host=config.get("host") or os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
    else:
        raise ValueError(
            f"Unsupported HYBRID_LOCAL_PROVIDER='{local_backend}'. "
            "Only 'ollama' is supported as a local provider."
        )

    return HybridProvider(cloud=cloud, local=local)


def _get_llm_config_info(config: dict | None = None) -> dict:
    """Return the current LLM configuration info."""
    if config is None:
        provider = os.getenv("LLM_PROVIDER", "bedrock").lower()
        if provider == "ollama":
            config = {
                "provider": "ollama",
                "model": os.getenv("OLLAMA_MODEL", "qwen3:8b"),
                "embed_model": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
                "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            }
        elif provider == "claude":
            config = {
                "provider": "claude",
                "model": os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
            }
        elif provider == "hybrid":
            config = {
                "provider": "hybrid",
                "cloud_provider": os.getenv("HYBRID_CLOUD_PROVIDER", "bedrock"),
                "cloud_model": os.getenv("HYBRID_CLOUD_MODEL")
                               or os.getenv("BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-6"),
                "local_provider": os.getenv("HYBRID_LOCAL_PROVIDER", "ollama"),
                "local_model": os.getenv("HYBRID_LOCAL_MODEL")
                               or os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            }
        else:
            config = {
                "provider": "bedrock",
                "model": os.getenv("BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-6"),
                "region": os.getenv("AWS_REGION", "us-west-2"),
            }
    return config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize LLM, search tools, checkpointer, and session store on app startup."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    llm = _create_llm_provider()
    search_tool = SearchTool()
    local_file_search = LocalFileSearch(
        qdrant_path=os.getenv("QDRANT_PATH", "./data/qdrant"),
    )

    # SQLite checkpointer (recovers research sessions after server restart)
    db_path = os.getenv("CHECKPOINT_DB", "./data/checkpoints.db")
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph(llm, search_tool, local_file_search, checkpointer=checkpointer)
        chat_graph = build_chat_graph(llm, search_tool)

        # Session metadata store (recovers after restart)
        session_store = SessionStore(
            path=os.getenv("SESSIONS_FILE", "./data/sessions.json")
        )

        app.state.graph = graph
        app.state.chat_graph = chat_graph
        app.state.local_file_search = local_file_search
        app.state.search_tool = search_tool
        app.state.checkpointer = checkpointer
        app.state.session_store = session_store
        app.state.llm_config = _get_llm_config_info()
        # in-memory sessions dict: initialized with data recovered from file
        app.state.sessions: dict[str, dict] = session_store.all_sessions()

        yield
        # checkpointer connection is automatically cleaned up when async with block exits


app = FastAPI(title="Deep Research API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ─────────────────────────────────────────────────

class StartRequest(BaseModel):
    query: str
    feature_flags: dict | None = None


class ApproveRequest(BaseModel):
    session_id: str
    approved: bool
    plan: dict | None = None          # Modified plan (keeps original if not provided)
    report_length: str = "detailed"   # brief | standard | detailed


class StreamRequest(BaseModel):
    approved: bool = True
    plan: dict | None = None


class IndexRequest(BaseModel):
    path: str                           # Directory or file path to index
    recursive: bool = True
    extensions: list[str] | None = None  # None means all supported formats


class ChatRequest(BaseModel):
    message: str


class SettingsRequest(BaseModel):
    provider: str                       # bedrock | claude | ollama | hybrid
    model: str | None = None
    embed_model: str | None = None      # ollama only
    host: str | None = None             # ollama only
    region: str | None = None           # bedrock only
    # hybrid only
    cloud_provider: str | None = None   # bedrock | claude
    cloud_model: str | None = None
    local_model: str | None = None      # ollama model


# ── SSE Event Helper ───────────────────────────────────────────────────────

def sse_event(event_type: str, data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/research/start")
async def start_research(req: StartRequest):
    """
    Step 1: Receive query → generate plan → return to client.
    Plan approval is handled separately via /research/approve.
    """
    session_id = uuid.uuid4().hex
    graph = app.state.graph

    # Auto-enable if local files are indexed
    local_file_search: LocalFileSearch = app.state.local_file_search
    local_search_enabled = local_file_search.has_content()

    state = initial_state(
        session_id=session_id,
        query=req.query,
        local_search_enabled=local_search_enabled,
        feature_flags=req.feature_flags,
    )
    config = {"configurable": {"thread_id": session_id}}

    # Execute up to generate_plan → plan_review(interrupt)
    # Returns None when stopped at interrupt
    result = await graph.ainvoke(state, config=config)

    # Retrieve plan from interrupt state
    snapshot = await graph.aget_state(config)
    interrupt_value = None
    for task in snapshot.tasks:
        if task.interrupts:
            interrupt_value = task.interrupts[0].value
            break

    if not interrupt_value:
        raise HTTPException(status_code=500, detail="Plan generation failed")

    from datetime import datetime, timezone
    session = {
        "config": config,
        "status": "plan_review",
        "query": req.query,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "conversation_history": [],
    }
    app.state.sessions[session_id] = session
    await app.state.session_store.save(session_id, session)

    return {
        "session_id": session_id,
        "plan": interrupt_value["plan"],
        "message": interrupt_value["message"],
    }


@app.post("/research/approve")
async def approve_plan(req: ApproveRequest):
    """
    Save plan approval info to session.
    Actual execution starts when SSE connection is made at /research/stream/{id}.
    """
    session_id = req.session_id
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = app.state.sessions[session_id]
    session["approval"] = {
        "approved": req.approved,
        "plan": req.plan,
        "report_length": req.report_length,
    }
    session["status"] = "approved"
    await app.state.session_store.save(session_id, session)

    return {"session_id": session_id, "status": "approved"}


@app.get("/research/stream/{session_id}")
async def stream_research(session_id: str):
    """
    SSE stream.
    Connecting to this endpoint after approval completes
    handles interrupt resumption + execution + streaming simultaneously.

    Event types:
      search_started    — Sub-query search started
      source_found      — Citation found
      synthesis_started — Report writing started
      critique_started  — Quality review started
      report_chunk      — Report text (streaming)
      report_complete   — Completed
      error             — Error
    """
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = app.state.sessions[session_id]
    if session.get("status") != "approved":
        raise HTTPException(status_code=400, detail="Plan not yet approved. Call /research/approve first.")
    graph = app.state.graph
    config = session["config"]
    approval = session.get("approval", {"approved": True, "plan": None})

    async def event_generator() -> AsyncIterator[str]:
        total_sources = 0
        try:
            async for event in graph.astream_events(
                Command(resume=approval),
                config=config,
                version="v2",
            ):
                kind = event.get("event", "")
                name = event.get("name", "")
                data = event.get("data", {})

                # Send SSE keepalive comment on every event to prevent browser timeout
                # during long-running nodes (write_draft, critique).
                yield ": keepalive\n\n"

                # ── Search fan-out started ────────────────────────────────
                if kind == "on_chain_start" and name == "search_orchestrator":
                    yield sse_event("search_started", {
                        "message": "Starting parallel sub-query search..."
                    })

                # ── Parallel worker completed → source_found ──────────────
                elif kind == "on_chain_end" and name in ("search_worker", "local_search_worker"):
                    citations = (data.get("output") or {}).get("citations", [])
                    for c in citations:
                        total_sources += 1
                        yield sse_event("source_found", {
                            "title": c.get("title"),
                            "url": c.get("url"),
                            "trust_level": c.get("trust_level"),
                            "confidence": c.get("confidence"),
                            "source_type": c.get("source_type", "web"),
                        })

                # ── Gap detection completed ────────────────────────────────
                elif kind == "on_chain_end" and name == "gap_detector":
                    output = data.get("output") or {}
                    gaps = output.get("identified_gaps", [])
                    gap_queries = output.get("gap_queries", [])
                    if gaps:
                        yield sse_event("gap_detected", {
                            "gaps": gaps,
                            "additional_queries": gap_queries,
                            "message": f"{len(gaps)} knowledge gap(s) found — proceeding with additional search.",
                        })

                # ── Gap search started/completed ───────────────────────────
                elif kind == "on_chain_start" and name == "gap_search":
                    yield sse_event("gap_search_started", {
                        "message": "Running additional search for gap coverage..."
                    })

                elif kind == "on_chain_end" and name == "gap_search":
                    citations = (data.get("output") or {}).get("citations", [])
                    for c in citations:
                        total_sources += 1
                        yield sse_event("source_found", {
                            "title": c.get("title"),
                            "url": c.get("url"),
                            "trust_level": c.get("trust_level"),
                            "confidence": c.get("confidence"),
                        })

                # ── Cross-validation ───────────────────────────────────────
                elif kind == "on_chain_start" and name == "cross_validator":
                    yield sse_event("validation_started", {
                        "message": "Cross-validating sources..."
                    })

                elif kind == "on_chain_end" and name == "cross_validator":
                    report = (data.get("output") or {}).get("cross_validation_report") or {}
                    yield sse_event("validation_complete", {
                        "quality_score": report.get("quality_score", 0),
                        "well_corroborated": report.get("well_corroborated_count", 0),
                        "contradictions": report.get("contradictions_found", 0),
                        "summary": report.get("summary", ""),
                    })

                # ── Report writing ─────────────────────────────────────────
                elif kind == "on_chain_start" and name == "write_draft":
                    yield sse_event("synthesis_started", {
                        "message": "Writing report..."
                    })

                # ── Quality review ─────────────────────────────────────────
                elif kind == "on_chain_start" and name == "critique":
                    yield sse_event("critique_started", {
                        "message": "Reviewing report quality..."
                    })

                # ── Completed ─────────────────────────────────────────────
                elif kind == "on_chain_end" and name == "finalize":
                    final_report = (data.get("output") or {}).get("final_report", "")

                    chunk_size = 100
                    for i in range(0, len(final_report), chunk_size):
                        yield sse_event("report_chunk", {"text": final_report[i:i + chunk_size]})
                        await asyncio.sleep(0.02)

                    yield sse_event("report_complete", {
                        "session_id": session_id,
                        "total_sources": total_sources,
                    })

                    # Persist completed status
                    session["status"] = "complete"
                    await app.state.session_store.save(session_id, session)
                    return

        except Exception as e:
            session["status"] = "error"
            await app.state.session_store.save(session_id, session)
            yield sse_event("error", {"message": str(e), "recoverable": False})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/research/{session_id}/chat")
async def chat(session_id: str, req: ChatRequest):
    """
    Phase 4: Conversation mode after research is complete.

    Event types:
      chat_routing    — Question classification result (memory/targeted/new_research)
      targeted_search — Additional search started
      source_found    — Sources found during additional search
      chat_chunk      — Answer text (streaming)
      chat_complete   — Answer completed
      new_research    — Indicates the question requires new research
      error           — Error
    """
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = app.state.sessions[session_id]
    graph = app.state.graph
    chat_graph = app.state.chat_graph
    config = session["config"]

    # Read completed research state
    research_state = (await graph.aget_state(config)).values
    if not research_state.get("final_report"):
        raise HTTPException(status_code=400, detail="Research is not yet complete")

    # Build research context
    research_context = {
        "query":               research_state.get("original_query", ""),
        "final_report":        research_state.get("final_report", ""),
        "citations":           research_state.get("citations", []),
        "plan_interpretation": (research_state.get("plan") or {}).get("interpretation", ""),
    }

    # Conversation history (accumulated in session)
    conversation_history: list[dict] = session.setdefault("conversation_history", [])

    chat_state = {
        "session_id":           session_id,
        "message":              req.message,
        "research_context":     research_context,
        "conversation_history": conversation_history,
        "route":                "",
        "response":             "",
        "extra_citations":      [],
    }

    async def event_generator() -> AsyncIterator[str]:
        response_text = ""
        is_new_research = False
        try:
            async for event in chat_graph.astream_events(
                chat_state,
                version="v2",
            ):
                kind = event.get("event", "")
                name = event.get("name", "")
                data = event.get("data", {})

                # Routing result
                if kind == "on_chain_end" and name == "router":
                    route = (data.get("output") or {}).get("route", "memory")
                    yield sse_event("chat_routing", {"route": route})

                # Additional search started
                elif kind == "on_chain_start" and name == "targeted_search":
                    yield sse_event("targeted_search", {
                        "message": "Searching for additional information..."
                    })

                # Additional search sources
                elif kind == "on_chain_end" and name == "targeted_search":
                    extra = (data.get("output") or {}).get("extra_citations", [])
                    for c in extra:
                        yield sse_event("source_found", {
                            "title": c.get("title"),
                            "url":   c.get("url"),
                            "trust_level": c.get("trust_level"),
                            "confidence":  c.get("confidence"),
                            "source_type": "web",
                        })

                # Answer streaming
                elif kind == "on_chain_end" and name == "memory_answer":
                    response_text = (data.get("output") or {}).get("response", "")
                    chunk_size = 80
                    for i in range(0, len(response_text), chunk_size):
                        yield sse_event("chat_chunk", {"text": response_text[i:i + chunk_size]})
                        await asyncio.sleep(0.02)
                    yield sse_event("chat_complete", {"session_id": session_id})

                # New research required signal
                elif kind == "on_chain_end" and name == "new_research_signal":
                    is_new_research = True
                    response_text = (data.get("output") or {}).get("response", "")
                    yield sse_event("new_research", {
                        "message": response_text,
                        "suggested_query": req.message,
                    })

        except Exception as e:
            yield sse_event("error", {"message": str(e), "recoverable": True})
            return

        # Update conversation history + persist (new_research not recorded)
        if response_text and not is_new_research:
            conversation_history.append({"role": "user",      "content": req.message})
            conversation_history.append({"role": "assistant", "content": response_text})
            # Keep at most 20 turns
            if len(conversation_history) > 40:
                session["conversation_history"] = conversation_history[-40:]
            await app.state.session_store.save(session_id, session)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/research/{session_id}/history")
async def get_chat_history(session_id: str):
    """Retrieve conversation history."""
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    history = app.state.sessions[session_id].get("conversation_history", [])
    return {"session_id": session_id, "history": history, "turn_count": len(history) // 2}


@app.get("/research/{session_id}")
async def get_report(session_id: str):
    """Retrieve the final report."""
    if session_id not in app.state.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    graph = app.state.graph
    config = app.state.sessions[session_id]["config"]
    state = (await graph.aget_state(config)).values

    return {
        "session_id": session_id,
        "query": state.get("original_query"),
        "final_report": state.get("final_report"),
        "citations": state.get("citations", []),
        "revision_count": state.get("revision_count", 0),
    }


# ── Local File Indexing (Phase 3) ─────────────────────────────────────────

@app.post("/files/index")
async def index_files(req: IndexRequest):
    """
    Add local files to the vector index.
    After indexing, local search is automatically enabled for all research sessions.

    Examples:
      {"path": "/Users/me/Documents", "recursive": true}
      {"path": "/Users/me/reports/q1.pdf"}
    """
    local_file_search: LocalFileSearch = app.state.local_file_search
    result = local_file_search.index_directory(
        path=req.path,
        extensions=req.extensions,
        recursive=req.recursive,
    )
    result["local_search_enabled"] = local_file_search.has_content()
    return result


@app.get("/files/status")
async def files_status():
    """Retrieve statistics for currently indexed local files."""
    local_file_search: LocalFileSearch = app.state.local_file_search
    stats = local_file_search.get_stats()
    stats["local_search_enabled"] = local_file_search.has_content()
    return stats


@app.delete("/files/index")
async def clear_index():
    """Clear the entire vector index."""
    local_file_search: LocalFileSearch = app.state.local_file_search
    local_file_search.clear()
    return {"status": "cleared"}


# ── Session List ───────────────────────────────────────────────────────────

@app.get("/sessions")
async def list_sessions():
    """List all research sessions (newest first)."""
    sessions = app.state.session_store.all_sessions()
    result = []
    for sid, s in sessions.items():
        result.append({
            "session_id": sid,
            "query": s.get("query", ""),
            "status": s.get("status", ""),
            "created_at": s.get("created_at", ""),
        })
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return {"sessions": result}


# ── LLM Settings ───────────────────────────────────────────────────────────

@app.get("/settings")
async def get_settings():
    """Retrieve current LLM configuration."""
    return app.state.llm_config


@app.post("/settings")
async def update_settings(req: SettingsRequest):
    """Switch the LLM provider at runtime."""
    config = {
        "provider": req.provider,
        "model": req.model,
        "embed_model": req.embed_model,
        "host": req.host,
        "region": req.region,
        "cloud_provider": req.cloud_provider,
        "cloud_model": req.cloud_model,
        "local_model": req.local_model,
    }
    # Create new LLM instance
    new_llm = _create_llm_provider(config)
    # Rebuild graph (keeping checkpointer)
    app.state.graph = build_graph(
        new_llm,
        app.state.search_tool,
        app.state.local_file_search,
        checkpointer=app.state.checkpointer,
    )
    app.state.chat_graph = build_chat_graph(new_llm, app.state.search_tool)
    # Save the actually applied configuration
    app.state.llm_config = _get_llm_config_info(config)
    return {"status": "ok", "config": app.state.llm_config}


# ── Health Check ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    local_file_search: LocalFileSearch = app.state.local_file_search
    return {
        "status": "ok",
        "local_search_enabled": local_file_search.has_content(),
        "llm": app.state.llm_config,
    }


# ── Local Run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
