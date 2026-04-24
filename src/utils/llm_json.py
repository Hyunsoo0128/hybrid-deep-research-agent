"""
DSAP Guard Functions — Safe LLM JSON parsing wrapper

Based on arxiv:2512.20660 (DSAP):

Level 1 — Context Refinement (implemented Phase 1):
  On JSON parse failure, retry with error message + schema hint fed back to LLM.
  System prompt switched to strict JSON-only mode on retry.

Level 2 — Informed Backtracking (implemented Phase 2-3):
  Stagnation detection: if consecutive attempts produce the same normalized error
  fingerprint, the repeating strategy is not working. Instead of wasting the final
  retry on the same approach, switch to a last-resort minimal prompt (clean slate,
  schema only, no error context).
  Stagnation events are recorded to an optional error_sink list for observability.

Level 3 — Escalation (out of scope):
  Full workflow restart with modified context. Not implemented (requires pipeline
  re-entry point and state management beyond this utility's scope).
"""

from __future__ import annotations
import json
import re

from ..providers.base import LLMProvider

# ── Prompt templates ──────────────────────────────────────────────────────────

_RETRY_SYSTEM = "You must respond with ONLY valid JSON. No markdown, no explanations, no code blocks."

_RETRY_USER = """Previous response could not be parsed as JSON.
Parse error: {error}

Respond with ONLY valid JSON matching this schema:
{schema_hint}"""

# Last-resort: clean slate with no accumulated error context.
# Intentionally shorter and simpler than _RETRY_USER — the error context may be
# confusing the model; removing it is the strategy change.
_LAST_RESORT_SYSTEM = "Output ONLY valid JSON. Nothing else. No markdown. No code blocks."

_LAST_RESORT_USER = """Output ONLY valid JSON using this exact schema structure (fill with appropriate values):
{schema_hint}"""


# ── Stagnation detection helper ───────────────────────────────────────────────

def _error_fingerprint(e: Exception) -> str:
    """
    Normalize exception message to detect stagnation across retries.

    Strips position-specific info (line/column/char numbers) that vary between
    attempts even for the same structural error, causing false non-stagnation.

    Example:
      "Expecting ',' delimiter: line 3 column 5 (char 42)"
      "Expecting ',' delimiter: line 4 column 2 (char 51)"
      → both → "JSONDecodeError:Expecting ',' delimiter: line X"
    """
    msg = str(e)
    msg = re.sub(r"line \d+ column \d+ \(char \d+\)", "line X", msg)
    msg = re.sub(r"char \d+", "char X", msg)
    return f"{type(e).__name__}:{msg[:60]}"


# ── Main guard function ───────────────────────────────────────────────────────

async def llm_json(
    llm: LLMProvider,
    messages: list[dict],
    system: str,
    schema_hint: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    max_retries: int = 2,
    fallback: dict | None = None,
    dsap_enabled: bool = True,
    error_sink: list | None = None,
    caller_tag: str = "",
) -> dict:
    """
    LLM call wrapper that guarantees a JSON response.

    DSAP Level 1: On parse failure, retry with error context (up to max_retries times).
    DSAP Level 2: Stagnation detection — if the same error fingerprint repeats on
      consecutive attempts, switch to last-resort minimal prompt instead of repeating
      the failing strategy.

    Args:
      error_sink: optional list; stagnation events are appended as dicts.
                  Caller is responsible for lifecycle (e.g., add to retrieval_quality).
      caller_tag: identifier for error_sink records (e.g., "mass_rag/synthesis").

    Returns the parsed dict, or fallback (empty dict if no fallback provided).
    """
    if not dsap_enabled:
        # Single attempt, no retry
        raw = await llm.complete(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return fallback if fallback is not None else {}

    current_messages = list(messages)
    current_system = system
    prev_fingerprint: str | None = None
    last_resort_used = False

    for attempt in range(max_retries + 1):
        raw = await llm.complete(
            messages=current_messages,
            system=current_system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        try:
            cleaned = (
                raw.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            if attempt >= max_retries:
                break  # exhausted all attempts

            fingerprint = _error_fingerprint(e)

            # ── Level 2: Stagnation detection ────────────────────────────────
            # Same error type+structure on consecutive attempts → strategy not working.
            # Switch to last-resort (clean slate) instead of repeating error-feedback loop.
            if prev_fingerprint is not None and fingerprint == prev_fingerprint and not last_resort_used:
                if error_sink is not None:
                    error_sink.append({
                        "caller_tag": caller_tag,
                        "error_type": type(e).__name__,
                        "error_msg": str(e)[:120],
                        "attempt": attempt,
                        "action": "last_resort",
                    })
                # Fresh messages: no accumulated error context (that's the strategy change)
                current_messages = list(messages) + [
                    {"role": "user", "content": _LAST_RESORT_USER.format(schema_hint=schema_hint)},
                ]
                current_system = _LAST_RESORT_SYSTEM
                last_resort_used = True

            else:
                # ── Level 1: Context Refinement ───────────────────────────────
                # Append assistant response + error feedback to message chain.
                current_messages = current_messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": _RETRY_USER.format(
                            error=str(e),
                            schema_hint=schema_hint,
                        ),
                    },
                ]
                current_system = _RETRY_SYSTEM

            prev_fingerprint = fingerprint

    return fallback if fallback is not None else {}
