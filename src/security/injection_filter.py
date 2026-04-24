"""
Prompt injection filter

Detects malicious instruction patterns in web crawl results
and automatically classifies trust levels.

References: arXiv:2302.12173, AgentDojo(arXiv:2406.13352)
"""

from __future__ import annotations
import re
from dataclasses import dataclass

# Injection patterns (case-insensitive)
_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|prior)\s+instructions?",
    r"disregard\s+(previous|all|prior)\s+instructions?",
    r"system\s*:\s*you\s+are",
    r"\[SYSTEM\]",
    r"\[INST\]",
    r"you\s+are\s+now\s+",
    r"new\s+persona",
    r"forget\s+(everything|all)\s+",
    r"이전\s*(지시|명령|설정).*무시",
    r"시스템.*역할.*변경",
    r"새로운\s+역할",
    r"assistant\s*:\s*sure",          # common jailbreak pattern
]

# HIGH trust level domains
_HIGH_TRUST_DOMAINS = {
    "anthropic.com", "openai.com", "deepmind.com", "microsoft.com",
    "arxiv.org", "github.com", "wikipedia.org", "nature.com",
    "python.org", "docs.python.org", "langchain.com", "langgraph.com",
}

# MEDIUM trust level domains
_MEDIUM_TRUST_DOMAINS = {
    "medium.com", "towardsdatascience.com", "huggingface.co",
    "stackoverflow.com", "reddit.com", "ycombinator.com",
    "techcrunch.com", "wired.com", "theverge.com",
}


@dataclass
class FilterResult:
    safe: bool
    trust_level: str          # high | medium | low
    flagged_patterns: list[str]
    sanitized_content: str


class InjectionFilter:

    def check(self, content: str, url: str = "") -> FilterResult:
        """
        Detect injection patterns in web content.
        Replaces matching sentences with [FILTERED] when found.
        """
        flagged = []
        sanitized = content

        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                flagged.append(pattern)
                # mask the matching line
                sanitized = re.sub(
                    pattern,
                    "[FILTERED: potential injection]",
                    sanitized,
                    flags=re.IGNORECASE,
                )

        trust_level = self._assign_trust(url)

        return FilterResult(
            safe=len(flagged) == 0,
            trust_level=trust_level,
            flagged_patterns=flagged,
            sanitized_content=sanitized,
        )

    def _assign_trust(self, url: str) -> str:
        if not url:
            return "low"
        domain = self._extract_domain(url)
        if any(d in domain for d in _HIGH_TRUST_DOMAINS):
            return "high"
        if any(d in domain for d in _MEDIUM_TRUST_DOMAINS):
            return "medium"
        return "low"

    @staticmethod
    def _extract_domain(url: str) -> str:
        url = url.lower().replace("https://", "").replace("http://", "")
        return url.split("/")[0]
