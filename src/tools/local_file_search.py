"""
Local File Search — Phase 3

Provides file indexing + vector search in a single class.

Supported formats: PDF, DOCX, TXT, MD, Python, JS/TS, etc.
Vector storage: Qdrant (local file-based, no server required)
Embeddings: fastembed (BAAI/bge-small-en-v1.5, multilingual model selectable)

Usage flow:
  1. Create LocalFileSearch() instance (Qdrant collection auto-initialized)
  2. Index files with index_directory(path)
  3. Semantic search with search(query)
"""

from __future__ import annotations
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import warnings
from qdrant_client import QdrantClient

COLLECTION_NAME = "research_local"
CHUNK_SIZE = 1000       # chunk size (characters)
CHUNK_OVERLAP = 200     # chunk overlap
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"   # 130MB, English-first
MULTILINGUAL_MODEL = "intfloat/multilingual-e5-small"  # 120MB, includes Korean

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".rst",
    ".docx", ".doc",
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".html", ".htm", ".csv", ".json",
}


@dataclass
class LocalSearchResult:
    filepath: str
    filename: str
    excerpt: str        # relevant chunk text
    score: float        # similarity score (0~1)
    chunk_index: int


class LocalFileSearch:
    """
    Local file search based on Qdrant + fastembed.
    Serverless: stores to local disk via QdrantClient(path=...).
    """

    def __init__(
        self,
        qdrant_path: str = "./data/qdrant",
        embedding_model: str | None = None,
    ):
        os.makedirs(qdrant_path, exist_ok=True)
        self._client = QdrantClient(path=qdrant_path)

        # Prefer multilingual model for Korean environments
        self._model = embedding_model or os.getenv("EMBED_MODEL", DEFAULT_MODEL)
        self._client.set_model(self._model)

        self._ensure_collection()

    # ── Collection initialization ────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        if not self._client.collection_exists(COLLECTION_NAME):
            self._client.create_collection(
                COLLECTION_NAME,
                vectors_config=self._client.get_fastembed_vector_params(on_disk=False),
            )

    # ── File parsing ────────────────────────────────────────────────────

    def _parse_file(self, filepath: Path) -> str:
        ext = filepath.suffix.lower()
        try:
            if ext == ".pdf":
                return self._parse_pdf(filepath)
            elif ext in {".docx", ".doc"}:
                return self._parse_docx(filepath)
            else:
                return filepath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return ""  # return empty string on parse failure (skip)

    def _parse_pdf(self, filepath: Path) -> str:
        from pypdf import PdfReader
        reader = PdfReader(str(filepath))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)

    def _parse_docx(self, filepath: Path) -> str:
        from docx import Document
        doc = Document(str(filepath))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    # ── Chunking ────────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> list[str]:
        """
        Fixed-size chunking with overlap.
        Removes empty chunks and chunks shorter than 50 characters.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end].strip()
            if len(chunk) >= 50:
                chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    # ── Indexing ───────────────────────────────────────────────────────

    def index_directory(
        self,
        path: str,
        extensions: list[str] | None = None,
        recursive: bool = True,
    ) -> dict:
        """
        Parse files in a directory → chunk → vector index.

        Returns:
            {"indexed_files": N, "total_chunks": M, "skipped": K, "errors": [...]}
        """
        root = Path(path)
        if not root.exists():
            return {"error": f"Path not found: {path}"}

        allowed_exts = {e.lower() for e in (extensions or SUPPORTED_EXTENSIONS)}
        pattern = "**/*" if recursive else "*"
        files = [f for f in root.glob(pattern) if f.is_file() and f.suffix.lower() in allowed_exts]

        docs: list[str] = []
        metadata: list[dict] = []
        ids: list[str] = []
        errors: list[str] = []
        skipped = 0

        for filepath in files:
            text = self._parse_file(filepath)
            if not text.strip():
                skipped += 1
                continue

            chunks = self._chunk_text(text)
            if not chunks:
                skipped += 1
                continue

            for i, chunk in enumerate(chunks):
                docs.append(chunk)
                metadata.append({
                    "text": chunk,                  # for restoring excerpt in search results
                    "filepath": str(filepath),
                    "filename": filepath.name,
                    "extension": filepath.suffix.lower(),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                })
                ids.append(str(uuid.uuid4()))

        if not docs:
            return {
                "indexed_files": 0,
                "total_chunks": 0,
                "skipped": skipped,
                "errors": errors,
            }

        # Batch upsert (Qdrant + fastembed auto embedding)
        # TODO: migrate to qdrant-client 1.17 new API (upsert + models.Document)
        batch_size = 64
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for i in range(0, len(docs), batch_size):
                self._client.add(
                    collection_name=COLLECTION_NAME,
                    documents=docs[i:i + batch_size],
                    metadata=metadata[i:i + batch_size],
                    ids=ids[i:i + batch_size],
                )

        return {
            "indexed_files": len(files) - skipped,
            "total_chunks": len(docs),
            "skipped": skipped,
            "errors": errors,
        }

    def index_file(self, filepath: str) -> dict:
        """Index a single file."""
        return self.index_directory(
            path=str(Path(filepath).parent),
            extensions=[Path(filepath).suffix.lower()],
            recursive=False,
        )

    # ── Search ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> list[LocalSearchResult]:
        """
        Semantic search.
        Results below score_threshold are excluded.
        """
        if not self.has_content():
            return []

        # TODO: migrate to qdrant-client 1.17 new API (query_points + models.Document)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            raw_results = self._client.query(
                collection_name=COLLECTION_NAME,
                query_text=query,
                limit=top_k,
                score_threshold=score_threshold,
            )

        out = []
        for r in raw_results:
            payload = r.metadata or {}
            excerpt = payload.get("text") or r.document or ""
            out.append(LocalSearchResult(
                filepath=payload.get("filepath", ""),
                filename=payload.get("filename", ""),
                excerpt=excerpt,
                score=r.score,
                chunk_index=payload.get("chunk_index", 0),
            ))
        return out

    # ── Status queries ────────────────────────────────────────────────────

    def has_content(self) -> bool:
        """Returns True if at least one document is indexed."""
        try:
            info = self._client.get_collection(COLLECTION_NAME)
            return info.points_count > 0
        except Exception:
            return False

    def get_stats(self) -> dict:
        """Returns collection statistics."""
        try:
            info = self._client.get_collection(COLLECTION_NAME)
            return {
                "total_chunks": info.points_count,
                "collection": COLLECTION_NAME,
                "status": info.status.value if hasattr(info.status, "value") else str(info.status),
            }
        except Exception:
            return {"total_chunks": 0, "collection": COLLECTION_NAME, "status": "not_initialized"}

    def clear(self) -> None:
        """Reset index (delete all)."""
        if self._client.collection_exists(COLLECTION_NAME):
            self._client.delete_collection(COLLECTION_NAME)
        self._ensure_collection()
