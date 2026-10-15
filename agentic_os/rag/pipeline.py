"""Retrieval-Augmented Generation.

Roadmap stage 10: documents are chunked, indexed twice (BM25 keywords and
vector embeddings) and retrieved with *hybrid search*, merged by Reciprocal
Rank Fusion. Retrieved chunks are rendered as a numbered context block so
the model can cite sources as [1], [2], ...
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from agentic_os.llm.tokens import estimate_tokens
from agentic_os.memory.chunking import chunk_text
from agentic_os.memory.embeddings import Embedder
from agentic_os.memory.vector_store import VectorStore
from agentic_os.rag.bm25 import BM25Index


@dataclass
class Document:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]

    @property
    def source(self) -> str:
        return str(self.metadata.get("source") or self.metadata.get("title") or self.metadata.get("doc_id", "unknown"))


class RAGPipeline:
    """Ingest documents, retrieve relevant chunks, and build grounded context."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        chunk_size: int = 200,
        overlap: int = 40,
        rrf_k: int = 60,
    ) -> None:
        self.vectors = VectorStore(embedder=embedder)
        self.keywords = BM25Index()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.rrf_k = rrf_k

    def __len__(self) -> int:
        return len(self.vectors)

    # -------------------------------------------------------------- ingest
    def add_document(self, doc: Document | str, **metadata: Any) -> list[str]:
        if isinstance(doc, str):
            doc = Document(doc, metadata)
        ids = []
        for i, chunk in enumerate(chunk_text(doc.text, self.chunk_size, self.overlap)):
            cid = f"{doc.id}:{i}"
            meta = {**doc.metadata, "doc_id": doc.id, "chunk": i}
            self.vectors.add(chunk, meta, id=cid)
            self.keywords.add(cid, chunk)
            ids.append(cid)
        return ids

    def add_documents(self, docs: list[Document | str]) -> int:
        return sum(len(self.add_document(d)) for d in docs)

    def add_file(self, path: str) -> list[str]:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        return self.add_document(Document(text, {"source": os.path.basename(path)}))

    # ------------------------------------------------------------ retrieve
    def retrieve(self, query: str, k: int = 4, mode: str = "hybrid") -> list[RetrievedChunk]:
        """Return the top ``k`` chunks. ``mode``: ``hybrid``, ``vector`` or ``keyword``."""
        pool = max(k * 4, 10)
        rankings: list[list[str]] = []
        if mode in ("hybrid", "vector"):
            rankings.append([h.record.id for h in self.vectors.search(query, k=pool, min_score=0.05)])
        if mode in ("hybrid", "keyword"):
            rankings.append([doc_id for doc_id, _ in self.keywords.search(query, k=pool)])
        if mode not in ("hybrid", "vector", "keyword"):
            raise ValueError(f"Unknown retrieval mode: {mode!r}")

        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out = []
        for cid, score in top:
            rec = self.vectors.get(cid)
            if rec is not None:
                out.append(RetrievedChunk(cid, rec.text, score, rec.metadata))
        return out

    def build_context(self, query: str, k: int = 4, max_tokens: int = 3000) -> tuple[str, list[RetrievedChunk]]:
        """Render retrieved chunks as a numbered, citable context block."""
        chunks = self.retrieve(query, k=k)
        lines, used, budget = [], [], max_tokens
        for chunk in chunks:
            entry = f"[{len(used) + 1}] (source: {chunk.source})\n{chunk.text}"
            cost = estimate_tokens(entry)
            if cost > budget:
                break
            budget -= cost
            lines.append(entry)
            used.append(chunk)
        return "\n\n".join(lines), used

    def as_tool(self, k: int = 4):
        """A ``search_knowledge`` tool so agents can query the knowledge base on demand."""
        from agentic_os.tools.registry import make_tool

        def search_knowledge(query: str) -> str:
            """Search the knowledge base for passages relevant to a query. Cite results by their [n] number.

            Args:
                query: A focused search query.
            """
            context, _ = self.build_context(query, k=k)
            return context or "No relevant documents found."

        return make_tool(search_knowledge, tags=["knowledge", "core", "readonly"])
