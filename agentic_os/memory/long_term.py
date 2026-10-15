"""Long-term memory: facts, episodes and lessons recalled by similarity."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from agentic_os.memory.embeddings import Embedder
from agentic_os.memory.vector_store import VectorStore


@dataclass
class Memory:
    id: str
    text: str
    kind: str
    score: float
    metadata: dict[str, Any]


class LongTermMemory:
    """Semantic memory backed by a :class:`VectorStore`.

    ``kind`` separates memory types: ``fact`` (semantic), ``episode`` (what
    happened in a past run) and ``lesson`` (self-reflection output).
    """

    def __init__(self, store: VectorStore | None = None, embedder: Embedder | None = None, path: str | None = None) -> None:
        self.store = store or VectorStore(embedder=embedder, path=path)

    def remember(self, text: str, kind: str = "fact", persist: bool = True, **metadata: Any) -> str:
        """Store a memory. Pass ``persist=False`` in bulk loops and call :meth:`save` once at the end."""
        meta = {"kind": kind, "created_at": time.time(), **metadata}
        rid = self.store.add(text, meta)
        if persist:
            self.save()
        return rid

    def save(self) -> None:
        if self.store.path:
            self.store.save()

    def recall(self, query: str, k: int = 3, kind: str | None = None, min_score: float = 0.15) -> list[Memory]:
        where = {"kind": kind} if kind else None
        return [
            Memory(h.record.id, h.record.text, h.record.metadata.get("kind", "fact"), h.score, h.record.metadata)
            for h in self.store.search(query, k=k, where=where, min_score=min_score)
        ]

    def forget(self, memory_id: str) -> bool:
        return self.store.delete(memory_id)

    def __len__(self) -> int:
        return len(self.store)

    def as_tools(self) -> list:
        """``remember`` / ``recall`` tools so agents can manage memory themselves."""
        from agentic_os.tools.registry import make_tool

        def remember_fact(fact: str) -> str:
            """Save an important fact to long-term memory for future conversations.

            Args:
                fact: A self-contained statement worth remembering.
            """
            self.remember(fact, kind="fact", source="agent")
            return "Saved to long-term memory."

        def recall_memories(query: str, k: int = 3) -> list:
            """Search long-term memory for facts, past episodes and lessons related to a query.

            Args:
                query: What to look for.
                k: Maximum number of memories to return.
            """
            return [{"text": m.text, "kind": m.kind, "score": round(m.score, 3)} for m in self.recall(query, k=k)]

        return [
            make_tool(remember_fact, tags=["memory"]),
            make_tool(recall_memories, tags=["memory", "readonly"]),
        ]
