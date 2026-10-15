"""Split documents into overlapping chunks for retrieval."""

from __future__ import annotations

import re


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Chunk ``text`` into windows of about ``chunk_size`` words.

    Paragraphs are kept together where possible; long paragraphs are split
    with ``overlap`` words carried into the next chunk so facts that cross a
    boundary stay retrievable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) > chunk_size:
            if current:
                chunks.append(" ".join(current))
            start, step = 0, chunk_size - overlap
            while start + chunk_size < len(words):
                chunks.append(" ".join(words[start : start + chunk_size]))
                start += step
            current = words[start:]
            continue
        if current and len(current) + len(words) > chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.extend(words)
    if current:
        chunks.append(" ".join(current))
    return chunks
