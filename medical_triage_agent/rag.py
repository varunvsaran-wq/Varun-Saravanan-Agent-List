"""
Local RAG retrieval module.

Indexes a medical knowledge base stored as plain-text or markdown files in
`medical_kb/` and retrieves the top-k most relevant chunks using embeddings
+ cosine similarity.

Supports two backends (auto-selected):
  1. sentence-transformers  (preferred — local model, no API key needed)
  2. numpy-only TF-IDF fallback (zero extra deps beyond numpy)

Public interface:
    retriever = MedicalKBRetriever(kb_dir="medical_kb/")
    results  = retriever.retrieve(query, k=8)
    # -> list[RetrievedChunk]
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class KBChunk:
    """A single chunk from the knowledge base."""

    chunk_id: str
    title: str
    text: str
    source_file: str


@dataclass
class RetrievedChunk:
    """A chunk plus its similarity score."""

    id: str
    title: str
    text: str
    score: float


# ── Embedding backends ────────────────────────────────────────────────────────


class EmbeddingBackend(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerBackend:
    """Uses `sentence-transformers` for dense retrieval."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class TFIDFFallbackBackend:
    """Minimal TF-IDF vectoriser using only numpy — no extra packages."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self._fitted = False

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def fit(self, corpus: list[str]) -> None:
        doc_freq: dict[str, int] = {}
        for doc in corpus:
            seen: set[str] = set()
            for tok in self._tokenize(doc):
                if tok not in seen:
                    doc_freq[tok] = doc_freq.get(tok, 0) + 1
                    seen.add(tok)
        self.vocab = {tok: i for i, tok in enumerate(sorted(doc_freq))}
        n = len(corpus)
        self.idf = np.zeros(len(self.vocab))
        for tok, idx in self.vocab.items():
            self.idf[idx] = np.log((n + 1) / (doc_freq.get(tok, 0) + 1)) + 1
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        vecs = np.zeros((len(texts), len(self.vocab)))
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            for tok in tokens:
                if tok in self.vocab:
                    vecs[i, self.vocab[tok]] += 1
            # TF-IDF weighting
            if self.idf is not None:
                vecs[i] *= self.idf
        # L2 normalise
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs /= norms
        return vecs


# ── Chunker ───────────────────────────────────────────────────────────────────


def _chunk_text(
    text: str, source_file: str, max_chunk_chars: int = 1500, overlap: int = 200
) -> list[KBChunk]:
    """Split a document into overlapping chunks with stable IDs."""
    # Try to split on markdown headings first
    sections = re.split(r"\n(?=##?\s)", text)
    chunks: list[KBChunk] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        # Extract title from first heading line, if any
        heading_match = re.match(r"^##?\s+(.+)", section)
        title = heading_match.group(1).strip() if heading_match else source_file

        if len(section) <= max_chunk_chars:
            cid = hashlib.md5(section[:200].encode()).hexdigest()[:10]
            chunks.append(
                KBChunk(
                    chunk_id=f"{Path(source_file).stem}_{cid}",
                    title=title,
                    text=section,
                    source_file=source_file,
                )
            )
        else:
            # Sliding window
            start = 0
            part = 0
            while start < len(section):
                end = start + max_chunk_chars
                snippet = section[start:end]
                cid = hashlib.md5(snippet[:200].encode()).hexdigest()[:10]
                chunks.append(
                    KBChunk(
                        chunk_id=f"{Path(source_file).stem}_{cid}_p{part}",
                        title=f"{title} (part {part + 1})",
                        text=snippet,
                        source_file=source_file,
                    )
                )
                start += max_chunk_chars - overlap
                part += 1

    return chunks


# ── Main retriever class ─────────────────────────────────────────────────────


class MedicalKBRetriever:
    """Indexes and retrieves from a local medical knowledge-base directory."""

    def __init__(
        self,
        kb_dir: str = "medical_kb/",
        use_sentence_transformers: bool | None = None,
    ):
        self.kb_dir = Path(kb_dir)
        self.chunks: list[KBChunk] = []
        self.embeddings: np.ndarray | None = None
        self.backend: EmbeddingBackend

        # Auto-select backend
        if use_sentence_transformers is None:
            try:
                self.backend = SentenceTransformerBackend()
            except ImportError:
                self.backend = TFIDFFallbackBackend()
        elif use_sentence_transformers:
            self.backend = SentenceTransformerBackend()
        else:
            self.backend = TFIDFFallbackBackend()

        self._index()

    def _index(self) -> None:
        """Load all .txt / .md files from kb_dir and build the index."""
        if not self.kb_dir.exists():
            print(f"[RAG] Warning: KB directory '{self.kb_dir}' not found. Using empty index.")
            return

        for fpath in sorted(self.kb_dir.iterdir()):
            if fpath.suffix.lower() in (".txt", ".md"):
                text = fpath.read_text(encoding="utf-8", errors="replace")
                self.chunks.extend(_chunk_text(text, fpath.name))

        if not self.chunks:
            print("[RAG] Warning: No KB chunks found. Retrieval will return empty results.")
            return

        texts = [c.text for c in self.chunks]
        # For TF-IDF backend, fit on the full corpus first
        if isinstance(self.backend, TFIDFFallbackBackend):
            self.backend.fit(texts)
        self.embeddings = self.backend.encode(texts)
        print(f"[RAG] Indexed {len(self.chunks)} chunks from {self.kb_dir}")

    def retrieve(self, query: str, k: int = 8) -> list[RetrievedChunk]:
        """Return the top-k most relevant chunks for the query."""
        if not self.chunks or self.embeddings is None:
            return []

        q_vec = self.backend.encode([query])  # (1, dim)
        scores = (q_vec @ self.embeddings.T).flatten()  # cosine similarity
        top_idx = np.argsort(scores)[::-1][:k]

        results: list[RetrievedChunk] = []
        for idx in top_idx:
            results.append(
                RetrievedChunk(
                    id=self.chunks[idx].chunk_id,
                    title=self.chunks[idx].title,
                    text=self.chunks[idx].text,
                    score=float(scores[idx]),
                )
            )
        return results
