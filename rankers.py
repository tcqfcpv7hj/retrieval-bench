"""Deterministic, dependency-free rankers: lexical BM25 and dense cosine."""
from __future__ import annotations

import math
import re

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 over an in-memory corpus with precomputed statistics."""

    def __init__(self, corpus: dict[str, str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._freqs: dict[str, dict[str, int]] = {}
        doc_freq: dict[str, int] = {}
        lengths: dict[str, int] = {}
        for doc_id, text in corpus.items():
            tokens = tokenize(text)
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self._freqs[doc_id] = counts
            lengths[doc_id] = len(tokens)
            for token in counts:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        total = len(corpus)
        self._avg_len = sum(lengths.values()) / total if total else 0.0
        self._idf = {
            term: math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }
        self._lengths = lengths

    def scores(self, query: str) -> dict[str, float]:
        terms = tokenize(query)
        ranked: dict[str, float] = {}
        for doc_id, counts in self._freqs.items():
            score = 0.0
            length_norm = 1.0 - self.b + self.b * self._lengths[doc_id] / (self._avg_len or 1.0)
            for term in terms:
                tf = counts.get(term)
                if not tf:
                    continue
                score += self._idf[term] * (tf * (self.k1 + 1.0)) / (tf + self.k1 * length_norm)
            ranked[doc_id] = score
        return ranked


class Dense:
    """Cosine similarity over precomputed L2-normalised embeddings."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._unit: dict[str, tuple[float, ...]] = {}
        for doc_id, vector in vectors.items():
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            self._unit[doc_id] = tuple(v / norm for v in vector)

    def scores(self, query_vector: list[float]) -> dict[str, float]:
        norm = math.sqrt(sum(v * v for v in query_vector)) or 1.0
        query = tuple(v / norm for v in query_vector)
        return {
            doc_id: sum(q * d for q, d in zip(query, vector))
            for doc_id, vector in self._unit.items()
        }


def top_k(scores: dict[str, float], k: int) -> list[tuple[str, float]]:
    """Rank by score descending, breaking ties on doc id for stability."""
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
