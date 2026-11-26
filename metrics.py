"""Ranking metrics computed from graded judgements.

Grades are non-negative integers; grade 0 rows are dropped during validation.
Recall and MRR treat any positive grade as relevant, nDCG uses the raw grades.
"""
from __future__ import annotations

import math

K = 10


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int = K) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hits / len(relevant)


def mrr_at_k(ranked: list[str], relevant: dict[str, int], k: int = K) -> float:
    for position, doc_id in enumerate(ranked[:k], 1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(position + 1) for position, grade in enumerate(grades, 1))


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int = K) -> float:
    if not relevant:
        return 0.0
    observed = [relevant.get(doc_id, 0) for doc_id in ranked[:k]]
    ideal = sorted(relevant.values(), reverse=True)[:k]
    best = _dcg(ideal)
    return _dcg(observed) / best if best else 0.0


METRICS = {"recall": recall_at_k, "mrr": mrr_at_k, "ndcg": ndcg_at_k}


def evaluate(
    runs: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Average each metric over judged queries; queries with no qrels are skipped."""
    judged = [qid for qid in runs if qrels.get(qid)]
    if not judged:
        raise ValueError("no judged queries to evaluate")
    results: dict[str, float] = {}
    for name, fn in METRICS.items():
        for k in ks:
            results[f"{name}@{k}"] = sum(fn(runs[qid], qrels[qid], k) for qid in judged) / len(judged)
    results["judged_queries"] = float(len(judged))
    return results
