"""Loading and validation for corpus, queries, qrels and embeddings.

All inputs are JSONL. Validation is strict on purpose: a harness that silently
drops rows produces numbers nobody can reproduce.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from typing import Any, Iterable


class DataError(ValueError):
    """Raised when an input file is malformed or internally inconsistent."""


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


def _rows(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise DataError(f"{path}:{lineno}: expected a JSON object")
            yield lineno, row


def _require_str(row: dict[str, Any], field: str, where: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DataError(f"{where}: missing non-empty string field {field!r}")
    return value.strip()


def load_corpus(path: str | Path) -> dict[str, Document]:
    docs: dict[str, Document] = {}
    for lineno, row in _rows(path):
        where = f"{path}:{lineno}"
        doc = Document(_require_str(row, "id", where), _require_str(row, "text", where))
        if doc.doc_id in docs:
            raise DataError(f"{where}: duplicate document id {doc.doc_id!r}")
        docs[doc.doc_id] = doc
    if not docs:
        raise DataError(f"{path}: corpus is empty")
    return docs


def load_queries(path: str | Path) -> dict[str, Query]:
    queries: dict[str, Query] = {}
    for lineno, row in _rows(path):
        where = f"{path}:{lineno}"
        query = Query(_require_str(row, "id", where), _require_str(row, "text", where))
        if query.query_id in queries:
            raise DataError(f"{where}: duplicate query id {query.query_id!r}")
        queries[query.query_id] = query
    if not queries:
        raise DataError(f"{path}: queries are empty")
    return queries


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for lineno, row in _rows(path):
        where = f"{path}:{lineno}"
        query_id = _require_str(row, "query_id", where)
        doc_id = _require_str(row, "doc_id", where)
        grade = row.get("grade", 1)
        if not isinstance(grade, int) or isinstance(grade, bool) or grade < 0:
            raise DataError(f"{where}: grade must be a non-negative integer")
        qrels.setdefault(query_id, {})[doc_id] = grade
    if not qrels:
        raise DataError(f"{path}: qrels are empty")
    return qrels


def load_embeddings(path: str | Path) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {}
    dim: int | None = None
    for lineno, row in _rows(path):
        where = f"{path}:{lineno}"
        doc_id = _require_str(row, "id", where)
        vector = row.get("vector")
        if not isinstance(vector, list) or not vector:
            raise DataError(f"{where}: vector must be a non-empty list of numbers")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vector):
            raise DataError(f"{where}: vector contains non-numeric values")
        values = [float(v) for v in vector]
        if not all(math.isfinite(v) for v in values):
            raise DataError(f"{where}: vector contains non-finite values")
        if dim is None:
            dim = len(values)
        elif len(values) != dim:
            raise DataError(f"{where}: dimension {len(values)} != first vector's {dim}")
        if doc_id in vectors:
            raise DataError(f"{where}: duplicate embedding id {doc_id!r}")
        vectors[doc_id] = values
    if not vectors:
        raise DataError(f"{path}: embeddings are empty")
    return vectors


def validate(
    corpus: dict[str, Document],
    queries: dict[str, Query],
    qrels: dict[str, dict[str, int]],
) -> list[str]:
    """Check cross-file consistency. Returns human-readable warnings."""
    for query_id, grades in qrels.items():
        if query_id not in queries:
            raise DataError(f"qrels reference unknown query {query_id!r}")
        for doc_id, grade in grades.items():
            if doc_id not in corpus:
                raise DataError(f"qrels reference unknown document {doc_id!r}")
            if grade == 0:
                del grades[doc_id]
    unjudged = sum(1 for qid in queries if qid not in qrels or not qrels[qid])
    warnings = [f"{unjudged} of {len(queries)} queries have no relevant judgements"]
    return warnings


def require_coverage(corpus: dict[str, Document], vectors: dict[str, list[float]]) -> None:
    missing = sorted(set(corpus) - set(vectors))
    if missing:
        head = ", ".join(missing[:5])
        raise DataError(f"{len(missing)} documents lack embeddings (e.g. {head})")


def split_digest(
    corpus: dict[str, Document],
    queries: dict[str, Query],
    qrels: dict[str, dict[str, int]],
) -> str:
    """Stable fingerprint of the evaluated data, independent of row order."""
    payload = {
        "corpus": sorted([d.doc_id, d.text] for d in corpus.values()),
        "queries": sorted([q.query_id, q.text] for q in queries.values()),
        "qrels": sorted([qid, did, g] for qid, g in qrels.items() for did, g in g.items()),
    }
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()
