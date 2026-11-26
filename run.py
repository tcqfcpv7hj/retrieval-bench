"""Run a ranker over a frozen split and write a reproducible run manifest.

Usage:
    python run.py --corpus data/corpus.jsonl --queries data/queries.jsonl \
        --qrels data/qrels.jsonl --ranker bm25 --out runs/bm25

The output directory receives run.jsonl (ranked ids) and manifest.json (config,
data digest, seed, versions). An existing directory is only reused with --force.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import data
import metrics
import rankers

VERSION = "0.2.0"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a retrieval ranker reproducibly.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--ranker", choices=["bm25", "dense"], default="bm25")
    parser.add_argument("--embeddings", help="document embeddings (dense ranker)")
    parser.add_argument("--query-embeddings", help="query embeddings keyed by query id (dense ranker)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be positive")
    if args.ranker == "dense" and not (args.embeddings and args.query_embeddings):
        parser.error("--ranker dense needs --embeddings and --query-embeddings")
    return args


def build_ranker(args: argparse.Namespace, corpus: dict[str, data.Document]):
    if args.ranker == "bm25":
        return rankers.BM25({d.doc_id: d.text for d in corpus.values()})
    vectors = data.load_embeddings(args.embeddings)
    data.require_coverage(corpus, vectors)
    return rankers.Dense(vectors)


def load_query_vectors(path: str, queries: dict[str, data.Query]) -> dict[str, list[float]]:
    vectors = data.load_embeddings(path)
    missing = sorted(set(queries) - set(vectors))
    if missing:
        raise data.DataError(f"{len(missing)} queries lack embeddings (e.g. {missing[0]})")
    return vectors


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)

    try:
        return execute(args)
    except data.DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def execute(args: argparse.Namespace) -> int:
    corpus = data.load_corpus(args.corpus)
    queries = data.load_queries(args.queries)
    qrels = data.load_qrels(args.qrels)
    for warning in data.validate(corpus, queries, qrels):
        print(f"warning: {warning}", file=sys.stderr)

    digest = data.split_digest(corpus, queries, qrels)
    out_dir = Path(args.out)
    if out_dir.exists():
        if not out_dir.is_dir():
            print(f"error: {out_dir} exists and is not a directory", file=sys.stderr)
            return 1
        if any(out_dir.iterdir()) and not args.force:
            print(f"error: {out_dir} is not empty (use --force)", file=sys.stderr)
            return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    ranker = build_ranker(args, corpus)
    query_vectors = (
        load_query_vectors(args.query_embeddings, queries) if args.ranker == "dense" else None
    )

    runs: dict[str, list[str]] = {}
    with (out_dir / "run.jsonl").open("w", encoding="utf-8") as handle:
        for query_id in sorted(queries):
            scores = (
                ranker.scores(query_vectors[query_id])
                if query_vectors is not None
                else ranker.scores(queries[query_id].text)
            )
            ranked = rankers.top_k(scores, args.k)
            runs[query_id] = [doc_id for doc_id, _ in ranked]
            for rank, (doc_id, score) in enumerate(ranked, 1):
                row = {"query_id": query_id, "doc_id": doc_id, "rank": rank, "score": round(score, 8)}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_ks = tuple(sorted({5, args.k}))
    results = metrics.evaluate(runs, qrels, metrics_ks)
    scores_used = {k: v for k, v in results.items() if k != "judged_queries"}

    manifest = {
        "tool": {"name": "retrieval-bench", "version": VERSION},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "k": args.k,
        "ranker": args.ranker,
        "ranker_config": {"k1": 1.2, "b": 0.75} if args.ranker == "bm25" else {"similarity": "cosine"},
        "data": {
            "digest": digest,
            "documents": len(corpus),
            "queries": len(queries),
            "judged_queries": len(qrels),
        },
        "python": platform.python_version(),
        "metrics": scores_used,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps({"ok": True, "out": str(out_dir), "digest": digest, "metrics": scores_used}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
