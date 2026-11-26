"""Turn a finished run into a per-query error table, worst queries first.

Reads run.jsonl + qrels.jsonl, never re-ranks. Queries without a positive
judgement are emitted with a note and excluded from nothing: they are reported
as unjudged so a split gap is visible next to the scores.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data  # noqa: E402
import metrics  # noqa: E402

COLUMNS = ("query_id", "n_relevant", "recall", "mrr", "ndcg", "note")


def load_run(path: str | Path) -> dict[str, list[str]]:
    rows: dict[str, list[tuple[int, str]]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                query_id, doc_id, rank = row["query_id"], row["doc_id"], int(row["rank"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"{path}:{lineno}: bad run row ({exc})") from exc
            rows.setdefault(query_id, []).append((rank, doc_id))
    return {qid: [doc for _, doc in sorted(items)] for qid, items in rows.items()}


def build_rows(runs: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int) -> list[dict]:
    table: list[dict] = []
    for query_id in sorted(runs):
        relevant = qrels.get(query_id, {})
        note = "" if relevant else "unjudged"
        table.append(
            {
                "query_id": query_id,
                "n_relevant": len(relevant),
                "recall": metrics.recall_at_k(runs[query_id], relevant, k),
                "mrr": metrics.mrr_at_k(runs[query_id], relevant, k),
                "ndcg": metrics.ndcg_at_k(runs[query_id], relevant, k),
                "note": note,
            }
        )
    return sorted(table, key=lambda row: (bool(row["note"]), row["ndcg"], row["query_id"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-query error report for a run file.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="only the worst N judged queries")
    parser.add_argument("--out", help="write CSV here instead of stdout")
    args = parser.parse_args(argv)

    runs = load_run(args.run)
    qrels = data.load_qrels(args.qrels)
    missing = sorted(set(qrels) - set(runs))
    if missing:
        print(f"warning: {len(missing)} judged queries absent from run", file=sys.stderr)

    rows = [row for row in build_rows(runs, qrels, args.k)]
    judged = [row for row in rows if row["note"] != "unjudged"]
    if args.limit:
        rows = judged[: args.limit] + [row for row in rows if row["note"] == "unjudged"]

    sink = open(args.out, "w", newline="", encoding="utf-8") if args.out else sys.stdout
    try:
        writer = csv.DictWriter(sink, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "recall": f"{row['recall']:.3f}", "mrr": f"{row['mrr']:.3f}",
                             "ndcg": f"{row['ndcg']:.3f}"})
    finally:
        if args.out:
            sink.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
