# Per-query error analysis

`per_query.py` reads a finished `run.jsonl` and the matching `qrels.jsonl` and
prints one CSV row per query, worst nDCG first. It never re-ranks, so the table
always describes the exact run that was written to disk.

```sh
python run.py --corpus data/corpus.jsonl --queries data/queries.jsonl \
  --qrels data/qrels.jsonl --ranker bm25 --out runs/bm25
python analysis/per_query.py --run runs/bm25/run.jsonl --qrels data/qrels.jsonl --limit 5
```

Columns: `query_id`, `n_relevant`, `recall`, `mrr`, `ndcg`, `note`.

Interpretation:

- Rows are sorted by nDCG ascending, so the first rows are the failures worth
  reading by hand.
- `note=unjudged` marks queries with no positive judgement. Their metrics are
  0.000 by construction and should not be compared to judged rows.
- `n_relevant` next to a low score distinguishes a genuine ranking failure from
  an under-judged query.
- Judged queries that never appear in the run file are counted on stderr.

This is a triage aid, not a report: five hand-read failures usually explain more
than an aggregate delta, but they do not replace one.
