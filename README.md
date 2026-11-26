# retrieval-bench

Small, dependency-free harness for running retrieval evaluations that other people
can reproduce. It loads a frozen corpus/query/judgement split, runs a ranker, and
writes both the ranked run and a manifest containing the seed, ranker settings and
a SHA-256 digest of the evaluated data.

The point is not to be a benchmark. It is to make the boring parts of an
evaluation hard to get wrong: silent row drops, mismatched splits, missing
judgements, and runs whose configuration nobody wrote down.

## Why

Retrieval numbers are only comparable when the split, the seed and the ranker
configuration are pinned. Most ad-hoc eval scripts pin none of them. This harness
refuses malformed inputs, records the data digest it actually used, and never
overwrites a previous run without `--force`.

## Install

Python 3.10+, standard library only.

```sh
pip install -e .
# or run the module directly
python run.py --help
```

## Usage

```sh
python run.py \
  --corpus data/corpus.jsonl \
  --queries data/queries.jsonl \
  --qrels data/qrels.jsonl \
  --ranker bm25 \
  --k 10 \
  --out runs/bm25-$(date +%s)
```

Dense evaluation takes precomputed embeddings (no model is loaded here):

```sh
python run.py ... --ranker dense \
  --embeddings data/embeddings.jsonl \
  --query-embeddings data/query_embeddings.jsonl \
  --out runs/dense
```

Inputs are JSONL:

| file | fields |
| --- | --- |
| corpus | `id`, `text` |
| queries | `id`, `text` |
| qrels | `query_id`, `doc_id`, `grade` (default 1) |
| embeddings | `id`, `vector` (one dimension for the whole file) |

Outputs:

- `run.jsonl`: `query_id`, `doc_id`, `rank`, `score`, sorted by query id
- `manifest.json`: data digest, seed, ranker and hyperparameters, counts, metrics

The run exits non-zero on duplicate ids, dangling qrels, dimension mismatches,
partial embedding coverage, or an occupied output directory.

## Metrics

`recall@k`, `mrr@k` and `ndcg@k` average over judged queries only; queries without
a positive judgement are counted in a warning and excluded from the average.
Grade 0 rows are discarded at validation. Ties are broken on document id so the
ranking is stable across filesystems.

## Results on the bundled fixtures

Six documents, four queries (three judged). These numbers exercise the plumbing;
they are not a benchmark claim.

| ranker | recall@10 | mrr@10 | ndcg@10 |
| --- | --- | --- | --- |
| bm25 | 1.000 | 1.000 | 0.963 |
| dense (cosine) | 1.000 | 1.000 | 1.000 |

## Layout

```
run.py              CLI: load, validate, rank, write run + manifest, evaluate
data.py             JSONL loaders, cross-file validation, split digest
rankers.py          BM25 and cosine rankers, stable top-k
metrics.py          recall, MRR, nDCG over graded judgements
data/               tiny fixture split and fixture embeddings
```

## Limitations

- The fixtures are synthetic and tiny; nothing here estimates real model quality.
- BM25 uses a simple lowercase tokenizer with no stemming or stopword handling.
- The dense ranker assumes embeddings are already computed and aligned with ids;
  it checks coverage but not the encoder used.
- Empty rankings score 0 rather than being skipped, so a broken ranker cannot
  hide behind an average.
- No significance testing yet; single-seed differences should not be read as wins.

## Prior work

The split/run/qrels layout follows the evaluation protocol popularised by
[BEIR](https://arxiv.org/abs/2104.08663) (Thakur et al., 2021). BM25 follows
Robertson and Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*
(2009). nDCG follows Järvelin and Kekäläinen (2002).

## License

MIT, see [LICENSE](LICENSE).
