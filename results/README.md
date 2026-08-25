# Results & reproduction

This folder contains the raw experiment outputs, the comparison tables, the figure, and
the reproduction script for the paper's three **comparison tables** (Table 6, 8, 9) and
**Figure 6**.

## Layout

```
results/
├── data/            raw per-method per-domain results (JSON) — the source of the tables
├── tables/          the reproduced comparison tables (CSV)
│   ├── table6.csv   main-domain 1000-query comparison (Standard RAG … ARDA-SR)
│   ├── table8.csv   cross-backbone (Gemini / Qwen / Phi / Mixtral / Llama) summary
│   └── table9.csv   cross-domain (CUAD / ConditionalQA / FinanceBench / PubMedQA / ID-GovQA)
├── figures/         Figure 6 (Relevance vs computational cost trade-off)
└── scripts/         reproduce_tables.py — regenerates the tables from data/
```

## Reproducing the comparison tables

From the repo root:

```bash
cd results
python scripts/reproduce_tables.py
```

This reads `data/*` and writes:

- `tables/table6.csv` — answer quality (Rel/Faith/Cov) + FRR + latency per method on the
  1,000 main-domain queries.
- `tables/table8.csv` — cross-backbone summary (per backbone on the pakdwi dataset).
- `tables/table9.csv` — cross-domain per-method metrics (Rel/Faith/Cov/FRR/Lat).

Dependencies: `numpy` (see `requirements.txt`). The script performs **no LLM calls** —
it only aggregates the already-stored per-query/per-domain metrics, so it is offline and
deterministic.

## Figure 6

`data/Figure6_relevance_vs_cost.png` is the Relevance-vs-computational-cost trade-off
figure (extracted from the manuscript). You can regenerate the underlying scatter from
`table6.csv` (Rel on the y-axis, latency on the x-axis) — see
`results/scripts/reproduce_tables.py` for the metric fields it uses.

## Data provenance

| Folder | What it is | Source |
|---|---|---|
| `data/*_results.json` | 1,000 main-domain per-query runs, per method (rel/faith/cov/is_refusal/latency/hit@5/ctx_rel/mode) | `new/ARDA-SR-FRAMEWORK-FULL/results/` |
| `data/arda_sr_*_pakdwi_summary.json` | cross-backbone summary on ID-GovQA (pakdwi) | cross-domain experiments |
| `data/summary_<domain>.json` / `_raw.json` | cross-domain per-method metrics + FRR/retrieval | cross-domain experiments |

The cross-domain **datasets** (download links, schema, per-dataset reproduction steps)
are documented in [`cross-datasets/`](../cross-datasets).