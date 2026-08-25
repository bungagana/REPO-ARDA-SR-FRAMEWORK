# Cross-domain datasets

Each dataset used in the cross-domain generalization experiments (Table 9 of the
manuscript) is publicly available and downloadable from its official source. Below is
the per-dataset link, the corresponding build script, and a short schema note.

The build scripts live in the framework at
[`supplementary/cross_domain/<dataset>/01_build_<dataset>_kb.py`](../supplementary/cross_domain/).

---

## Datasets

| Dataset | Domain | Size (used) | Official source | Citation (in the manuscript) | Build script |
|---|---|---|---|---|---|
| **CUAD** (Contract Understanding Atticus Dataset) | Legal contracts (SQuAD-style clause QA) | 180 queries | [`theatticusproject/cuad-qa` on Hugging Face](https://huggingface.co/datasets/theatticusproject/cuad-qa) | Hendrycks, Burns, Chen & Ball (2021) | `supplementary/cross_domain/cuad/01_build_cuad_kb.py` |
| **ConditionalQA** | Government / public policy (gov.uk) | 180 queries | [`haitian-sun/ConditionalQA` on GitHub](https://github.com/haitian-sun/ConditionalQA) | Sun, Cohen & (co-authors) | `supplementary/cross_domain/conditionalqa/01_build_conditionalqa_kb.py` |
| **FinanceBench** | Financial audits (SEC filings) | 180 queries | [`PatronusAI/financebench` on Hugging Face](https://huggingface.co/datasets/PatronusAI/financebench) | Islam, Kannappan, Kiela, Qian, Scherner & (co-authors) | `supplementary/cross_domain/financebench/01_build_financebench_kb.py` |
| **PubMedQA** (PQA-L) | Biomedical / health | 200 queries | [`qiaojin/PubMedQA` on Hugging Face](https://huggingface.co/datasets/qiaojin/PubMedQA) | Jin et al. (2019) | `supplementary/cross_domain/pubmedqa/01_build_pubmedqa_kb.py` |
| **ID-GovQA** (pakdwi) | Indonesian government policy | 111 queries | Collected from **public Indonesian open-data portals**; bundled in [`data/id_govqa_pakdwi_test_sample.json`](../data/id_govqa_pakdwi_test_sample.json) | Authors' own (described in the manuscript) | `supplementary/cross_domain/pakdwi/01_build_pakdwi_kb.py` |

> **Citation note.** CUAD, ConditionalQA, FinanceBench and PubMedQA are **public
> benchmark datasets** and are cited in the manuscript. **ID-GovQA** is a dataset of
> real-world Indonesian government queries assembled by the authors from **public
> open-data portals**; it is not a third-party benchmark but is shared with this
> repository (`data/id_govqa_pakdwi_test_sample.json`, 111 queries, 12 unanswerable) so
> the cross-domain result can be reproduced.

### Download links (direct)

| Dataset | Direct link |
|---|---|
| CUAD | `https://huggingface.co/datasets/theatticusproject/cuad-qa` (split `train`, revision `refs/convert/parquet`) |
| ConditionalQA | `https://github.com/haitian-sun/ConditionalQA` (JSON files under `v1_0`) |
| FinanceBench | `https://huggingface.co/datasets/PatronusAI/financebench` (split `train`) |
| PubMedQA | `https://huggingface.co/datasets/qiaojin/PubMedQA` (`pqa_labeled`, split `train`) |

---

## How to reproduce the cross-domain comparison (Table 9)

Each dataset folder contains the two-step reproduction pipeline. Build the KB, then run
the comparison on the same method set as the paper (Standard RAG, Self-RAG, ARDA-SR):

```bash
# e.g. PubMedQA
cd supplementary/cross_domain/pubmedqa
python 01_build_pubmedqa_kb.py                 # download + build KB (one-time)
python 02_run_pubmedqa_test.py                # run Standard RAG / Self-RAG / ARDA-SR
```

The same holds for `cuad/`, `conditionalqa/`, and `financebench/`. Each run writes a
per-method results file under `<dataset>/results/`, and `summary.json` /
`summary_refusal_fixed.json` aggregate the metrics (Rel / Faith / Cov / FRR / latency)
used in Table 9.

### Quick look per dataset

| Dataset | Context window | Queries | Answerable | Unanswerable slice |
|---|---|---|---|---|
| CUAD | union clauses | 180 | 87 | 93/180 |
| ConditionalQA | policy paragraphs | 180 | 168 | 12/180 |
| FinanceBench | SEC filings | 180 | 180 | 0 |
| PubMedQA | abstracts | 180 | 180 | 0 |
| ID-GovQA (pakdwi) | gov. policy | 111 | 99 | 12/111 |

> The models used to generate the Table 9 numbers are recorded in the manuscript; the
> evaluation protocol (judge, metrics, refusal handling) is the same as the main-domain
> experiment (see `evaluation/`).

---

## Reproducibility of the reported numbers

The aggregate metrics in Table 9 can be recomputed from the released per-method result
files in this repository:

```
results/data/summary_<domain>.json      → Rel / Faith / Cov / latency
results/data/summary_<domain>_raw.json  → FRR / retrieval / tool / SRCompl
```

and the cross-backbone numbers in Table 8 from:

```
results/data/arda_sr_*_pakdwi_summary.json
```

Run `results/scripts/reproduce_tables.py` to regenerate `results/tables/table6.csv`,
`table8.csv`, and `table9.csv` from these files.