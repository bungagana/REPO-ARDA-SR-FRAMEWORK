# Real-World Deployment — BPJS/INA-CBG Claim Screening

A real-world deployment of ARDA-SR for automated BPJS Kesehatan / INA-CBG
inpatient-claim screening. The deployment runs on the Senopati AI platform
([https://senopati.its.ac.id/klaim-bpjs/](https://senopati.its.ac.id/klaim-bpjs/)).

## Data

The de-identified evaluation dataset is provided as a single **password-protected
archive**:

```
data/deployment_data_public.zip
```

The archive is **encrypted** (ZIP password) — it can be downloaded directly from this
repository, but **cannot be opened without the password**. The password is **not**
stored in this repository; to obtain it and the usage terms, email the corresponding
author (address in the manuscript). Access is granted on a per-request basis for
research/reproducibility purposes.

### Contents of the archive

> Paths below are relative to the archive root; the evaluation-related JSON/CSV files
> (`judge_scores_*`, `judge_agreement`, `final_3judge_metrics`, `juri_sweep_rule`,
> `selfrag_full`) are stored under a `results/` subfolder inside the zip.

| File | Description |
|---|---|
| `predict_master.xlsx` | **437 rule-level rows** (per-rule PASS/FAIL criteria + model/judge verdicts). Note: 437 is the number of **criteria/rules**, derived from the **49** eligible episodes. The source log folder holds **404** claim files; the 60-episode sample yields 49 eligible episodes — the other **11** are excluded because the production system produced **no verification rules** for them (their `rule_compliance` is empty or is a single `NEEDS_REVIEW` placeholder, i.e. the claimed ICD code has no guideline criteria in the rule database). |
| `suggestion_master.xlsx` | **60 claim-level rows** (one per episode — per-claim verdict and correction suggestions per system/judge). |
| `predict_base.json` / `suggestion_base.json` | Base rows before judge columns. |
| `sample60_filelist.json` | The deterministic 60-episode sample. |
| `full_report_401.json` | Flat qwen (Standard RAG) re-run results (401 runs in the production report). |
| `arda_sr_haji_pilot60_v2.json` / `_detailed.json` | ARDA-SR pipeline output. |
| `claude_judgments.json` | Claude judge column. |
| `gemini_predict_scores.json` | Gemini **rule-level** predictions (Gemini as a candidate judge, excluded from the final panel). |
| `judge_scores.json` | Judge 1 = **DeepSeek** — Rel/Faith/Cov scores (1–5) per claim. |
| `judge_scores_claude.json` | Judge 2 = **Claude** — Rel/Faith/Cov scores. |
| `judge_scores_qwen38max.json` | Judge 3 = **Qwen3.8Max** — Rel/Faith/Cov scores. |
| `judge_agreement.json` | Inter-judge agreement + Fleiss' κ. |
| `final_3judge_metrics.json` | Final rule- and suggestion-level metrics (3-judge panel). |
| `juri_sweep_rule.csv` | Adjudicator-subset robustness sweep. |
| `selfrag_full.json` | Self-RAG baseline verdicts + suggestions. |

> **Note on the judge set.** The final adjudication panel uses **three** judges: DeepSeek,
> Claude, and Qwen3.8Max. Two candidate judges are **excluded** and (by design) their
> per-claim score files are **not** included in the archive: **Gemini** — leniency bias
> (62.7% PASS on rules; 66.7% of its suggestions read "already meets criteria") — and
> **GPT** — degenerate refusal pattern (94% of rules labelled FAIL). `gemini_predict_scores.json`
> is retained only to document the Gemini rule-level predictions; its suggestion-quality
> scores are omitted.

### No personal data

The released material contains no patient names, IDs, national identity numbers (NIK),
phone numbers, addresses, e-mail addresses, or medical-record numbers; only coarse age
and sex are retained. The dataset is subject to on-premise data-residency requirements
under Indonesia's Personal Data Protection Act (UU No. 27/2022).

## Main result

Table 1 compares ARDA-SR against the most relevant baselines (Standard RAG = flat
single-pass qwen; Self-RAG) on both the rule level and the suggestion level, under a
three-judge adjudication panel.

| Method | Acc.↑ | Prec.↑ | Rec.↑ | FRR↓ | ARR↑ | R | F | C |
|---|---|---|---|---|---|---|---|---|
| Standard RAG (baseline) | 0.79 | 0.78 | 0.65 | 0.12 | 0.65 | 3.12 | 3.53 | 2.63 |
| Self-RAG * | 0.78 | 0.76 | 0.63 | 0.18 | 0.71 | 2.13 | 2.86 | 1.82 |
| ARDA-SR (Ours) | **0.81** | 0.82 | 0.68 | **0.10** | 0.68 | **3.25** | **3.71** | **2.89** |

Acc./Prec./Rec./FRR/ARR = rule-level accuracy/precision/recall and refusal rates;
R/F/C = suggestion Relevance / Faithfulness / Coverage on a 1–5 scale. All metrics are
computed under a single three-judge panel (Qwen3.8Max, DeepSeek-V4, Claude; ties →
FAIL). Rule-level metrics are over **n=437** criteria for Standard RAG and ARDA-SR, and
over **n=347** for Self-RAG; suggestion metrics are over **n=180** ratings for Standard
RAG and ARDA-SR and over **n=54** episodes for Self-RAG.

\* Self-RAG parsed **44 of 49** episodes carrying criteria and **failed on 5** (90
criteria) because the model returned an unparseable JSON for long/compound rules; it is
therefore computed on n=347 criteria, not the full n=437.

Δ over the Standard-RAG baseline: accuracy 0.79→0.81 (+0.02); FRR 0.12→0.10 (−0.02);
ARR 0.65→0.68 (+0.03); suggestion coverage 2.63→2.89 (+0.26).

## Figure

`figures/` contains the deployment figure referenced in the manuscript (aggregate
results only; no patient-level detail).

## Reproducibility

The aggregate metrics and figure are reproducible from the released tables; the
evaluation and plotting scripts live in the root `evaluation/` and `pipeline/` folders.