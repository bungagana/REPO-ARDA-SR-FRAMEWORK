# Real-World Deployment — BPJS/INA-CBG Claim Screening

ARDA-SR has been deployed in a real-world setting to screen inpatient claims for
BPJS Kesehatan / INA-CBG automatically. The system operates on the Senopati AI
platform ([https://senopati.its.ac.id/klaim-bpjs/](https://senopati.its.ac.id/klaim-bpjs/)).

## Data

A single **password-protected archive** holds the de-identified evaluation
dataset:

```
data/deployment_data_public.zip
```

This archive is **encrypted** with a ZIP password: you may download it straight from
this repository, yet it **cannot be opened without the password**. That password is
**not** kept in this repository; e-mail the corresponding author (address given in the
manuscript) to request it together with the usage terms. Approval is given on a
per-request basis, for research/reproducibility purposes.

### Contents of the archive

> The paths listed below are relative to the archive root; the zip keeps its
> evaluation-related JSON/CSV files (`judge_scores_*`, `judge_agreement`,
> `final_3judge_metrics`, `juri_sweep_rule`, `selfrag_full`) inside a `results/`
> subfolder.

| File | Description |
|---|---|
| `predict_master.xlsx` | **437 rule-level rows** (PASS/FAIL criteria per rule + model/judge verdicts). Note: the figure 437 counts **criteria/rules**, generated from the **49** eligible episodes. The source log folder contains **404** claim files; the 60-episode sample produces 49 eligible episodes, while the remaining **11** are dropped since the production system created **no verification rules** for them (their `rule_compliance` is empty or holds only a single `NEEDS_REVIEW` placeholder, meaning the claimed ICD code has no guideline criteria in the rule database). |
| `suggestion_master.xlsx` | **60 claim-level rows** (one row per episode — verdict and correction suggestions for each claim, per system/judge). |
| `predict_base.json` / `suggestion_base.json` | Rows in their base form, ahead of the judge columns. |
| `sample60_filelist.json` | The 60-episode sample, selected deterministically. |
| `full_report_401.json` | Re-run results for flat qwen (Standard RAG) (401 runs inside the production report). |
| `arda_sr_haji_pilot60_v2.json` / `_detailed.json` | Output from the ARDA-SR pipeline. |
| `claude_judgments.json` | The Claude judge column. |
| `gemini_predict_scores.json` | **Rule-level** predictions from Gemini (a candidate judge that was dropped from the final panel). |
| `judge_scores.json` | **DeepSeek** as Judge 1 — Rel/Faith/Cov scores (1–5) for each claim. |
| `judge_scores_claude.json` | **Claude** as Judge 2 — Rel/Faith/Cov scores. |
| `judge_scores_qwen38max.json` | **Qwen3.8Max** as Judge 3 — Rel/Faith/Cov scores. |
| `judge_agreement.json` | Agreement between judges + Fleiss' κ. |
| `final_3judge_metrics.json` | Rule- and suggestion-level metrics for the final 3-judge panel. |
| `juri_sweep_rule.csv` | Robustness sweep over adjudicator subsets. |
| `selfrag_full.json` | Verdicts + suggestions for the Self-RAG baseline. |

> **Note on the judge set.** The final adjudication panel is made up of **three**
> judges: DeepSeek, Claude, and Qwen3.8Max. **Two** candidate judges are **excluded**,
> and by design the archive does **not** include their per-claim score files:
> **Gemini** — leniency bias (62.7% PASS on rules; 66.7% of its suggestions read
> "already meets criteria") — and **GPT** — degenerate refusal pattern (94% of rules
> labelled FAIL). `gemini_predict_scores.json` is kept only to document the Gemini
> rule-level predictions; its suggestion-quality scores are left out.

### No personal data

No patient names, IDs, national identity numbers (NIK), phone numbers, addresses,
e-mail addresses, or medical-record numbers appear in the released material; only
coarse age and sex are kept. On-premise data-residency requirements under Indonesia's
Personal Data Protection Act (UU No. 27/2022) apply to this dataset.

## Main result

On both the rule level and the suggestion level, Table 1 pits ARDA-SR against the
most relevant baselines (Standard RAG = flat single-pass qwen; Self-RAG), using a
three-judge adjudication panel.

| Method | Acc.↑ | Prec.↑ | Rec.↑ | FRR↓ | ARR↑ | R | F | C |
|---|---|---|---|---|---|---|---|---|
| Standard RAG (baseline) | 0.795 | 0.78 | 0.65 | 0.115 | 0.646 | 3.12 | 3.53 | 2.63 |
| Self-RAG * | 0.778 | 0.76 | 0.63 | 0.180 | 0.708 | 2.13 | 2.86 | 1.82 |
| ARDA-SR (Ours) | **0.839** | 0.82 | 0.68 | **0.083** | **0.708** | **3.25** | **3.71** | **2.89** |

Acc./Prec./Rec./FRR/ARR stand for rule-level accuracy/precision/recall and refusal
rates; R/F/C stand for suggestion Relevance / Faithfulness / Coverage on a 1–5 scale.
Every rule-level metric is computed under one three-judge panel (Qwen3.8Max,
DeepSeek-V4, Claude; ties → FAIL), and across the **same n=347 criteria for all three
methods** so the comparison stays fair (apple-to-apple). For Standard RAG and ARDA-SR,
the suggestion metrics cover **n=180** ratings; for Self-RAG they cover **n=54**
episodes.

\* Out of the **49** episodes carrying criteria, Self-RAG parsed **44** and **failed on
5** (90 criteria), since the model returned unparseable JSON for long/compound rules. So
that the comparison stays fair, Table 11 computes rule-level metrics for **all three**
methods over the same n=347 criteria that Self-RAG could adjudicate (instead of n=437
for the other two).

Δ relative to the Standard-RAG baseline: accuracy 0.795→0.839 (+0.044); FRR 0.115→0.083 (−0.032);
ARR 0.646→0.708 (+0.062); suggestion coverage 2.63→2.89 (+0.26).

## Figure

The deployment figure cited in the manuscript sits in `figures/` (aggregate results
only; no patient-level detail).

## Reproducibility

The released tables let you reproduce the aggregate metrics and the figure; the
evaluation and plotting scripts are in the root `evaluation/` and `pipeline/` folders.