# Data schema — BPJS Claim Screening evaluation tables

This file documents, column by column, the released evaluation tables. **No actual
data** is included here; the tables themselves are available on request from the
authors (password-protected). See `README.md`.

---

## 1. `predict_master.xlsx` — rule-level corpus (437 rows)

One row per clinical criterion (per ICD entry of an episode), keyed by
`namafile` + `no`.

| Column | Description |
|---|---|
| `namafile` | Anonymised file identifier (opaque timestamp + code, e.g. `20260714_120833_448858_unknown`). No hospital name or patient identifier. |
| `no` | Sequential criterion index within the file. |
| `expected` | The guideline requirement text (prefixed with its ICD code, e.g. `[I21.4]`). |
| `reality` | (Display-only) the production system's interpretation of the evidence. |
| `prediksi_json` | Original production verdict: `PASS` / `FAIL`. |
| `re_predict_qwen` | Flat single-pass qwen re-run (the "Standard RAG" baseline). |
| `ARDA-SR` | ARDA-SR verdict (qwen generator + gemma arbiter). |
| `Claude` | Judge 1 verdict. |
| `DeepSeekV4` | Judge 2 verdict. |
| `GPT` | Candidate judge (excluded). |
| `Gemini` | Candidate judge (excluded). |
| `Qwen3.8Max` | Judge 3 verdict. |
| `Burhan`, `Alia`, `Giza` | Reserved / empty. |

**Adjudicated reference label** = majority of `Qwen3.8Max`, `DeepSeekV4`, `Claude`
(tie → FAIL). Used for the rule-level metrics in `final_3judge_metrics.json`.

## 2. `suggestion_master.xlsx` — claim-level corpus (60 rows)

One row per episode (claim), keyed by `namafile`; `no` is always 1.

| Column | Description |
|---|---|
| `namafile` | Anonymised file identifier (same as above). |
| `no` | Always 1. |
| `status` | Overall claim verdict: `REJECTED` / `ACCEPTED` / `NEEDS_REVIEW`. |
| `suggestion_json` | Production system's correction suggestion. |
| `re_suggestion_qwen` | Flat qwen suggestion (baseline). |
| `ARDA-SR` | ARDA-SR suggestion. |
| `Claude`, `DeepSeekV4`, `GPT`, `Gemini`, `Qwen3.8Max` | Per-judge suggestion. |
| `Burhan`, `Alia`, `Giza` | Reserved / empty. |

## 3. `judge_scores_*.json` — per-(claim, judge) quality scores

Each file is a JSON array of objects:
`{"case_id": "...", "rel": 1-5, "faith": 1-5, "cov": 1-5}`
where `case_id` = `<namafile>__<model>` and the fields are the judge's Relevance /
Faithfulness / Coverage ratings of that model's suggestion.

| File | Judge |
|---|---|
| `judge_scores.json` | DeepSeek |
| `judge_scores_claude.json` | Claude |
| `judge_scores_qwen38max.json` | Qwen3.8Max |

## 4. `judge_agreement.json`

Pairwise agreement (with the same `json` schema) plus `fleiss_kappa` (0.751) among the
three retained judges.

## 5. `final_3judge_metrics.json`

Final rule-level metrics (accuracy / precision / recall / specificity / F1 / FRR / ARR /
FAR) and suggestion-level means (Rel / Faith / Cov, 1–5) for flat qwen and ARDA-SR,
against the three-judge reference.

## 6. `juri_sweep_rule.csv`

Robustness sweep: for each non-empty subset of the three judges, the Δ agreement
(ARDA − flat). ARDA-SR leads in all 7 subsets (+0.005 … +0.028).

## 7. `selfrag_full.json`

Self-RAG baseline (generation → self-reflection → finalize) verdicts + per-claim
suggestion, for the comparison in `Table 1` of the deployment section.