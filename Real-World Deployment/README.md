# Real-World Deployment — BPJS Claim Screening (INA-CBG)

This folder documents a real-world deployment of ARDA-SR for automated
BPJS Kesehatan / INA-CBG inpatient-claim screening. It is the "run in production"
instance of the ARDA-SR framework described in the accompanying manuscript.

> **Data availability.** The evaluation data are **available on request from the
> authors**. They are not pushed to this repository. To obtain the de-identified
> dataset, email the corresponding author; you will receive a **password-protected
> archive** of the evaluation tables (per-rule and per-claim). The institution and the
> platform are intentionally **anonymised** in all released material.

---

## What this folder contains

| Path | Description |
|---|---|
| `README.md` | Overview, data-availability policy, and request instructions (this file) |
| `DATA_SCHEMA.md` | Column-by-column description of the released tables (no actual data) |
| `figures/` | The deployment figure(s) referenced in the manuscript |

## Why the data is not committed

The dataset contains patient-derived clinical content (structured medical records)
that, while fully **de-identified** (no names, IDs, NIK, phone, address, or medical-record
numbers; only coarse age and sex retained), is subject to on-premise data-residency
requirements under Indonesia's Personal Data Protection Act (UU No. 27/2022). The data
therefore remain under hospital control and are shared only on explicit request, under a
write-in usage agreement, rather than being published openly.

Figure(s) that only plot aggregate results (no patient-level detail) are committed and
appear under `figures/`.

## Dataset summary (what you will receive on request)

| Table | Content |
|---|---|
| `predict_master.xlsx` | 437 rule-level rows (per-rule PASS/FAIL criteria). Reviewer columns: the two systems under comparison (flat qwen / ARDA-SR) and the three independent LLM adjudicators. |
| `suggestion_master.xlsx` | 60 claim-level rows (per-claim verdict + correction suggestions per system/judge). |
| `judge_scores_*.json` | Per-(claim, judge) Relevance / Faithfulness / Coverage scores (1–5). |
| `judge_agreement.json` | Inter-judge pairwise agreement + Fleiss' kappa (κ = 0.751). |
| `final_3judge_metrics.json` | Final rule- and suggestion-level metrics reported in the paper. |
| `juri_sweep_rule.csv` | Adjudicator-subset robustness sweep (Δ agreement over all 7 non-empty judge subsets). |
| `selfrag_full.json` | Self-RAG baseline verdicts + suggestions (used as a comparison method). |

All figures in the manuscript are reproducible from the tables above; the plotting and
evaluation scripts are in the main `evaluation/` and `pipeline/` folders of this
repository.

## Requesting the data

Email the corresponding author (address in the manuscript / `README.md` of the root
repository) with:

- your name, affiliation, and research purpose;
- a statement that the data will be used for research/reproducibility only and will not
  be redistributed.

You will receive the **password-protected** archive of `predict_master.xlsx`,
`suggestion_master.xlsx`, and the supporting JSON/CSV tables, together with the password.
Institution and platform identifiers are **anonymised** (referred to as "Hospital A" /
"the platform") throughout the released material.