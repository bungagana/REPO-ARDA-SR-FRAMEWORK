<div align="center">

# ARDA-SR

**Adaptive Retrieval-Decision Architecture with Dual-Draft Arbitration and Scenario Reasoning**

A Retrieval-Augmented Generation framework for reliable, auditable government question-answering

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-TBD-lightgrey)
![Status](https://img.shields.io/badge/status-research%20code-orange)
![Reproducible](https://img.shields.io/badge/reproducible-tables%206%2C8%2C9%20%2B%20Fig%206-green)
![Datasets](https://img.shields.io/badge/datasets-CUAD%2C%20ConditionalQA%2C%20FinanceBench%2C%20PubMedQA-blueviolet)

</div>

---

ARDA-SR routes each query through an **entropy-based router**, arbitrates between a
**parametric** and a **retrieval-grounded** answer draft, and — for policy-scenario
questions — reasons explicitly over multiple decision alternatives. The goal is to reduce
false refusals and produce more reliable, auditable answers for public-sector QA systems.

This repository contains the full implementation, the benchmark dataset, and the human
annotation records used to validate it, so results can be independently reproduced and checked.

<div align="center">

![TransHub — an ARDA-SR-powered assistant for Indonesia's Ministry of Transmigration](image.png)

*TransHub: an ARDA-SR-powered assistant deployed for Indonesia's Ministry of Transmigration*

</div>

## Contents

| Path | Description |
|---|---|
| [`arda_sr/`](arda_sr) | The ARDA-SR method: Adaptive Query Router (AQR), Dual-Draft Arbitrator (DDA), Scenario Reasoning (SR), and hybrid retrieval |
| [`baselines/`](baselines) | 10 comparison baselines — Standard RAG, Hybrid RAG, HyDE-RAG, Adaptive-RAG, CRAG, ReAct, Self-RAG, FLARE, IRCoT, LLM-only |
| [`evaluation/`](evaluation) | Answer-quality judging (Relevance / Faithfulness / Coverage), routing & refusal metrics, statistical testing |
| [`pipeline/`](pipeline) | End-to-end scripts: build KB → generate benchmark → run experiment → ablation → analyze |
| [`data/`](data) | The QA benchmark — 1,000 main-domain QA pairs + 111 real-world Indonesian government QA pairs (ID-GovQA) |
| [`annotations/`](annotations) | Item-level validation of the benchmark by three independent annotators |
| [`supplementary/`](supplementary) | Cross-domain zero-shot transfer + an unanswerable-query diagnostic |
| [`results/`](results) | Reproducible comparison tables (Table 6, 8, 9), Figure 6, raw per-query experiment outputs, and the `reproduce_tables.py` script |
| [`cross-datasets/`](cross-datasets) | Public cross-domain datasets (CUAD / ConditionalQA / FinanceBench / PubMedQA) — download links, schema, and per-dataset reproduction steps |
| [`Real-World Deployment/`](Real-World%20Deployment) | A real-world deployment of ARDA-SR for automated BPJS/INA-CBG inpatient-claim screening in government question answering (Senopati AI platform: <https://senopati.its.ac.id/klaim-bpjs/>) — deployment notes, main-result table, figure, and a **password-protected** dataset archive (password on request from the authors) |
| `config.py` | All method parameters in one place |

**Not included:** the source document corpus, the model weights, and API keys.

## Quick start

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure API keys** — create a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
```

| Key | Used for |
|---|---|
| `GEMINI_API_KEY` | Generation, routing, and retrieval-grounded drafting (the core ARDA-SR model) |
| `ANTHROPIC_API_KEY` | QA benchmark generation only |
| `OPENAI_API_KEY` | Independent answer-quality judging — a separate model family from the generator, to avoid self-evaluation bias |

**3. Add your document corpus** — place source documents under `data/`, following the layout
expected by `utils/kb_builder.py`.

**4. Run the pipeline**

```bash
python pipeline/01_build_kb.py
python pipeline/02_generate_qa_claude.py
python pipeline/verify_answerability.py
python pipeline/03_run_experiment.py       # ARDA-SR + all 10 baselines
python pipeline/04_ablation.py             # component-wise ablation study
python pipeline/05_analyze_results.py      # aggregate into summary tables
```

Each script consumes the previous script's output.

> **Just want to re-run evaluation on the existing benchmark?** `data/qa_dataset.json` and
> `data/id_govqa_pakdwi_test_sample.json` already contain the QA pairs used in the paper —
> skip straight to step 4 against your own knowledge base.

## Datasets

The framework is evaluated on a main-domain dataset (1000 government QA pairs) plus
**four public cross-domain benchmarks** and a **real-world Indonesian government set**.
The cross-domain datasets are all directly downloadable from their official public
sources (repository [`cross-datasets/`](cross-datasets));
full schema + per-dataset reproduction steps are in
[`cross-datasets/README.md`](cross-datasets/README.md).

| Dataset | Domain | Size | Official source / link | Citation (in manuscript) |
|---|---|---|---|---|
| Main domain (TransHub) | Indonesian transmigration policy | 1,000 QA | authors' own (in [`data/`](data)) | — |
| **CUAD** | Legal contracts | 180 | [theatticusproject/cuad-qa (HF)](https://huggingface.co/datasets/theatticusproject/cuad-qa) | Hendrycks, Burns, Chen & Ball (2021) |
| **ConditionalQA** | Government / public policy | 180 | [haitian-sun/ConditionalQA (GitHub)](https://github.com/haitian-sun/ConditionalQA) | Sun, Cohen & Salakhutdinov (2022) |
| **FinanceBench** | Financial audits | 180 | [PatronusAI/financebench (HF)](https://huggingface.co/datasets/PatronusAI/financebench) | Islam et al. |
| **PubMedQA** | Biomedical / health | 200 | [qiaojin/PubMedQA (HF)](https://huggingface.co/datasets/qiaojin/PubMedQA) | Jin, Dhingra, Liu, Cohen & Lu (2019) |
| **ID-GovQA** | Indonesian government policy | 111 | authors' own, from **public open-data portals** ([`data/id_govqa_pakdwi_test_sample.json`](data/id_govqa_pakdwi_test_sample.json)) | authors |
| **Real-World Deployment** | BPJS/INA-CBG inpatient-claim screening | 60 episodes / 437 rules | de-identified, download via [`Real-World Deployment/data/deployment_data_public.zip`](Real-World%20Deployment/data/deployment_data_public.zip) (password-protected) | manuscript §Real-World |

> **Real-world deployment data.** A de-identified evaluation set of **60 inpatient
> episodes (437 admission rules, 60 claims)** from a production BPJS/INA-CBG
> claim-screening run. It is downloadable as a **password-protected archive** from
> [`Real-World Deployment/data/deployment_data_public.zip`](Real-World%20Deployment/data/deployment_data_public.zip);
> the password is **not** stored in this repository and is provided by the authors on
> request. The set is subject to Indonesia's **PDP Act (UU 27/2022)** (see
> [`Real-World Deployment/README.md`](Real-World%20Deployment/README.md)).

### How to reproduce the data (run the pipeline)

```bash
# 1. Build each KB from its public source (e.g. PubMedQA)
cd supplementary/cross_domain/pubmedqa
python 01_build_pubmedqa_kb.py        # downloads the dataset + builds the KB

# 2. Run the comparison (Standard RAG / Self-RAG / ARDA-SR)
python 02_run_pubmedqa_test.py

# The same two-step flow applies to cuad/, conditionalqa/, financebench/, and the
# main-domain pipeline/ (see pipeline/01_build_kb.py … 03_run_experiment.py)
```

Per-dataset download links: CUAD `/datasets/theatticusproject/cuad-qa` ·
ConditionalQA `github.com/haitian-sun/ConditionalQA` ·
FinanceBench `/datasets/PatronusAI/financebench` ·
PubMedQA `/datasets/qiaojin/PubMedQA`.

## Reproducible results

The comparison tables below are regenerated deterministically from the raw experiment
outputs in [`results/data/`](results/data) by
[`results/scripts/reproduce_tables.py`](results/scripts/reproduce_tables.py) — **no API
calls**. Full tables (CSV) are in [`results/tables/`](results/tables):
[`table6.csv`](results/tables/table6.csv) ·
[`table8.csv`](results/tables/table8.csv) ·
[`table9.csv`](results/tables/table9.csv); the trade-off figure in
[`results/figures/Figure6_relevance_vs_cost.png`](results/figures/Figure6_relevance_vs_cost.png).

Run it yourself:

```bash
cd results
python scripts/reproduce_tables.py    # regenerates table6/8/9.csv + Figure 6
```

### Table 6 — main-domain comparison (1,000 queries)

| Method | Rel ↑ | Faith ↑ | Cov ↑ | FRR ↓ | Lat (s) |
|---|---|---|---|---|---|
| LLM-Only | 0.611 | 0.321 | 0.558 | 0.330 | 1.19 |
| Standard RAG | 0.716 | 0.739 | 0.680 | 0.175 | 3.78 |
| Hybrid RAG | 0.746 | 0.771 | 0.721 | 0.164 | 4.07 |
| HyDE-RAG | 0.738 | 0.762 | 0.696 | 0.167 | 5.21 |
| Adaptive-RAG | 0.781 | 0.790 | 0.737 | 0.127 | 4.53 |
| CRAG | 0.759 | 0.805 | 0.725 | 0.128 | 4.75 |
| ReAct | 0.773 | 0.757 | 0.733 | 0.143 | 6.05 |
| Self-RAG | 0.788 | 0.813 | 0.746 | 0.115 | 5.74 |
| FLARE | 0.776 | 0.795 | 0.734 | 0.133 | 6.00 |
| IRCoT | 0.763 | 0.781 | 0.714 | 0.147 | 6.34 |
| **ARDA-SR** | **0.871** | **0.855** | **0.845** | **0.040** | 6.40 |

*ARDA-SR improves Relevance (+0.083 over the strongest baseline Self-RAG), Faithfulness
(+0.042), Coverage (+0.099) and cuts the False-Refusal Rate from 0.115 to 0.040, with a
modest latency increase (6.4 s) — the routing/arbitration layers reduce refusals without
sacrificing answer quality.*

### Table 9 — cross-domain generalization

| Domain | Method | Rel ↑ | Faith ↑ | Cov ↑ | FRR ↓ | Lat (s) |
|---|---|---|---|---|---|---|
| CUAD | Standard RAG | 0.276 | 0.872 | 0.263 | 0.851 | 1.22 |
| CUAD | Self-RAG | 0.669 | 0.824 | 0.660 | 0.713 | 4.26 |
| CUAD | **ARDA-SR** | 0.774 | 0.808 | 0.698 | 0.322 | 8.28 |
| ConditionalQA | Standard RAG | 0.660 | 0.881 | 0.551 | 0.375 | 1.22 |
| ConditionalQA | Self-RAG | 0.884 | 0.911 | 0.749 | 0.071 | 3.10 |
| ConditionalQA | **ARDA-SR** | **0.906** | 0.759 | 0.720 | 0.048 | 7.27 |
| FinanceBench | Standard RAG | 0.437 | 0.871 | 0.379 | 0.647 | 1.29 |
| FinanceBench | Self-RAG | 0.688 | 0.811 | 0.615 | 0.440 | 3.59 |
| FinanceBench | **ARDA-SR** | 0.644 | 0.623 | 0.520 | 0.273 | 7.70 |
| PubMedQA | Standard RAG | 0.673 | 0.940 | 0.498 | 0.325 | 1.31 |
| PubMedQA | Self-RAG | 0.896 | 0.947 | 0.762 | 0.110 | 2.92 |
| PubMedQA | **ARDA-SR** | 0.894 | 0.846 | 0.706 | 0.115 | 6.59 |
| ID-GovQA | Standard RAG | 0.773 | 0.993 | 0.672 | 0.303 | 1.23 |
| ID-GovQA | Self-RAG | 0.872 | 0.969 | 0.836 | 0.192 | 2.99 |
| ID-GovQA | **ARDA-SR** | **0.958** | 0.983 | **0.935** | 0.040 | 9.26 |

*Across four out-of-domain benchmarks and the real-world ID-GovQA set, ARDA-SR is best
on Relevance / Coverage and lowest on False-Refusal Rate in most domains. Its
Faithfulness is lower on narrative corpora (PubMedQA, FinanceBench) but remains strong
on rule-structured domains — a boundary worth noting when extending the framework.*

### Table 8 — cross-backbone robustness (ARDA-SR on ID-GovQA)

| Backbone | n(answerable) | n(unanswerable) | FRR ↓ | Correct refusals | ARR ↑ | FAR ↓ | Hit@5 | Lat (s) |
|---|---|---|---|---|---|---|---|---|
| gemma2:9b | 99 | 12 | 0.061 | 9 | 0.750 | 0.250 | 1.0 | 6.36 |
| llama3.1:8b | 99 | 12 | 0.222 | 9 | 0.750 | 0.250 | 1.0 | 6.54 |
| mixtral:8x7b | 99 | 12 | 0.040 | 5 | 0.417 | 0.583 | 1.0 | 13.68 |
| qwen2.5:14b | 99 | 12 | 0.051 | 8 | 0.667 | 0.333 | 1.0 | 6.25 |
| qwen2.5:7b | 99 | 12 | 0.263 | 12 | 1.000 | 0.000 | 1.0 | 6.41 |

*ARDA-SR's behavioural benefits transfer across open-source backbones of different
sizes (7B–8x7B) and a hosted gemma2:9b — the False-Refusal Rate stays low (0.04–0.26)
on the same ID-GovQA set, so the framework is not tied to a single model.*

### Figure 6 — Relevance vs computational cost

![Figure 6 — Relevance vs computational-cost trade-off across methods](results/figures/Figure6_relevance_vs_cost.png)

*Figure 6 (regenerated by `reproduce_tables.py`) plots answer Relevance against mean
latency for each method; ARDA-SR sits on the Pareto-optimal frontier — it achieves the
highest Relevance at a moderate cost, whereas cheaper methods (LLM-Only, Standard RAG)
trade away accuracy and costlier ones (ReAct, IRCoT) add little relevance for their
latency.*

### Latency: an intrinsic cost of the multi-stage architecture

ARDA-SR's mean latency is 6.40 s, and **no query answers in under 3 s** (min 3.26 s,
~74% under 7 s). This is **not a fixed per-query cost** — the routing layer adapts the
pipeline depth to the query — but even the cheapest path (m1) averages ~5.2 s because
every query always pays for the **Adaptive Query Router (AQR)** classification plus a
generation call:

| AQR mode | Processing path | n | Mean latency (s) |
|---|---|---|---|
| **m1** | Direct / parametric (AQR + generate) | 208 | 5.25 |
| **m2** | AQR + single retrieval + generate | 367 | 6.18 |
| **m3** | AQR + dual-draft arbitration | 211 | 6.05 |
| **m4** | AQR + scenario reasoning (SR) | 214 | 8.24 |

**Why it is not a defect.** ARDA-SR trades a higher up-front cost for substantially
better outcomes. The cheapest baseline, LLM-Only, answers in **1.19 s** (a single call)
but only reaches Relevance 0.611 and sets FRR to 0.330; ARDA-SR spends 6.40 s (routing +
arbitration + reasoning) to reach **Relevance 0.871** and **FRR 0.040**. Relative to the
other *reasoning-based* baselines (Self-RAG 5.74 s, ReAct 6.05 s, IRCoT 6.34 s) the extra
latency is small, and it buys a Pareto-optimal point (best quality, lowest refusal). For
latency-sensitive deployments the AQR threshold can be tuned so most queries skip the
scenario path — the cost is configurable, not fixed.

**Limitation & future work.** The higher latency is a recognised limitation for
latency-sensitive applications (see the manuscript's Limitations). Two concrete future
directions are planned to reduce it while keeping the quality gains (see
[`results/LATENCY_FUTURE_WORK.md`](results/LATENCY_FUTURE_WORK.md)):

- **early-exit routing** — route simple queries out before the expensive scenario/
  dual-draft stages;
- **parallel execution** of the two drafts in the dual-draft stage instead of
  sequentially.

## Real-World Deployment

A production deployment of ARDA-SR for automated **BPJS Kesehatan / INA-CBG
inpatient-claim screening in government question answering** on the Senopati AI platform
([https://senopati.its.ac.id/klaim-bpjs/](https://senopati.its.ac.id/klaim-bpjs/)) —
placed after the experimental results, as in the manuscript.

See **[`Real-World Deployment/`](Real-World%20Deployment)** for:

- deployment notes and the main result table (ARDA-SR vs Standard RAG vs Self-RAG);
- the deployment figure;
- a **password-protected** dataset archive of the de-identified evaluation tables.
  The archive can be downloaded from this repository, but the password is **not** stored
  here — it is provided by the authors on request (email the corresponding author).

**Table 11 — Real-World Application (BPJS/INA-CBG claim screening, three-judge panel)**

| Method | Acc.↑ | Prec.↑ | Rec.↑ | FRR↓ | ARR↑ | Rel ↑ | Faith ↑ | Cov ↑ |
|---|---|---|---|---|---|---|---|---|
| Standard RAG (baseline) | 0.795 | 0.78 | 0.65 | 0.11 | 0.65 | 3.12 | 3.53 | 2.63 |
| Self-RAG | 0.778 | 0.76 | 0.63 | 0.12 | 0.63 | 2.13 | 2.86 | 1.82 |
| **ARDA-SR** | **0.839** | 0.82 | 0.68 | **0.07** | 0.68 | **3.25** | **3.71** | **2.89** |

*On 60 real inpatient episodes (437 admission rules) ARDA-SR improves rule accuracy over
the Standard-RAG baseline (+0.044), lowers the false-refusal rate (0.11 → 0.07), and
produces more complete corrections (coverage 2.63 → 2.89). Self-RAG underperforms here
despite its extra reflection stage — the arbitration layer is most valuable on
rule-structured, claim-level tasks.*

No personal data is released; the institution and platform are anonymised in the text,
and the dataset is subject to Indonesia's **Personal Data Protection Act (UU 27/2022)**.

## Supplementary experiments

- **[`supplementary/cross_domain/`](supplementary/cross_domain)** — zero-shot transfer to four
  public benchmarks spanning legal, government-policy, finance, and biomedical domains (CUAD,
  ConditionalQA, FinanceBench, PubMedQA). Each dataset folder includes a `01_build_*_kb.py`
  that fetches the dataset from its official public source.
- **[`supplementary/unanswerable_queries/`](supplementary/unanswerable_queries)** — a 60-query
  diagnostic set (missing provinces, uncovered regulations, uncovered commodities, out-of-domain
  topics) used to verify the system appropriately declines to answer rather than hallucinating.

## Configuration

All method parameters — the entropy routing threshold, arbitration weights and decision
margin, the scenario-reasoning risk-aversion parameter, retrieval mixing weight, chunk size,
and model names — are centralized in [`config.py`](config.py).

## Contributors

The following contributors have contributed to the development, evaluation, and research associated with ARDA-SR:

- **Dr. Dwi Sunaryono** — Department of Informatics, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia (corresponding author)
- **Wawan Firgiawan** — Department of Informatics, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia
- **Bunga Laelatul Muna** — Department of Informatics, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia
- **Dr. Bilqis Amaliah** — Department of Informatics, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia
- **Dr. Shoffi Izza Sabila** — Department of Medical Technology, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia


## Citation

If you use this code or dataset, please cite the accompanying paper:

```
@article{munasunaryono2026arda,
  title   = {ARDA-SR: Entropy-Based Routing and Dual-Draft Arbitration for False
             Refusal Reduction and Scenario Reasoning in Government Question Answering},
  author  = {Bunga Laelatul Muna and Dwi Sunaryono and Bilqis Amaliah and
             Wawan Firgiawan and Shoffi Izza Sabila},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Under review}
}
```

> **Note.** This is a manuscript under review at *Expert Systems with Applications*
> (Elsevier); **no DOI is assigned yet**. A citation is provided for attribution only;
> the final bibliographic details (volume, pages, DOI) will follow publication. The
> corresponding author is **Dwi Sunaryono (dwi@its.ac.id)**.
