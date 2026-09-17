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

Every query first passes through an **entropy-based router**; a **parametric** draft and a
**retrieval-grounded** draft are then arbitrated against each other, and for questions tied
to policy scenarios the system reasons in an explicit way over several decision options.
The aim is twofold: fewer false refusals, and answers for public-sector QA systems that are
both more dependable and easier to audit.

Bundled here are the complete implementation, the benchmark dataset, and the records of
human annotation that were used for validation, allowing the results to be reproduced and
checked by others.

<div align="center">

![TransHub — an ARDA-SR-powered assistant for Indonesia's Ministry of Transmigration](image.png)

*TransHub: an assistant powered by ARDA-SR, deployed for Indonesia's Ministry of Transmigration*

</div>

## Contents

| Path | Description |
|---|---|
| [`arda_sr/`](arda_sr) | The ARDA-SR method itself: hybrid retrieval plus Adaptive Query Router (AQR), Dual-Draft Arbitrator (DDA), and Scenario Reasoning (SR) |
| [`baselines/`](baselines) | 10 baselines for comparison — Standard RAG, Hybrid RAG, HyDE-RAG, Adaptive-RAG, CRAG, ReAct, Self-RAG, FLARE, IRCoT, LLM-only |
| [`evaluation/`](evaluation) | Judging of answer quality (Relevance / Faithfulness / Coverage), metrics for routing and refusal, and statistical testing |
| [`pipeline/`](pipeline) | Scripts covering the whole flow: build KB → generate benchmark → run experiment → ablation → analyze |
| [`data/`](data) | The QA benchmark — 1,000 main-domain QA pairs + 111 real-world Indonesian government QA pairs (ID-GovQA) |
| [`annotations/`](annotations) | Benchmark validation at item level, carried out by three independent annotators |
| [`supplementary/`](supplementary) | Zero-shot cross-domain transfer + a diagnostic for unanswerable queries |
| [`results/`](results) | Comparison tables that reproduce (Table 6, 8, 9), Figure 6, raw per-query experiment outputs, and the `reproduce_tables.py` script |
| [`cross-datasets/`](cross-datasets) | Public cross-domain datasets (CUAD / ConditionalQA / FinanceBench / PubMedQA) — download links, schema, and per-dataset reproduction steps |
| [`Real-World Deployment/`](Real-World%20Deployment) | A real-world deployment of ARDA-SR for automated BPJS/INA-CBG inpatient-claim screening in government question answering (Senopati AI platform: <https://senopati.its.ac.id/klaim-bpjs/>) — deployment notes, main-result table, figure, and a **password-protected** dataset archive (password on request from the authors) |
| `config.py` | Every method parameter gathered in one place |

**Not included:** the source document corpus, the model weights, and API keys.

## Quick start

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure API keys** — put a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
```

| Key | Used for |
|---|---|
| `GEMINI_API_KEY` | Generation, routing, and retrieval-grounded drafting (the core ARDA-SR model) |
| `ANTHROPIC_API_KEY` | Used for QA benchmark generation only |
| `OPENAI_API_KEY` | Answer-quality judging done independently — drawn from a different model family than the generator, so that self-evaluation bias is avoided |

**3. Add your document corpus** — drop source documents into `data/`, using the layout that
`utils/kb_builder.py` expects.

**4. Run the pipeline**

```bash
python pipeline/01_build_kb.py
python pipeline/02_generate_qa_claude.py
python pipeline/verify_answerability.py
python pipeline/03_run_experiment.py       # ARDA-SR + all 10 baselines
python pipeline/04_ablation.py             # component-wise ablation study
python pipeline/05_analyze_results.py      # aggregate into summary tables
```

Every script takes the output of the one before it as input.

> **Only want to re-run evaluation on the benchmark that already exists?** The QA pairs used
> in the paper are already present in `data/qa_dataset.json` and
> `data/id_govqa_pakdwi_test_sample.json` — so you can jump directly to step 4, using your
> own knowledge base.

## Datasets

Evaluation of the framework rests on a main-domain dataset (1000 government QA pairs),
augmented by **four public cross-domain benchmarks** and a **real-world Indonesian
government set**. Every cross-domain dataset can be obtained directly from its official
public source (see the [`cross-datasets/`](cross-datasets) repository); the full schema
together with per-dataset reproduction steps appears in
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

> **Real-world deployment data.** From a production BPJS/INA-CBG claim-screening run comes
> a de-identified evaluation set of **60 inpatient episodes (437 admission rules, 60
> claims)**. You can download it as a **password-protected archive** from
> [`Real-World Deployment/data/deployment_data_public.zip`](Real-World%20Deployment/data/deployment_data_public.zip);
> the password is **not** kept in this repository and the authors supply it on request.
> Indonesia's **PDP Act (UU 27/2022)** applies to this set (see
> [`Real-World Deployment/README.md`](Real-World%20Deployment/README.md)).

### How to reproduce the data (run the pipeline)

```bash
# 1. Build each KB from its public source (e.g. PubMedQA)
cd supplementary/cross_domain/pubmedqa
python 01_build_pubmedqa_kb.py        # fetches the dataset, then builds the KB

# 2. Run the comparison (Standard RAG / Self-RAG / ARDA-SR)
python 02_run_pubmedqa_test.py

# This same two-step flow works for cuad/, conditionalqa/, financebench/, and the
# main-domain pipeline/ (see pipeline/01_build_kb.py … 03_run_experiment.py)
```

Download links per dataset: CUAD `/datasets/theatticusproject/cuad-qa` ·
ConditionalQA `github.com/haitian-sun/ConditionalQA` ·
FinanceBench `/datasets/PatronusAI/financebench` ·
PubMedQA `/datasets/qiaojin/PubMedQA`.

## Reproducible results

The comparison tables shown below are rebuilt deterministically, with **no API calls**,
from the raw experiment outputs in [`results/data/`](results/data) using
[`results/scripts/reproduce_tables.py`](results/scripts/reproduce_tables.py). The full
tables (CSV) live in [`results/tables/`](results/tables):
[`table6.csv`](results/tables/table6.csv) ·
[`table8.csv`](results/tables/table8.csv) ·
[`table9.csv`](results/tables/table9.csv); the trade-off figure in
[`results/figures/Figure6_relevance_vs_cost.png`](results/figures/Figure6_relevance_vs_cost.png).

Run it yourself:

```bash
cd results
python scripts/reproduce_tables.py    # rebuilds table6/8/9.csv + Figure 6
```

### Table 6 — main-domain comparison (1,000 queries)

| Method | Rel ↑ | Faith ↑ | Cov ↑ | Hit@5 ↑ | CtxRel ↑ | RoutingAcc ↑ | SRComp ↑ | FRR ↓ | Lat (s) ↓ |
|---|---|---|---|---|---|---|---|---|---|
| LLM-Only | 0.611±0.151 | 0.321±0.130 | 0.558±0.143 | – | – | – | – | 0.330 | 1.2 |
| Standard RAG | 0.716±0.095 | 0.739±0.116 | 0.680±0.097 | 0.768 | 0.702 | – | – | 0.175 | 3.8 |
| Hybrid RAG | 0.746±0.092 | 0.771±0.106 | 0.721±0.094 | 0.796 | 0.733 | – | – | 0.164 | 4.1 |
| HyDE-RAG | 0.738±0.095 | 0.762±0.112 | 0.696±0.092 | 0.786 | 0.719 | – | – | 0.167 | 5.2 |
| Adaptive-RAG | 0.782±0.090 | 0.790±0.107 | 0.737±0.089 | 0.837 | 0.768 | 0.727 | 0.711 | 0.127 | 4.5 |
| CRAG | 0.759±0.092 | 0.805±0.103 | 0.725±0.091 | 0.832 | 0.751 | – | – | 0.128 | 4.8 |
| ReAct | 0.773±0.088 | 0.757±0.111 | 0.733±0.091 | 0.797 | 0.754 | 0.687 | 0.673 | 0.143 | 6.0 |
| Self-RAG | 0.788±0.088 | 0.813±0.098 | 0.746±0.086 | 0.840 | 0.774 | 0.706 | 0.698 | 0.115 | 5.7 |
| FLARE | 0.776±0.090 | 0.795±0.105 | 0.734±0.086 | 0.819 | 0.761 | – | – | 0.133 | 6.0 |
| IRCoT | 0.763±0.094 | 0.781±0.109 | 0.714±0.089 | 0.805 | 0.757 | – | – | 0.147 | 6.3 |
| **ARDA-SR** | **0.871±0.080†** | **0.855±0.081†** | **0.845±0.084†** | **0.878†** | **0.812†** | **0.878†** | **0.821†** | **0.040†** | 6.4 |

*Relative to the strongest baseline (Self-RAG), ARDA-SR lifts Relevance by +0.083,
Faithfulness by +0.042, and Coverage by +0.099, while the False-Refusal Rate falls from
0.115 to 0.040 at only a modest latency cost (6.4 s) — in other words, the
routing/arbitration layers bring refusals down without giving up answer quality. The †
marker for statistical significance against the best baseline comes from a Wilcoxon
signed-rank test, *p* < 0.001. This table is a reproduction of the manuscript's Table 6;
raw per-query outputs for Rel/Faith/Cov/FRR/Lat sit in [`results/data/`](results/data),
whereas the `Hit@5`, `CtxRel`, `RoutingAcc` and `SRComp` columns together with the
standard deviations are taken from the manuscript (the
[`reproduce_tables.py`](results/scripts/reproduce_tables.py) script rebuilds the mean
columns out of the raw outputs).*

### Table 8 — cross-backbone robustness

Here we reproduce the manuscript's **Table 8** (a cross-backbone comparison over 1,000 test
queries), which reports **Standard RAG, Self-RAG, and ARDA-SR** across three distinct
backbones. The raw per-backbone, per-method order data behind this table is **not
included in this repository** (only the aggregated values appear in the manuscript); the
three backbones are Gemini 2.5 Flash, Qwen2.5-1.5B, and Phi-3mini:

| Backbone | Method | Rel ↑ | Faith ↑ | Cov ↑ | Hit@5 | CtxRel | RoutingAcc | SRComp | FRR ↓ | Lat (s) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemini 2.5 Flash | Standard RAG | 0.716 | 0.739 | 0.680 | 0.768 | 0.702 | – | – | 0.175 | 3.8 |
| Gemini 2.5 Flash | Self-RAG | 0.788 | 0.813 | 0.746 | 0.840 | 0.774 | 0.706 | 0.698 | 0.115 | 5.7 |
| Gemini 2.5 Flash | **ARDA-SR** | 0.871 | 0.855 | 0.845 | 0.878 | 0.812 | 0.878 | 0.821 | 0.040 | 6.4 |
| Qwen2.5-1.5B | Standard RAG | 0.665 | 0.690 | 0.640 | 0.724 | 0.658 | – | – | 0.235 | 5.9 |
| Qwen2.5-1.5B | Self-RAG | 0.735 | 0.755 | 0.700 | 0.791 | 0.725 | 0.642 | 0.632 | 0.158 | 7.6 |
| Qwen2.5-1.5B | **ARDA-SR** | 0.812 | 0.826 | 0.786 | 0.866 | 0.803 | 0.801 | 0.742 | 0.083 | 8.9 |
| Phi-3mini | Standard RAG | 0.690 | 0.712 | 0.662 | 0.748 | 0.684 | – | – | 0.215 | 7.1 |
| Phi-3mini | Self-RAG | 0.758 | 0.778 | 0.724 | 0.816 | 0.748 | 0.671 | 0.655 | 0.140 | 9.2 |
| Phi-3mini | **ARDA-SR** | 0.833 | 0.842 | 0.807 | 0.884 | 0.821 | 0.829 | 0.776 | 0.067 | 10.8 |

*The behavioural gains of ARDA-SR carry over to backbones that differ in size and in
MRLM-family (Gemini, Qwen, Phi): quality is retained and the False-Refusal Rate remains
low (0.040–0.083), which shows the framework is not bound to any single model. What drives
the improvement is the architectural design (entropy-based routing, dual-draft arbitration,
structured reasoning) rather than model capacity.*

#### Backbone sweep on ID-GovQA (ARDA-SR only)

Also present among the repository's raw outputs is a separate **ARDA-SR-only backbone
sweep** on ID-GovQA (the `results/data/arda_sr_{backbone}_pakdwi_summary.json` files).
Because the backbones in the manuscript's cross-backbone table differ, this counts as a
supplementary result; it therefore appears here under its own heading and is **not** a
reproduction of Table 8:

| Backbone | n(answerable) | n(unanswerable) | FRR ↓ | Correct refusals | ARR ↑ | FAR ↓ | Hit@5 | Lat (s) |
|---|---|---|---|---|---|---|---|---|
| gemma2:9b | 99 | 12 | 0.061 | 9 | 0.750 | 0.250 | 1.0 | 6.36 |
| llama3.1:8b | 99 | 12 | 0.222 | 9 | 0.750 | 0.250 | 1.0 | 6.54 |
| mixtral:8x7b | 99 | 12 | 0.040 | 5 | 0.417 | 0.583 | 1.0 | 13.68 |
| qwen2.5:14b | 99 | 12 | 0.051 | 8 | 0.667 | 0.333 | 1.0 | 6.25 |
| qwen2.5:7b | 99 | 12 | 0.263 | 12 | 1.000 | 0.000 | 1.0 | 6.41 |

*As this in-repo sweep demonstrates, ARDA-SR maintains a low False-Refusal Rate (0.04–0.26)
on ID-GovQA across open-source backbones of varied size (7B–8x7B) as well as a hosted
gemma2:9b.*

### Table 9 — cross-domain generalization

Zero-shot transfer onto four out-of-domain datasets plus the real-world ID-GovQA set, run
with Gemini 2.5 Flash and the same setup as the main dataset. This is a reproduction of the
manuscript's Table 9 (standard deviations appear in the manuscript; since
`reproduce_tables.py` does not use the raw per-query outputs, the mean values below are
taken from the manuscript).

| Dataset | Method | Rel ↑ | Faith ↑ | Cov ↑ | FRR ↓ |
|---|---|---|---|---|---|
| CUAD | Standard RAG | 0.276±0.234† | 0.872±0.287 | 0.263±0.198† | 0.851 |
| CUAD | Self-RAG | 0.669±0.384† | 0.824±0.308 | 0.660±0.367 | 0.713 |
| CUAD | **ARDA-SR** | **0.774±0.344** | **0.808±0.320** | **0.698±0.339** | **0.322** |
| ConditionalQA | Standard RAG | 0.660±0.389† | 0.881±0.265‡ | 0.551±0.328† | 0.375 |
| ConditionalQA | Self-RAG | 0.884±0.242 | 0.911±0.186‡ | 0.749±0.242 | 0.071 |
| ConditionalQA | **ARDA-SR** | **0.906±0.229** | 0.759±0.295 | 0.720±0.227 | **0.048** |
| FinanceBench | Standard RAG | 0.437±0.357† | 0.871±0.278‡ | 0.379±0.301† | 0.647 |
| FinanceBench | Self-RAG | 0.688±0.379 | 0.811±0.317‡ | 0.615±0.358‡ | 0.440 |
| FinanceBench | **ARDA-SR** | 0.644±0.378 | 0.623±0.384 | 0.520±0.340 | **0.273** |
| PubMedQA | Standard RAG | 0.673±0.372† | 0.940±0.191‡ | 0.498±0.273† | 0.325 |
| PubMedQA | Self-RAG | 0.896±0.236 | 0.947±0.157‡ | 0.762±0.233‡ | 0.110 |
| PubMedQA | **ARDA-SR** | 0.894±0.225 | 0.846±0.249 | 0.706±0.226 | 0.115 |
| ID-GovQA | Standard RAG | 0.773±0.358† | 0.993±0.076 | 0.672±0.335† | 0.303 |
| ID-GovQA | Self-RAG | 0.872±0.291† | 0.969±0.140 | 0.836±0.267† | 0.192 |
| ID-GovQA | **ARDA-SR** | **0.958±0.095** | 0.983±0.072 | **0.935±0.168** | **0.040** |

**Note.** Unanswerable slice: ID-GovQA 12/111, ConditionalQA 12/180, CUAD 93/180;
FinanceBench/PubMedQA 0. †/‡ mark paired Wilcoxon tests against ARDA-SR; the best value for
each dataset and metric appears in bold. The latency column is deliberately left out here,
since ARDA-SR is not benchmarked for cross-domain latency — cross-domain results are about
generalisation, and the reference latency (6.4 s) can be found in Table 6.

*On the four out-of-domain benchmarks and ID-GovQA, ARDA-SR achieves the best Relevance /
Coverage and the lowest False-Refusal Rate in most domains. Its Faithfulness drops on
narrative corpora (PubMedQA, FinanceBench) yet stays strong on rule-structured domains — a
boundary worth keeping in mind when the framework is extended.*

### Figure 6 — Relevance vs computational cost

![Figure 6 — Relevance vs computational-cost trade-off across methods](results/figures/Figure6_relevance_vs_cost.png)

*Figure 6 (rebuilt by `reproduce_tables.py`) charts answer Relevance against mean latency
for every method; ARDA-SR lies on the Pareto-optimal frontier — it reaches the highest
Relevance for a moderate cost, while the cheaper methods (LLM-Only, Standard RAG) give up
accuracy, and the costlier ones (ReAct, IRCoT) gain little Relevance for the latency they
add.*

### Latency: an intrinsic cost of the multi-stage architecture

ARDA-SR averages 6.40 s of latency, and **no query is answered in under 3 s** (min 3.26 s,
with ~74% under 7 s). That figure is **not a fixed per-query cost** — the routing layer
adjusts how deep the pipeline goes according to the query — yet the cheapest path (m1)
still averages ~5.2 s, since each query always incurs the **Adaptive Query Router (AQR)**
classification plus one generation call:

| AQR mode | Processing path | n | Mean latency (s) |
|---|---|---|---|
| **m1** | Direct / parametric (AQR + generate) | 208 | 5.25 |
| **m2** | AQR + single retrieval + generate | 367 | 6.18 |
| **m3** | AQR + dual-draft arbitration | 211 | 6.05 |
| **m4** | AQR + scenario reasoning (SR) | 214 | 8.24 |

**Why this is not a defect.** In exchange for a higher up-front cost, ARDA-SR delivers
substantially better outcomes. LLM-Only, the cheapest baseline, replies in **1.19 s** (one
call) but reaches only Relevance 0.611 and an FRR of 0.330; by spending 6.40 s (routing +
arbitration + reasoning), ARDA-SR reaches **Relevance 0.871** and an **FRR of 0.040**. Set
against the other *reasoning-based* baselines (Self-RAG 5.74 s, ReAct 6.05 s, IRCoT
6.34 s), the added latency is small, and it secures a Pareto-optimal point (best quality,
lowest refusal). Deployments that are sensitive to latency can tune the AQR threshold so
that the scenario path is skipped for most queries — the cost is configurable rather than
fixed.

**Latency depends on the workload; it is not a fixed per-query cost.**
The 6.40 s mean is taken on the **cloud Gemini backbone** at **1,000 queries**. In the
on-premise deployment, latency instead scales **in proportion to the number of clinical
rules** processed (Pearson r = 0.925 across the 60-episode deployment corpus), rather than
being a fixed per-query cost:

| Clinical rules processed | n | On-premise latency (s), median (range) |
|---|---|---|
| 0 (no rule triggered) | 11 | 0.80 (0.80–0.81) |
| 2–4 (typical claim) | 18 | 5.6 (3.2–17.3) |
| 5–10 | 16 | 16.1 (6.3–33.0) |
| 11–23 | 11 | 24.6 (19.6–46.2) |
| 24–36 (batch) | 4 | 45.4 (36.7–73.8) |

Latency therefore climbs with the workload: a claim triggering no rules finishes in 0.80 s,
whereas a large batch of 24–36 rules needs 36.7–73.8 s. Across the corpus the mean is 15.4 s
per claim (median 11.7 s; per-rule ≈ 2.3 s; total n = 60 episodes). Bear in mind that some
2–4-rule claims are still slow (up to 17.3 s), because in those cases the extra latency
stems from the **scenario reasoning (m4)** / complex rules and not from rule count alone — so
"latency ∝ rule count" holds most strongly at large scale. That is precisely what the
efficiency improvements planned below are meant to address.

**Limitation & future work.** For applications sensitive to latency, the higher latency is
acknowledged as a limitation (see the manuscript's Limitations). Two specific directions
for future work are planned, aimed at cutting it while preserving the quality gains:

- **early-exit routing** — send simple queries out before reaching the costly scenario/
  dual-draft stages;
- **parallel execution** of both drafts within the dual-draft stage, and **parallelising
  rule-level processing** in the claim-verification workflow, in place of sequential
  handling.

## Real-World Deployment

A production deployment of ARDA-SR that automates **BPJS Kesehatan / INA-CBG
inpatient-claim screening in government question answering**, running on the Senopati AI
platform ([https://senopati.its.ac.id/klaim-bpjs/](https://senopati.its.ac.id/klaim-bpjs/)) —
positioned after the experimental results, following the manuscript.

See **[`Real-World Deployment/`](Real-World%20Deployment)** for:

- deployment notes together with the main result table (ARDA-SR vs Standard RAG vs Self-RAG);
- the deployment figure;
- a **password-protected** dataset archive holding the de-identified evaluation tables.
  Although the archive is downloadable from this repository, the password is **not** stored
  here — the authors provide it upon request (contact the corresponding author by email).

**Table 11 — Real-World Application (BPJS/INA-CBG claim screening, three-judge panel)**

| Method | Rule level Acc.↑ | Prec.↑ | Rec.↑ | FRR ↓ | ARR ↑ | F1 | Suggestion quality Rel.↑ | Faith.↑ | Cov.↑ |
|---|---|---|---|---|---|---|---|---|---|
| Standard RAG (baseline) | 0.795 | 0.78 | 0.65 | 0.115 | 0.646 | 0.703 | 3.12 | 3.53 | 2.63 |
| Self-RAG * | 0.778 | 0.76 | 0.63 | 0.180 | 0.708 | 0.705 | 2.13 | 2.86 | 1.82 |
| **ARDA-SR (Ours)** | **0.839** | 0.82 | 0.68 | **0.083** | **0.708** | **0.767** | **3.25** | **3.71** | **2.89** |
| Δ (ARDA-SR − Standard RAG) | +0.044 | +0.04 | +0.03 | −0.032 | +0.062 | +0.064 | +0.13 | +0.18 | +0.26 |

*A production screening system for BPJS Kesehatan / INA-CBG inpatient claims, whose reference
labels come from an independent three-LLM judge panel (Qwen3.8Max, DeepSeek-V4, Claude; ties
resolve to FAIL). Every rule-level metric below is computed over the **same n=347 criteria**
for the three methods, keeping the comparison fair (apple-to-apple): 347 is the count of
criteria that all three systems were able to adjudicate — Self-RAG parsed **44 of 49**
episodes carrying criteria and **failed on 5** (90 criteria), since the model returned an
unparseable JSON for long or compound rules. Suggestion quality (Rel/Faith/Cov on a 1–5
scale) is measured over **n=180** ratings for Standard RAG and ARDA-SR and over **n=54**
episodes for Self-RAG. A judge scores the criterion against the raw electronic medical record
alone (the production verdict is never shown). Every model runs locally on-premise
(Qwen3.5:9B as backbone, Gemma2:9B as ARDA-SR dual-draft arbiter); running on-premise keeps
patient data in line with Law No. 27 of 2022 on Personal Data Protection.*

*\* Self-RAG relies on its reflection-only adaptation (generation -> self-reflection ->
finalize). Among the 49 episodes carrying adjudicated criteria, Self-RAG parsed **44** (347
criteria) successfully and **failed on 5** (90 criteria) — the LLM returned an unparseable
JSON for long or compound rules, and repeated attempts at the run gave the same result. So
that the comparison stays fair, the rule-level metrics in Table 11 for **all three** methods
(Standard RAG, Self-RAG, ARDA-SR) are computed over the **same n=347 criteria** Self-RAG
could adjudicate, rather than over n=437 for Standard RAG / ARDA-SR and n=347 for Self-RAG.
(Note: n=437 is the dataset's total criteria count, given in Section 3.7 as a structural
fact; n=347 is the subset that is comparable across the three systems.)*

No personal data is released; the text anonymises both the institution and the platform,
and Indonesia's **Personal Data Protection Act (UU 27/2022)** governs the dataset.

## Supplementary experiments

- **[`supplementary/cross_domain/`](supplementary/cross_domain)** — zero-shot transfer onto four
  public benchmarks that span the legal, government-policy, finance, and biomedical domains
  (CUAD, ConditionalQA, FinanceBench, PubMedQA). Inside each dataset folder is a
  `01_build_*_kb.py` that pulls the dataset from its official public source.
- **[`supplementary/unanswerable_queries/`](supplementary/unanswerable_queries)** — a 60-query
  diagnostic set (missing provinces, uncovered regulations, uncovered commodities,
  out-of-domain topics), used to confirm that the system declines to answer where appropriate
  rather than hallucinating.

## Configuration

Every method parameter — entropy routing threshold, arbitration weights and decision margin,
the scenario-reasoning risk-aversion parameter, retrieval mixing weight, chunk size, and
model names — is centralized in [`config.py`](config.py).

## Contributors

The people listed below contributed to the development, evaluation, and research work behind ARDA-SR:

- **Dr. Dwi Sunaryono** — Department of Informatics, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia (corresponding author)
- **Bunga Laelatul Muna** — Department of Informatics, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia
- **Wawan Firgiawan** — Department of Informatics, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia
- **Dr. Shoffi Izza Sabila** — Department of Medical Technology, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia
- **Dr. Bilqis Amaliah** — Department of Informatics, Institut Teknologi Sepuluh Nopember, Surabaya 60111, Indonesia

## Citation

Should you use this code or dataset, please cite the paper that accompanies it:

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

> **Note.** The manuscript is under review at *Expert Systems with Applications* (Elsevier);
> **no DOI has been assigned yet**. The citation above serves attribution only; the final
> bibliographic details (volume, pages, DOI) will appear once publishing is complete. The
> corresponding author is **Dwi Sunaryono (dwi@its.ac.id)**.
