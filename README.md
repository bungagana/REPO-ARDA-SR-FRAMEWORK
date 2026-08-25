<div align="center">

# ARDA-SR

**Adaptive Retrieval-Decision Architecture with Dual-Draft Arbitration and Scenario Reasoning**

A Retrieval-Augmented Generation framework for reliable, auditable government question-answering

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-TBD-lightgrey)
![Status](https://img.shields.io/badge/status-research%20code-orange)

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
| `config.py` | All method parameters in one place |

**Not included:** the source document corpus, model-generated answers/results from the
reported experiments, and API keys.

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

- **Wawan Firgiawan**


## Citation

If you use this code or dataset, please cite the accompanying paper (details to follow upon
publication).
