# Latency — limitation and future work

## Why latency matters

ARDA-SR is a **multi-stage** pipeline: every query first passes the Adaptive Query Router
(AQR) and then (depending on the route) a retrieval, a dual-draft arbitration, and/or
scenario reasoning. This yields the best answer quality (Relevance 0.871) and lowest
false-refusal rate (FRR 0.040) of all compared methods, at a mean latency of **6.40 s**
(range 3.3–10.1 s; 74% under 7 s). Even the cheapest path (m1) averages ~5.2 s because it
still pays for the AQR routing call plus generation, and **no query answers in under 3 s**.

This is a recognised **limitation** for latency-sensitive deployments (e.g. interactive or
high-throughput production use). It is stated explicitly in the manuscript's Limitations
section: *"ARDA-SR has a higher mean response latency than the single-path baselines
(6.4 s), which may limit its use in latency-sensitive applications."*

## Planned future work (mirrors the manuscript)

Two concrete directions are planned to cut the latency **without losing** the quality /
robustness gains, both flagged in the manuscript's Future Work:

1. **Early-exit routing** (AQR). Currently the router decides the deepest stage needed;
   an early-exit mechanism would let simple, unambiguous queries be answered by the fast
   parametric path and **return before** entering the retrieval/arbitration/scenario
   stages, so the majority of queries pay only the m1 cost.
2. **Parallel execution in dual-draft arbitration.** The two drafts (direct and
   extractive) are currently generated **sequentially**; executing them **in parallel**
   (independent LLM calls) would roughly halve the arbitration time for ambiguous
   queries without changing the resulting verdict.

Additional levers under study:
- **Lighter AQR prompt / cached routing** to shorten the routing call itself.
- **Configurable AQR threshold** so deployments can trade a small amount of accuracy for
  responsiveness.

## Summary

| | ARDA-SR | LLM-Only | Self-RAG |
|---|---|---|---|
| Latency (s) | 6.40 | 1.19 | 5.74 |
| Relevance | **0.871** | 0.611 | 0.788 |
| FRR ↓ | **0.040** | 0.330 | 0.115 |

The latency is an intrinsic, acknowledged cost of a quality/robustness trade-off, and
concrete efficiency improvements (early-exit + parallel DDA) are already scoped as the
immediate next step.