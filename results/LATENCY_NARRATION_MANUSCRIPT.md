# Draft: latency & routing narration for the manuscript (honest version)

## Correct framing

The manuscript currently states:

> "ARDA-SR has the highest average latency (6.4 seconds), reflecting the additional
> processing steps in its multi-stage architecture."

This is accurate but under-specified. The key facts that must be communicated — so the
latency is read as an **intrinsic cost of a quality/robustness trade-off**, not as a
defect — are:

1. **No query answers in under 3 s.** The observed latency range is 3.3–10.1 s
   (mean 6.40, median 6.07, p90 8.72). There is no "fast" sub-second path.
2. **The latency is not uniform** but adaptive: the router picks the cheapest sufficient
   stage, so the average is dominated by a tail of complex scenario queries. However,
   even the cheapest path (m1) averages ~5.2 s because **every query pays for the AQR
   routing call in addition to generation** (two LLM calls in the cheap path).
3. **It is the price of much higher quality.** The fastest baseline, LLM-Only, answers
   in **1.19 s** (a single call) but reaches Relevance 0.611 and a False-Refusal Rate of
   0.330. ARDA-SR spends 6.40 s to reach **Relevance 0.871** and **FRR 0.040**. Relative
   to other *reasoning-based* baselines (Self-RAG 5.74 s, ReAct 6.05 s, IRCoT 6.34 s)
   the additional latency is modest and yields a Pareto-optimal point.

## Replacement paragraph (Results)

"ARDA-SR incurs the highest mean latency of the evaluated methods (6.40 s; range
3.3–10.1 s, 74% under 7 s). This reflects an intrinsic cost of its multi-stage design:
each query first passes the Adaptive Query Router (AQR) — an extra LLM call — so even
the direct/parametric path (m1) averages 5.25 s, while scenario-based reasoning (m4)
reaches 8.24 s. Crucially, this cost buys a substantial gain in answer quality and
robustness: the fastest baseline, LLM-Only, answers in 1.19 s with a single call but
achieves only Relevance 0.611 and a False-Refusal Rate of 0.330, whereas ARDA-SR
achieves Relevance 0.871 and FRR 0.040. Compared with the other reasoning-based
baselines (Self-RAG 5.74 s, ReAct 6.05 s, IRCoT 6.34 s), the extra latency is modest and
places ARDA-SR on the Pareto-optimal frontier of the quality/cost trade-off. For
latency-sensitive deployments the AQR threshold can be tuned so most queries skip the
scenario path."

## Optional table (place near Figure 6)

| AQR mode | Processing path | n | Mean latency (s) |
|---|---|---|---|
| m1 | Direct / parametric (AQR + generate) | 208 | 5.25 |
| m2 | AQR + single retrieval + generate | 367 | 6.18 |
| m3 | AQR + dual-draft arbitration | 211 | 6.05 |
| m4 | AQR + scenario reasoning (SR) | 214 | 8.24 |
| **all** | — | 1000 | **6.40** |

## Response-to-reviewer note

> "ARDA-SR does not produce sub-second answers: its latency ranges 3.3–10.1 s (median
> 6.1 s) because every query pays for the routing call plus generation, and complex
> policy queries add scenario reasoning. This is the deliberate price of higher quality
> (Relevance 0.871 vs 0.611 for LLM-Only) and a lower false-refusal rate (FRR 0.040 vs
> 0.330). Relative to other reasoning baselines the cost is modest and Pareto-optimal;
> the AQR threshold is tunable for latency-constrained deployments."