# Teks untuk kolega — revisi manuskrip (latency & consistency)

Berikut teks yang perlu **ditambahkan/diganti** di manuskrip agar konsisten dengan data
dan repo GitHub. Silakan copy-paste ke bagian yang sesuai.

---

## 1. GANTI kalimat ini (di Results, tentang latency)

Kalimat yang ada sekarang (perlu diganti):
> "ARDA-SR has the highest average latency (6.4 seconds), reflecting the additional
> processing steps in its multi-stage architecture."

Ganti dengan paragraf ini:

> "ARDA-SR incurs the highest mean latency of the evaluated methods (6.40 s; range
> 3.3–10.1 s, 74% under 7 s). This is an intrinsic cost of its multi-stage design: each
> query first passes the Adaptive Query Router (AQR), an additional LLM call, so even
> the direct/parametric path (m1) averages 5.25 s, while the scenario-based path (m4)
> reaches 8.24 s. Notably, **no query is answered in under 3 s**. Crucially, this cost
> buys a large gain in answer quality and robustness: the fastest baseline, LLM-Only,
> answers in 1.19 s with a single call but attains only Relevance 0.611 and a False-
> Refusal Rate of 0.330, whereas ARDA-SR attains Relevance 0.871 and FRR 0.040. Compared
> with the other reasoning-based baselines (Self-RAG 5.74 s, ReAct 6.05 s, IRCoT
> 6.34 s), the additional latency is modest and places ARDA-SR on the Pareto-optimal
> frontier of the quality/cost trade-off (Figure 6)."

---

## 2. TAMBAHKAN tabel ini (dekat Figure 6)

**Table 12 — Per-mode latency (adaptive routing).**

| AQR mode | Processing path | n | Mean latency (s) |
|---|---|---|---|
| m1 | Direct / parametric (AQR + generate) | 208 | 5.25 |
| m2 | AQR + single retrieval + generate | 367 | 6.18 |
| m3 | AQR + dual-draft arbitration | 211 | 6.05 |
| m4 | AQR + scenario reasoning (SR) | 214 | 8.24 |
| **all** | — | 1000 | **6.40** |

---

## 3. PERKUAT Limitation #3 (sudah ada, perlu ditambah detail)

Limitation #3 saat ini:
> "Third, ARDA-SR has a higher mean response latency than the single path baselines
> (6.4 s), which may limit its use in latency-sensitive applications."

Perkuat dengan kalimat berikut (setelahnya):

> "This latency is especially relevant for interactive or high-throughput production
> use: even straightforward queries take 3.3–5.3 s to answer, so the current
> configuration is best suited to asynchronous, decision-support workloads rather than
> real-time chat."

---

## 4. PERKUAT Future Work #1 (kunci menjawab kekhawatiran reviewer)

Future Work saat ini (poin 1):
> "1. Improve efficiency through early-exit routing mechanisms and parallel execution
> during the dual-draft stage."

Ganti dengan yang lebih spesifik & meyakinkan:

> "1. **Reduce latency without sacrificing quality.** Concretely, we plan (a) an
> early-exit routing mechanism in the AQR so that simple, unambiguous queries are
> answered by the fast parametric path and return before entering the retrieval,
> arbitration, or scenario stages — today even the cheapest path (m1) costs ~5.2 s
> because every query pays for the routing call plus generation; and (b) parallel
> execution of the two drafts in the dual-draft stage, which are currently generated
> sequentially. A lighter AQR prompt and a tunable entropy threshold will let
> deployments trade a small amount of accuracy for responsiveness in latency-sensitive
> settings."

---

## 5. Catatan konsistensi (penting)

- Pastikan angka yang dipakai **konsisten** di seluruh manuskrip: mean latency **6.40 s**,
  sub-3s **tidak ada**, per-mode **m1 5.25 / m4 8.24 s**.
- Narasi **jangan** menyebut latensi "adaptif sehingga simple query *jauh lebih cepat*"
  — faktanya semua query ≥ 3.3 s dan m1 rata-rata 5.25 s. Klaim "fast path" akan
  dibantah reviewer oleh data yang sama.
- Framing yang aman: **biaya intrinsik multi-stage → harga kualitas (Relevance 0.871,
  FRR 0.040) vs LLM-Only (0.611 / 0.330)** → Pareto-optimal → dapat dikonfigurasi +
  future work optimasi.

---

Sumber data untuk verifikasi: `results/scripts/reproduce_tables.py` dan
`results/LATENCY_NARRATION_MANUSCRIPT.md` di repo GitHub
(https://github.com/bungagana/REPO-ARDA-SR-FRAMEWORK).