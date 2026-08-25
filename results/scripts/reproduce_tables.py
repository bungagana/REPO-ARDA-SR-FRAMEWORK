#!/usr/bin/env python3
"""Reproduce the paper's comparison tables (6, 8, 9) from the raw experiment data.

Data expectations (paths relative to this script or an override via --data):
  data/*.json                   main-domain per-method results (Table 6)
    each item: {method, rel, faith, cov, is_refusal, latency_s, hit_at_k, ...}
  data/arda_sr_*_pakdwi_summary.json   cross-backbone per-method summaries (Table 8)
  data/summary_<domain>_*       per-domain per-method summaries (Table 9)

Outputs (--out):
  results/tables/table6.csv, table8.csv, table9.csv   (+ .md copies)

Run: python reproduce_tables.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "tables")


def plot_figure6(t6, fig_path=None, fig_path_pdf=None):
    """Reproduce Figure 6 (Relevance vs computational cost / latency).

    Emphasises ARDA-SR and draws the Pareto-optimal frontier (higher relevance at
    lower latency is better), matching the manuscript trade-off figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print("matplotlib not available; skip Figure 6:", exc)
        return

    items = [(m, t6[m]["Rel"], t6[m]["Lat"]) for m in t6 if t6[m]["Rel"] is not None]
    methods = [i[0] for i in items]
    rel = np.array([i[1] for i in items])
    lat = np.array([i[2] for i in items])

    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    # scatter all methods
    ax.scatter(lat, rel, s=70, c="#4c72b0", alpha=0.9, edgecolors="k", linewidths=0.6, label="Method")
    # ARDA-SR highlighted
    arda = np.array([m for m in methods if m in ("arda_pmr", "arda_sr", "ARDA-SR")])
    for m, x, y in zip(methods, lat, rel):
        is_arda = m in ("arda_pmr", "arda_sr", "ARDA-SR")
        if is_arda:
            ax.scatter([x], [y], s=170, c="#dd8452", edgecolors="k", linewidths=1.2, zorder=5,
                       marker="*", label="ARDA-SR")
        ax.annotate(m.replace("_", " ").title(), (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=7.5, weight="bold" if is_arda else "normal")

    # Pareto frontier: keep points that are not dominated in (latency low, rel high)
    pts = [(x, y) for x, y in zip(lat, rel)]
    pac = []
    for i, (xi, yi) in enumerate(pts):
        dominated = any((xj <= xi and yj >= yi) and (xj < xi or yj > yi) for j, (xj, yj) in enumerate(pts))
        if not dominated:
            pac.append((xi, yi))
    pac.sort()
    if len(pac) >= 2:
        px, py = zip(*pac)
        ax.plot(px, py, "--", color="#55a868", lw=1.5, label="Pareto frontier")

    ax.set_xlabel("Computational cost — latency (s)")
    ax.set_ylabel("Answer relevance (0–1)")
    ax.set_title("Trade-off between Relevance and computational cost\nacross ARDA-SR and baseline methods")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(left=0.8, right=max(lat) * 1.15)
    ax.set_ylim(0.55, 0.92)
    fig.tight_layout()

    fig_path = fig_path or os.path.join(HERE, "..", "figures", "Figure6_relevance_vs_cost.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    if fig_path_pdf:
        fig.savefig(fig_path_pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fig_path)
    plt.close(fig)
    print("wrote", fig_path)


# ---------------------------------------------------------------- helpers -----
def mean_is_refusal(results, answerable_only=True):
    an = [r for r in results if not r.get("should_be_answerable") is False]
    if not an:
        return None
    return float(np.mean([r["is_refusal"] for r in an]))


def metric_mean(results, field):
    vals = [r[field] for r in results if r.get(field) is not None]
    return float(np.mean(vals)) if vals else None


def full_metrics(results):
    return {
        "Rel": metric_mean(results, "rel"),
        "Faith": metric_mean(results, "faith"),
        "Cov": metric_mean(results, "cov"),
        "FRR": mean_is_refusal(results),
        "Lat": metric_mean(results, "latency_s"),
        "Hit@5": metric_mean(results, "hit_at_k"),
    }


def build_table6(data_dir):
    """Table 6: per-method answer/behaviour metrics on the 1,000 main queries."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*_results.json"))):
        base = os.path.basename(path)
        if base in ("arda_sr_full_results.json",):
            continue
        d = json.load(open(path))
        if not d:
            continue
        method = d[0].get("method", base.replace("_results.json", "").replace("_", " "))
        m = full_metrics(d)
        rows[method] = m
    return rows


def build_table9(data_dir):
    """Table 9: cross-domain per-method summaries.
    Primary source = summary_<domain>_fixed.json (rel/faith/cov/frr/latency, matches
    the manuscript). Falls back to summary_<domain>.json + summary_<domain>_raw.json
    if the _fixed file is absent.
    """
    doms = {}
    # prefer _fixed files
    for path in sorted(glob.glob(os.path.join(data_dir, "summary_*_fixed.json"))):
        ds = os.path.basename(path).replace("summary_", "").replace("_fixed.json", "")
        doms[ds] = {"_main": json.load(open(path)), "_raw": {}}
    # backfill with _raw for any domain missing it (for fields not in _fixed)
    for path in sorted(glob.glob(os.path.join(data_dir, "summary_*_raw.json"))):
        ds = os.path.basename(path).replace("summary_", "").replace("_raw.json", "")
        doms.setdefault(ds, {})["_raw"] = json.load(open(path))
    return doms


def table9_to_csv(doms, name="table9.csv"):
    """Rows = (domain, method); cols = Rel, Faith, Cov, FRR, Lat."""
    import csv
    rows = []
    for ds, store in sorted(doms.items()):
        methods = sorted(set(store.get("_main", {})) | set(store.get("_raw", {})))
        for m in methods:
            mm = store.get("_main", {}).get(m, {})
            rr = store.get("_raw", {}).get(m, {})
            def g(a, k, section):
                return a.get(k) if isinstance(a, dict) else None
            rows.append({
                "domain": ds, "method": m,
                "Rel": g(mm, "rel", "_main"), "Faith": g(mm, "faith", "_main"),
                "Cov": g(mm, "cov", "_main"),
                "FRR": g(mm, "frr", "_main") if g(mm, "frr", "_main") is not None else g(rr, "frr", "_raw"),
                "Lat": g(mm, "latency_s", "_main") or g(rr, "latency_s", "_raw"),
            })
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "method", "Rel", "Faith", "Cov", "FRR", "Lat"])
        for r in rows:
            w.writerow([r["domain"], r["method"], r["Rel"], r["Faith"], r["Cov"], r["FRR"], r["Lat"]])
    print("wrote", path)
    return rows


def table8_to_csv(data_dir, name="table8.csv"):
    """Table 8: cross-backbone summary (arda_sr_*_pakdwi_summary.json)."""
    import csv
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, "arda_sr_*_pakdwi_summary.json"))):
        d = json.load(open(path))
        d["backbone_file"] = os.path.basename(path)
        rows.append(d)
    path_out = os.path.join(OUT, name)
    with open(path_out, "w", newline="") as f:
        w = csv.writer(f)
        # single header (no duplicated 'file')
        keys = ["backbone_file", "model", "n_answerable", "n_unanswerable", "frr",
                "n_correctly_refused", "arr_unanswerable", "far_unanswerable",
                "hit_at_5", "latency_mean_s"]
        w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k, "") for k in keys])
    print("wrote", path_out)
    return rows


def save_csv(rows, name, keys):
    rows = sorted(rows.items(), key=lambda kv: kv[0])
    import csv
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + keys)
        for method, m in rows:
            if isinstance(m, dict):
                w.writerow([method] + [f"{m.get(k):.3f}" if isinstance(m.get(k), float) else m.get(k) for k in keys])
            else:
                w.writerow([method, m])
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t6 = build_table6(args.data)
    keys = ["Rel", "Faith", "Cov", "FRR", "Lat"]
    save_csv(t6, "table6.csv", keys)

    t8 = table8_to_csv(args.data, "table8.csv")
    t9 = table9_to_csv(build_table9(args.data), "table9.csv")

    # print Table 6
    print("\n=== Table 6 (reproduced) ===")
    print(f"{'method':16s} {'Rel':>6s} {'Faith':>7s} {'Cov':>7s} {'FRR':>7s} {'Lat':>6s}")
    for m, r in sorted(t6.items()):
        print(f"{m:16s} {r['Rel']:.3f} {r['Faith']:.3f} {r['Cov']:.3f} {r['FRR']:.3f} {r['Lat']:.2f}")

    print("\n=== Table 9 (reproduced) — Rel/Faith/Cov/FRR/Lat per domain ===")
    print(f"{'domain':16s} {'method':14s} {'Rel':>6s} {'Faith':>7s} {'Cov':>7s} {'FRR':>7s} {'Lat':>6s}")
    for r in t9:
        print(f"{r['domain']:16s} {r['method']:14s} {str(r['Rel'])[:6]:>6s} {str(r['Faith'])[:7]:>7s} {str(r['Cov'])[:7]:>7s} {str(r['FRR'])[:7]:>7s} {str(r['Lat'])[:6]:>6s}")

    # Repro Figure 6 (Relevance vs latency)
    if t6:
        plot_figure6(t6, fig_path_pdf=os.path.join(HERE, "..", "figures", "Figure6_relevance_vs_cost.pdf"))


if __name__ == "__main__":
    main()