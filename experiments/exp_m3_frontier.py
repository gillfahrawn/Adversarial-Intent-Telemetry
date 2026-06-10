#!/usr/bin/env python3
"""
Experiment D4: M3 Operating-Point Frontier on PAN 2012

Sweeps banded MinHash (b, r) operating points at fixed L = 256 on the
author-disjoint PAN 2012 split (D3) and plots the resulting recall / FPR
frontier. Defends the (b=16, r=16) selection from validation/synthetic by
real-data evidence rather than by synthetic-distribution assumption.

Operating points: (8,32), (16,16), (32,8), (64,4), (128,2). All share
L = b*r = 256, so a single MinHash signature per conversation is reshaped
into different band layouts per point.

Protocol: federated pool = union of all training-positive signatures
(same protocol as exp_m3_author_split.py). The author-disjoint split,
PAN 2012 parsing, and MinHash primitives are imported from
exp_m3_author_split.py — split logic is not re-implemented.

Output: experiments/results/m3_frontier.json + .pdf + .png
Paper: Sec. 12 (M3), Maturity Matrix D4 entry.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
OUT = SCRIPT_DIR / "results"
OUT.mkdir(parents=True, exist_ok=True)

# Re-use D3 author-disjoint split + MinHash primitives.
sys.path.insert(0, str(SCRIPT_DIR))
from exp_m3_author_split import (  # noqa: E402
    SEED,
    L,
    VOCAB_SIZE,
    FPR_CEILING,
    DATA_XML,
    DATA_PRED,
    make_hash_funcs,
    conv_to_sig,
    build_band_index,
    query_band_index,
    load_predators,
    parse_pan12,
    author_disjoint_split,
)

OPERATING_POINTS = [(8, 32), (16, 16), (32, 8), (64, 4), (128, 2)]
RECOMMENDED = (16, 16)


def evaluate_at_bands(test_convs, test_sigs, pool_index, b, r):
    tp = fp = fn = tn = 0
    for (_, label, _), sig in zip(test_convs, test_sigs):
        if sig is None:
            fn += (label == 1)
            tn += (label == 0)
            continue
        hit = query_band_index(pool_index, sig, b, r)
        if label == 1:
            tp += hit;  fn += (not hit)
        else:
            fp += hit;  tn += (not hit)
    recall = tp / max(tp + fn, 1)
    fpr    = fp / max(fp + tn, 1)
    return float(recall), float(fpr), int(tp), int(fp), int(fn), int(tn)


def main() -> None:
    missing = [p for p in (DATA_XML, DATA_PRED) if not p.exists()]
    if missing:
        print("ERROR: Required PAN 2012 data files not found:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        print(
            "\nDownload from: "
            "https://pan.webis.de/clef12/pan12-web/sexual-predator-identification.html\n"
            "Place the training XML and predator list in data/pan12/train/",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Parsing PAN 2012 XML (this takes ~60 s on a single core) …")
    predators = load_predators(DATA_PRED)
    conversations = parse_pan12(DATA_XML, predators)
    print(f"  Loaded {len(conversations)} conversations, "
          f"{len(predators)} known predator IDs")

    print("\nBuilding author-disjoint split (D3 logic) …")
    train, test, train_authors, test_authors, all_authors = (
        author_disjoint_split(conversations, predators)
    )
    train_pos = [c for c in train if c[1] == 1]
    test_pos  = [c for c in test  if c[1] == 1]
    test_neg  = [c for c in test  if c[1] == 0]
    print(f"  train={len(train)} (pos={len(train_pos)})  "
          f"test={len(test)} (pos={len(test_pos)} neg={len(test_neg)})")
    print(f"  predator authors: {len(train_authors)} train / "
          f"{len(test_authors)} test (total {len(all_authors)})")

    # MinHash signatures at L=256 — computed once, reshaped per (b, r).
    a_hf, b_hf, p_hf = make_hash_funcs(l=L, vocab_size=VOCAB_SIZE, seed=SEED)

    print("\nComputing signatures (train positives + full test set) …")
    train_pos_sigs = [conv_to_sig(msgs, a_hf, b_hf, p_hf)
                     for _, _, msgs in train_pos]
    valid_pos_sigs = [s for s in train_pos_sigs if s is not None]
    test_sigs = [conv_to_sig(msgs, a_hf, b_hf, p_hf) for _, _, msgs in test]
    print(f"  valid train_pos sigs: {len(valid_pos_sigs)} / {len(train_pos)}")
    print(f"  test sigs (non-empty): "
          f"{sum(1 for s in test_sigs if s is not None)} / {len(test)}")

    # ── Sweep operating points ────────────────────────────────────────────────
    print(f"\nSweeping {len(OPERATING_POINTS)} operating points (L={L}) …")
    point_results = []
    for b, r in OPERATING_POINTS:
        assert b * r == L, f"b*r must equal L={L}; got b={b} r={r}"
        J_star = (1.0 / b) ** (1.0 / r)
        pool_index = build_band_index(valid_pos_sigs, b, r)
        recall, fpr, tp, fp, fn, tn = evaluate_at_bands(
            test, test_sigs, pool_index, b, r
        )
        rec = {
            "b": b,
            "r": r,
            "J_star": float(J_star),
            "federated_recall": recall,
            "federated_fpr": fpr,
            "below_ceiling": fpr <= FPR_CEILING,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        point_results.append(rec)
        flag = "below" if rec["below_ceiling"] else "ABOVE"
        print(f"  (b={b:>3}, r={r:>2})  J*={J_star:.3f}  "
              f"recall={recall:.4f}  FPR={fpr:.4f}  [{flag} α₀]")

    # ── Figure: operating-point frontier ──────────────────────────────────────
    fprs = np.array([p["federated_fpr"]    for p in point_results])
    recs = np.array([p["federated_recall"] for p in point_results])
    below = np.array([p["below_ceiling"]   for p in point_results])

    x_max = max(float(fprs.max()) * 1.25, FPR_CEILING * 1.6)
    y_max = max(float(recs.max()) * 1.25, 0.01)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))

    # Non-(16,16) operating points: circle markers.
    for p, x, y, ok in zip(point_results, fprs, recs, below):
        if (p["b"], p["r"]) == RECOMMENDED:
            continue
        ax.scatter([x], [y], s=60,
                   c=("#1f4e79" if ok else "#c0392b"),
                   marker="o", edgecolors="white", linewidths=0.8, zorder=3)

    # Recommended (16,16): diamond marker, slightly larger.
    for p, x, y, ok in zip(point_results, fprs, recs, below):
        if (p["b"], p["r"]) != RECOMMENDED:
            continue
        ax.scatter([x], [y], s=110,
                   c=("#1f4e79" if ok else "#c0392b"),
                   marker="D", edgecolors="white", linewidths=0.9, zorder=4)

    # Per-point labels on the main axes. Bottom-left cluster is handled
    # by the inset below; on the main plot those two labels are suppressed
    # to avoid overlapping the x-axis and each other.
    main_label_offsets = {
        (32, 8):  (8,    6),
        (64, 4):  (8,  -14),
        (128, 2): (-10,  8),
    }
    main_label_align = {(128, 2): "right"}
    for p, x, y in zip(point_results, fprs, recs):
        key = (p["b"], p["r"])
        if key not in main_label_offsets:
            continue
        dx, dy = main_label_offsets[key]
        ax.annotate(f"(b={p['b']}, r={p['r']})", (x, y),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, ha=main_label_align.get(key, "left"))

    # α₀ ceiling on main axes.
    ax.axvline(FPR_CEILING, color="#555555", lw=1.2, ls="--", zorder=2)
    ax.text(FPR_CEILING, y_max * 0.5, f" α₀ = {FPR_CEILING}",
            rotation=90, va="center", ha="left",
            fontsize=8, color="#555555",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.85, pad=1.5))

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("False positive rate (FPR)")
    ax.set_ylabel("Recall")
    ax.set_title(
        "D4: Operating-point frontier — M3 federated detection\n"
        "PAN 2012 (author-disjoint split, L = 256)"
    )
    ax.grid(alpha=0.25)

    # ── Inset: zoom into the sub-ceiling region so (8,32) and (16,16)
    # are distinguishable. The two points are otherwise glued at the
    # bottom-left of the main plot because recall ≪ 1.
    inset = ax.inset_axes([0.10, 0.66, 0.32, 0.30])
    inset_xmax = FPR_CEILING * 1.4
    inset_ymax = max(0.025, recs[below].max() * 1.6) if below.any() else 0.025
    for p, x, y, ok in zip(point_results, fprs, recs, below):
        if not ok:
            continue
        key = (p["b"], p["r"])
        marker = "D" if key == RECOMMENDED else "o"
        size = 70 if key == RECOMMENDED else 50
        inset.scatter([x], [y], s=size, c="#1f4e79",
                      marker=marker, edgecolors="white",
                      linewidths=0.8, zorder=3)
        suffix = "\n(recommended)" if key == RECOMMENDED else ""
        # Offset labels above the points; (b=8, r=32) shifted lower
        # so it doesn't collide with the (b=16, r=16) annotation.
        dy = 8 if key == RECOMMENDED else -14
        inset.annotate(f"(b={p['b']}, r={p['r']}){suffix}", (x, y),
                       xytext=(6, dy), textcoords="offset points",
                       fontsize=7)
    inset.axvline(FPR_CEILING, color="#555555", lw=1.0, ls="--", zorder=2)
    inset.text(FPR_CEILING, inset_ymax * 0.92, f" α₀ = {FPR_CEILING}",
               fontsize=6.5, color="#555555", va="top", ha="left")
    inset.set_xlim(0, inset_xmax)
    inset.set_ylim(0, inset_ymax)
    inset.set_title("zoom: sub-ceiling region", fontsize=8, pad=2)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.25)
    ax.indicate_inset_zoom(inset, edgecolor="#888888", alpha=0.5)

    fig.tight_layout()
    fig.savefig(OUT / "m3_frontier.pdf", bbox_inches="tight")
    fig.savefig(OUT / "m3_frontier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── JSON output ───────────────────────────────────────────────────────────
    recommended_rec = next(
        p for p in point_results if (p["b"], p["r"]) == RECOMMENDED
    )
    below_ceiling = [p for p in point_results if p["below_ceiling"]]
    best_compliant = (
        max(below_ceiling, key=lambda p: p["federated_recall"])
        if below_ceiling else None
    )

    output = {
        "experiment": "m3_frontier",
        "split_type": "author_disjoint",
        "depends_on": "exp_m3_author_split.py (D3)",
        "dataset": (
            "PAN 2012 Sexual Predator Identification, training XML, "
            "author-disjoint 80/20 split (D3)"
        ),
        "L": L,
        "vocab_size": VOCAB_SIZE,
        "fpr_ceiling": FPR_CEILING,
        "recommended_point": {"b": RECOMMENDED[0], "r": RECOMMENDED[1]},
        "n_total_conversations": len(conversations),
        "n_train": len(train),
        "n_train_pos": len(train_pos),
        "n_valid_train_pos_sigs": len(valid_pos_sigs),
        "n_test": len(test),
        "n_test_pos": len(test_pos),
        "n_test_neg": len(test_neg),
        "n_predator_authors_train": len(train_authors),
        "n_predator_authors_test":  len(test_authors),
        "operating_points": point_results,
        "recommended_result": recommended_rec,
        "best_compliant_point": best_compliant,
        "figures": [
            str(OUT / "m3_frontier.pdf"),
            str(OUT / "m3_frontier.png"),
        ],
        "caveats": [
            "All operating points share L = b*r = 256, so a single MinHash "
            "signature per conversation is reshaped into different (b, r) "
            "band layouts for matching.",
            "FPR and recall are reported on the federated pool (union of all "
            "training-positive signatures) against the test set; the per-shard "
            "single-provider baseline used in the M3 lift experiment is not "
            "repeated here.",
            "Author-disjoint split: no predator author appears in both train "
            "and test (see exp_m3_author_split.py).",
            "The (b=16, r=16) selection is empirically defended on PAN 2012 if "
            "it lies on the Pareto frontier of the points below the α₀ ceiling.",
            "Results are on PAN 2012 (human grooming, 2012). Calibration of "
            "operating-point selection to A2 agentic trajectories requires "
            "either synthetic A2 trajectory data or pilot deployment data.",
        ],
    }

    out_path = OUT / "m3_frontier.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== D4: M3 Operating-Point Frontier (PAN 2012, author-disjoint) ===\n")
    ceil_str = f"vs. α₀={FPR_CEILING}"
    print(f"  {'(b, r)':>10}  {'J*':>6}  {'recall':>8}  "
          f"{'FPR':>8}  {ceil_str:>12}")
    print("  " + "-" * 54)
    for p in point_results:
        flag = "below" if p["below_ceiling"] else "ABOVE"
        print(f"  ({p['b']:>3}, {p['r']:>2})  {p['J_star']:.3f}  "
              f"{p['federated_recall']:>8.4f}  {p['federated_fpr']:>8.4f}  "
              f"{flag:>12}")
    print()
    if best_compliant is not None:
        bp = best_compliant
        print(f"  Best recall under α₀: (b={bp['b']}, r={bp['r']})  "
              f"recall={bp['federated_recall']:.4f}  "
              f"FPR={bp['federated_fpr']:.4f}")
    rec = recommended_rec
    status = "compliant" if rec["below_ceiling"] else "violates ceiling"
    print(f"  Recommended (16, 16): recall={rec['federated_recall']:.4f}  "
          f"FPR={rec['federated_fpr']:.4f}  ({status})")
    print(f"\nResults: {out_path}")
    print(f"Figure:  {OUT / 'm3_frontier.pdf'} (+ .png)")


if __name__ == "__main__":
    main()
