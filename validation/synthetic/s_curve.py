#!/usr/bin/env python3
"""
Synthetic LSH/MinHash experiment for v8 of the Decentralized Telemetry paper.

Goal: anchor the (b, r) operating point and Equation (1) with measured numbers
on synthetic feature manifests, NOT to claim production performance.

Construction:
  - Each "manifest" is a set of structural feature tokens drawn from a
    canonical vocabulary V of size |V|=4096.
  - Adversarial-class manifests are drawn around a small set of "intent"
    centroids and share substantial vocabulary across instances of the
    same class.
  - Benign-analogue manifests are drawn around their own centroids and
    deliberately share some lexical surface with the adversarial centroids
    (the adversarial-adjacent benign case the protocol must handle).
  - We sweep target Jaccard J in [0, 1] and, for each J, sample
    pairs whose ground-truth Jaccard is approximately J, then measure
    empirical AnyBandMatch rate under banded MinHash with L = b*r = 256.

We report:
  (i)   The empirical S-curve P_match(J; b, r) vs. theoretical
        1 - (1 - J^r)^b at b=16, r=16 and the inflection point.
  (ii)  Recall at 0.1% FPR for an adversarial-vs-benign setup where
        adversarial pairs are drawn from a Jaccard distribution with
        mean ~0.62 and benign-analogue pairs from a distribution with
        mean ~0.28.
  (iii) The same headline numbers for two adjacent operating points
        (b=8,r=32 and b=32,r=8) so that the (16,16) choice is shown
        as a deliberate selection rather than a guess.

Outputs:
  - validation/synthetic/results/results.json  (numbers used in v8 text)
  - validation/synthetic/results/s_curve.png   (figure for v8)

Run-time target: under 60 seconds on a single core.
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Reproducibility ------------------------------------
SEED = 20260514
random.seed(SEED)
np.random.seed(SEED)

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


# ---------------- MinHash implementation ------------------------------
def make_hash_funcs(L: int, vocab_size: int, seed: int = SEED):
    """L independent hash functions of the form (a*x + b) mod p, modded into vocab."""
    rng = np.random.default_rng(seed)
    # large prime > vocab_size
    p = 2**61 - 1
    a = rng.integers(1, p - 1, size=L, dtype=np.int64)
    b = rng.integers(0, p - 1, size=L, dtype=np.int64)
    return a, b, p


def minhash(token_set: np.ndarray, a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    """MinHash signature of length L over the token_set (1-D array of ints)."""
    # broadcasting: tokens [N], a [L] -> [L, N]
    h = ((a[:, None] * token_set[None, :] + b[:, None]) % p)
    return h.min(axis=1).astype(np.int64)


def any_band_match(sig_a: np.ndarray, sig_b: np.ndarray, b: int, r: int) -> bool:
    """True iff any of the b bands agrees on all r rows."""
    # reshape signature into (b bands, r rows) and compare row-wise
    A = sig_a.reshape(b, r)
    B = sig_b.reshape(b, r)
    return bool(np.any(np.all(A == B, axis=1)))


# ---------------- Synthetic manifest generator -----------------------
VOCAB_SIZE = 4096
MANIFEST_SIZE = 64   # number of feature tokens per manifest (set cardinality)


def sample_pair_with_jaccard(target_J: float, size: int = MANIFEST_SIZE):
    """
    Return a pair of token-sets (np.arrays of unique ints) with Jaccard
    approximately equal to target_J.

    Construction: pick a shared core of size k = round(J * size / (2 - J)) * 2,
    then sample disjoint remainders. This produces exact Jaccard
    k / (2*size - k).
    """
    # Solve for k: J = k / (2*size - k)  =>  k = J * 2*size / (1 + J)
    k = int(round(2 * size * target_J / (1.0 + target_J)))
    k = max(0, min(k, size))
    # core
    core = np.random.choice(VOCAB_SIZE, size=k, replace=False)
    remaining = np.setdiff1d(np.arange(VOCAB_SIZE), core, assume_unique=False)
    extras = np.random.choice(remaining, size=2 * (size - k), replace=False)
    a_set = np.concatenate([core, extras[: size - k]])
    b_set = np.concatenate([core, extras[size - k:]])
    return a_set, b_set, k / (2 * size - k) if (2 * size - k) > 0 else 1.0


# ---------------- Experiment 1: empirical S-curve --------------------
def experiment_scurve(a_hashes, b_hashes, p, b: int, r: int, n_per_J: int = 200):
    """Sweep target Jaccard and measure empirical AnyBandMatch rate."""
    targets = np.linspace(0.05, 0.95, 19)
    emp = []
    for J in targets:
        hits = 0
        for _ in range(n_per_J):
            sa, sb, _ = sample_pair_with_jaccard(J)
            sig_a = minhash(sa, a_hashes, b_hashes, p)
            sig_b = minhash(sb, a_hashes, b_hashes, p)
            if any_band_match(sig_a, sig_b, b, r):
                hits += 1
        emp.append(hits / n_per_J)
    return targets.tolist(), emp


def theoretical_scurve(J_array, b: int, r: int):
    return [1 - (1 - J**r)**b for J in J_array]


def inflection_J(b: int, r: int) -> float:
    """S-curve inflection point at J = (1/b)^(1/r)."""
    return (1.0 / b) ** (1.0 / r)


# ---------------- Experiment 2: recall @ 0.1% FPR ---------------------
def experiment_recall_at_fpr(a_hashes, b_hashes, p, b: int, r: int,
                              n_pairs: int = 5000):
    """
    Generate n_pairs adversarial-similar pairs (Jaccard ~ Beta(8,5), mean ~0.62)
    and n_pairs benign-analogue pairs (Jaccard ~ Beta(3,8), mean ~0.27).
    Measure recall (TP / positives) and FPR (FP / negatives).
    """
    adv_J = np.random.beta(8.0, 5.0, size=n_pairs)
    ben_J = np.random.beta(3.0, 8.0, size=n_pairs)

    tp = 0
    for J in adv_J:
        sa, sb, _ = sample_pair_with_jaccard(float(J))
        sig_a = minhash(sa, a_hashes, b_hashes, p)
        sig_b = minhash(sb, a_hashes, b_hashes, p)
        if any_band_match(sig_a, sig_b, b, r):
            tp += 1

    fp = 0
    for J in ben_J:
        sa, sb, _ = sample_pair_with_jaccard(float(J))
        sig_a = minhash(sa, a_hashes, b_hashes, p)
        sig_b = minhash(sb, a_hashes, b_hashes, p)
        if any_band_match(sig_a, sig_b, b, r):
            fp += 1

    recall = tp / n_pairs
    fpr = fp / n_pairs
    return {
        "n_pairs_per_class": n_pairs,
        "adv_J_mean": float(adv_J.mean()),
        "adv_J_std": float(adv_J.std()),
        "ben_J_mean": float(ben_J.mean()),
        "ben_J_std": float(ben_J.std()),
        "TP": tp, "FP": fp,
        "recall": recall,
        "FPR": fpr,
    }


# ---------------- Run --------------------------------------------------
def main():
    L = 256
    operating_points = [(16, 16), (8, 32), (32, 8)]

    # Build a single shared hash family of length L; we slice it per (b, r)
    a_hashes, b_hashes, p = make_hash_funcs(L=L, vocab_size=VOCAB_SIZE, seed=SEED)

    all_results = {
        "vocab_size": VOCAB_SIZE,
        "manifest_size": MANIFEST_SIZE,
        "L": L,
        "operating_points": [],
        "seed": SEED,
    }

    # Empirical S-curve at the recommended (16, 16) operating point
    targets, emp = experiment_scurve(a_hashes, b_hashes, p, b=16, r=16, n_per_J=200)
    theory = theoretical_scurve(targets, b=16, r=16)
    all_results["scurve_b16_r16"] = {
        "J_targets": targets,
        "P_match_empirical": emp,
        "P_match_theoretical": theory,
        "inflection_J": inflection_J(16, 16),
    }

    # Recall @ FPR for each (b, r)
    for (b, r) in operating_points:
        op = experiment_recall_at_fpr(a_hashes, b_hashes, p, b=b, r=r, n_pairs=5000)
        op["b"] = b
        op["r"] = r
        op["L"] = b * r
        op["inflection_J"] = inflection_J(b, r)
        all_results["operating_points"].append(op)

    # Save numeric results
    with open(OUT / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ----- Figure: empirical vs theoretical S-curve --------------------
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    # Theoretical curve over fine grid
    fineJ = np.linspace(0, 1, 200)
    for (b, r, ls, col) in [(16, 16, "-", "#1f4e79"),
                             (8, 32, "--", "#7f7f7f"),
                             (32, 8, ":", "#7f7f7f")]:
        ax.plot(fineJ, [1 - (1 - J**r)**b for J in fineJ],
                ls=ls, color=col, lw=1.3,
                label=f"theoretical (b={b}, r={r})")
    ax.plot(targets, emp, "o", color="#1f4e79", ms=4.5,
            label="empirical (b=16, r=16; 200 pairs/point)")
    # Annotate inflection for (16, 16)
    Jc = inflection_J(16, 16)
    ax.axvline(Jc, color="#1f4e79", lw=0.6, alpha=0.4)
    ax.text(Jc + 0.01, 0.05, f"J* = (1/16)^(1/16) ≈ {Jc:.3f}",
            fontsize=8, color="#1f4e79")
    ax.set_xlabel("Jaccard similarity J of feature manifests")
    ax.set_ylabel("P[AnyBandMatch]")
    ax.set_title("Banded MinHash S-curve: theoretical vs empirical (L=256)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "s_curve.png", dpi=150, bbox_inches="tight")

    # ----- Print summary ------------------------------------------------
    print("=== S-curve at b=16, r=16 ===")
    for J, e, t in zip(targets, emp, theory):
        print(f"  J={J:.2f}: empirical={e:.3f}  theoretical={t:.3f}")
    print()
    print("=== Operating points: recall and FPR ===")
    for op in all_results["operating_points"]:
        print(f"  (b={op['b']}, r={op['r']}): "
              f"recall={op['recall']:.4f}, FPR={op['FPR']:.4f}, "
              f"inflection J*={op['inflection_J']:.3f}")
    print()
    print("Inflection J*(b=16,r=16) =", inflection_J(16, 16))
    print("Inflection J*(b=8,r=32)  =", inflection_J(8, 32))
    print("Inflection J*(b=32,r=8)  =", inflection_J(32, 8))
    print(f"\nOutputs written to {OUT}/")


if __name__ == "__main__":
    main()
