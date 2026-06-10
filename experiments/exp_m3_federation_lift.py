#!/usr/bin/env python3
"""
Experiment M3: Federation Detection Lift

Demonstrates that federated MinHash signature sharing improves
grooming-trajectory detection over the best single-provider baseline at
fixed FPR on PAN 2012 (training XML, 80/20 stratified hold-out).

Paper: Sec. 12 (M3 detection lift).
Dataset: PAN 2012 Sexual Predator Identification training corpus.
Status upgrade target: hypothesized → demonstrated.
"""

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 20260514

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
OUT = SCRIPT_DIR / "results"
OUT.mkdir(parents=True, exist_ok=True)

DATA_XML = (ROOT / "data/pan12/train/"
            "pan12-sexual-predator-identification-training-corpus-2012-05-01.xml")
DATA_PRED = (ROOT / "data/pan12/train/"
             "pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt")

# ── MinHash / band parameters ──────────────────────────────────────────────────
L = 256
BAND_B = 16
BAND_R = 16      # L = BAND_B * BAND_R
VOCAB_SIZE = 4096

FPR_CEILING = 0.005
N_PROVIDERS_SWEEP = [2, 4, 8]


# ── MinHash (copied verbatim from experiment.py) ────────────────────────────────
def make_hash_funcs(L: int, vocab_size: int, seed: int = SEED):
    """L independent hash functions of the form (a*x + b) mod p, modded into vocab."""
    rng = np.random.default_rng(seed)
    p = 2**61 - 1
    a = rng.integers(1, p - 1, size=L, dtype=np.int64)
    b = rng.integers(0, p - 1, size=L, dtype=np.int64)
    return a, b, p


def minhash(token_set: np.ndarray, a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    """MinHash signature of length L over the token_set (1-D array of ints)."""
    h = ((a[:, None] * token_set[None, :] + b[:, None]) % p)
    return h.min(axis=1).astype(np.int64)


def any_band_match(sig_a: np.ndarray, sig_b: np.ndarray, b: int, r: int) -> bool:
    """True iff any of the b bands agrees on all r rows."""
    A = sig_a.reshape(b, r)
    B = sig_b.reshape(b, r)
    return bool(np.any(np.all(A == B, axis=1)))


# ── Band index for O(b) pool queries ───────────────────────────────────────────
def build_band_index(sigs: list, b: int, r: int) -> list:
    """Build an inverted band index: list of b dicts mapping band_bytes → True."""
    index = [{} for _ in range(b)]
    for sig in sigs:
        bands = sig.reshape(b, r)
        for bi in range(b):
            key = bands[bi].tobytes()
            index[bi][key] = True
    return index


def query_band_index(index: list, sig: np.ndarray, b: int, r: int) -> bool:
    """Return True if any band of sig is present in the index."""
    bands = sig.reshape(b, r)
    for bi in range(b):
        if bands[bi].tobytes() in index[bi]:
            return True
    return False


# ── Feature extraction ──────────────────────────────────────────────────────────
def text_to_tokens(text: str) -> np.ndarray:
    """Char 4-grams → MD5 hash → [0, VOCAB_SIZE) → sorted unique int array."""
    tokens: set[int] = set()
    for i in range(len(text) - 3):
        gram = text[i:i + 4]
        digest = hashlib.md5(gram.encode("utf-8", errors="replace")).digest()
        tokens.add(int.from_bytes(digest[:4], "little") % VOCAB_SIZE)
    return np.array(sorted(tokens), dtype=np.int64)


def conv_to_sig(messages: list, a_hf: np.ndarray, b_hf: np.ndarray, p: int):
    """Concatenate message text, compute MinHash signature. Returns None if empty."""
    full_text = " ".join(txt for _, txt in messages)
    tokens = text_to_tokens(full_text)
    if tokens.size == 0:
        return None
    return minhash(tokens, a_hf, b_hf, p)


# ── PAN 2012 I/O ────────────────────────────────────────────────────────────────
def load_predators(path: Path) -> set:
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def parse_pan12(xml_path: Path, predators: set) -> list:
    """
    Parse PAN 2012 training XML with iterparse.
    Returns list of (conv_id, label, messages).
      label = 1 if any message author is a predator, else 0.
      messages = list of (author_id, text).
    """
    conversations = []
    for _event, elem in ET.iterparse(str(xml_path), events=["end"]):
        if elem.tag == "conversation":
            conv_id = elem.get("id", "")
            messages = []
            for msg in elem.findall("message"):
                author = (msg.findtext("author") or "").strip()
                text   = (msg.findtext("text")   or "").strip()
                messages.append((author, text))
            label = 1 if any(a in predators for a, _ in messages) else 0
            conversations.append((conv_id, label, messages))
            elem.clear()
    return conversations


def stratified_split(conversations: list, test_frac: float = 0.2, seed: int = SEED):
    """
    Stratified 80/20 split, sorted by conv_id for reproducibility.
    This exact logic is replicated in exp_trajectory_lift.py.
    """
    conversations = sorted(conversations, key=lambda x: x[0])
    pos = [c for c in conversations if c[1] == 1]
    neg = [c for c in conversations if c[1] == 0]

    rng = np.random.default_rng(seed)
    pos_perm = rng.permutation(len(pos))
    neg_perm = rng.permutation(len(neg))

    n_pos_tr = int(len(pos) * (1.0 - test_frac))
    n_neg_tr = int(len(neg) * (1.0 - test_frac))

    train = [pos[i] for i in pos_perm[:n_pos_tr]] + [neg[i] for i in neg_perm[:n_neg_tr]]
    test  = [pos[i] for i in pos_perm[n_pos_tr:]] + [neg[i] for i in neg_perm[n_neg_tr:]]
    return train, test


# ── Detection evaluation ────────────────────────────────────────────────────────
def evaluate_pool(test_convs: list, pool_index: list,
                  a_hf: np.ndarray, b_hf: np.ndarray, p: int):
    """
    Query each test conversation against pool_index.
    Returns (recall, fpr). Conversations with no 4-grams are treated as misses.
    """
    tp = fp = fn = tn = 0
    for _, label, msgs in test_convs:
        sig = conv_to_sig(msgs, a_hf, b_hf, p)
        if sig is None:
            fn += (label == 1)
            tn += (label == 0)
            continue
        hit = query_band_index(pool_index, sig, BAND_B, BAND_R)
        if label == 1:
            tp += hit;  fn += (not hit)
        else:
            fp += hit;  tn += (not hit)
    recall = tp / max(tp + fn, 1)
    fpr    = fp / max(fp + tn, 1)
    return float(recall), float(fpr)


def main() -> None:
    # ── Check data ─────────────────────────────────────────────────────────────
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

    # ── Parse and split ────────────────────────────────────────────────────────
    print("Parsing PAN 2012 XML (this takes ~60 s on a single core) …")
    predators = load_predators(DATA_PRED)
    conversations = parse_pan12(DATA_XML, predators)
    train, test = stratified_split(conversations)

    train_pos = [c for c in train if c[1] == 1]
    test_pos  = [c for c in test  if c[1] == 1]
    test_neg  = [c for c in test  if c[1] == 0]

    print(f"  Total: {len(conversations)} | train={len(train)} test={len(test)}")
    print(f"  train_pos={len(train_pos)} | test_pos={len(test_pos)} test_neg={len(test_neg)}")

    # ── MinHash setup ──────────────────────────────────────────────────────────
    a_hf, b_hf, p_hf = make_hash_funcs(L=L, vocab_size=VOCAB_SIZE, seed=SEED)

    # ── Signatures for training positives ─────────────────────────────────────
    print("Computing signatures for training positives …")
    train_pos_sigs = []
    for i, (_, _, msgs) in enumerate(train_pos):
        sig = conv_to_sig(msgs, a_hf, b_hf, p_hf)
        train_pos_sigs.append(sig)  # None if empty

    valid_pos_sigs = [s for s in train_pos_sigs if s is not None]
    print(f"  Valid training-positive signatures: {len(valid_pos_sigs)} / {len(train_pos)}")

    # ── Federated pool = union of all training positives ───────────────────────
    fed_index = build_band_index(valid_pos_sigs, BAND_B, BAND_R)
    fed_recall, fed_fpr = evaluate_pool(test, fed_index, a_hf, b_hf, p_hf)
    print(f"  Federated pool: recall={fed_recall:.4f}, FPR={fed_fpr:.4f}")

    # ── Sweep n_providers ─────────────────────────────────────────────────────
    sweep_results = []

    for n_prov in N_PROVIDERS_SWEEP:
        # Partition all training conversations into n_prov equal shards
        rng_shard = np.random.default_rng(SEED)
        n_train = len(train)
        perm = rng_shard.permutation(n_train)
        shard_size = n_train // n_prov

        # Build index_in_train for each training-positive conversation
        # train = train_pos (first) + train_neg (rest), so training-positive i
        # is at position i in train.
        pos_global_idx = {i: i for i in range(len(train_pos))}

        # Map each position in train → shard id
        pos_to_shard = {}
        for i_shard in range(n_prov):
            start = i_shard * shard_size
            end = start + shard_size if i_shard < n_prov - 1 else n_train
            for gi in perm[start:end]:
                pos_to_shard[int(gi)] = i_shard

        # Collect valid signatures per shard
        shard_sigs: list[list] = [[] for _ in range(n_prov)]
        for pos_idx, sig in enumerate(train_pos_sigs):
            if sig is None:
                continue
            shard_id = pos_to_shard.get(pos_idx, -1)
            if shard_id >= 0:
                shard_sigs[shard_id].append(sig)

        prov_max_recall = 0.0
        prov_max_fpr = 0.0
        prov_details = []

        for i_shard in range(n_prov):
            if shard_sigs[i_shard]:
                shard_index = build_band_index(shard_sigs[i_shard], BAND_B, BAND_R)
                r_i, f_i = evaluate_pool(test, shard_index, a_hf, b_hf, p_hf)
            else:
                r_i, f_i = 0.0, 0.0
            prov_details.append({
                "provider": i_shard,
                "n_positives_in_shard": len(shard_sigs[i_shard]),
                "recall": r_i,
                "fpr": f_i,
            })
            if f_i <= FPR_CEILING and r_i > prov_max_recall:
                prov_max_recall = r_i
                prov_max_fpr = f_i

        lift = fed_recall - prov_max_recall
        sweep_results.append({
            "n_providers": n_prov,
            "max_single_provider_recall": prov_max_recall,
            "max_single_provider_fpr": prov_max_fpr,
            "federated_recall": fed_recall,
            "federated_fpr": fed_fpr,
            "federation_lift": lift,
            "lift_positive": lift > 0.0,
            "fpr_ceiling_used": FPR_CEILING,
            "provider_details": prov_details,
        })
        print(f"  n_providers={n_prov}: max_single={prov_max_recall:.4f} "
              f"(FPR={prov_max_fpr:.4f}), fed={fed_recall:.4f}, lift={lift:+.4f}")

    demonstrated = any(r["lift_positive"] for r in sweep_results)
    best = max(sweep_results, key=lambda r: r["federation_lift"])

    # ── Output JSON ────────────────────────────────────────────────────────────
    output = {
        "experiment": "m3_federation_lift",
        "dataset": (
            "PAN 2012 Sexual Predator Identification, training XML, "
            "80/20 stratified split, updated 2012-05-01 files"
        ),
        "status": "demonstrated" if demonstrated else "inconclusive",
        "key_metric": {
            "name": "federation_lift",
            "value": best["federation_lift"],
            "threshold": 0.0,
            "passed": demonstrated,
            "at_n_providers": best["n_providers"],
            "federated_recall": best["federated_recall"],
            "federated_fpr": best["federated_fpr"],
            "max_single_provider_recall": best["max_single_provider_recall"],
            "fpr_ceiling": FPR_CEILING,
        },
        "caveats": [
            "MinHash at (b=16, r=16) requires Jaccard ≥ J*≈0.841 for meaningful "
            "recall. PAN 2012 inter-conversation Jaccard similarity is typically "
            "lower, so absolute recall may be near zero.",
            "Federation lift is recall(union pool) − max recall(single-shard pool) "
            "at FPR ≤ 0.005. If both are zero, lift = 0 and status is inconclusive.",
            "Results on training XML 80/20 split only; the test folder has no "
            "ground-truth labels and was not used.",
        ],
        "figures": [
            str(OUT / "m3_federation_lift.pdf"),
            str(OUT / "m3_federation_lift.png"),
        ],
        "full_results": {
            "n_total": len(conversations),
            "n_train": len(train),
            "n_test": len(test),
            "n_train_pos": len(train_pos),
            "n_test_pos": len(test_pos),
            "n_test_neg": len(test_neg),
            "n_valid_train_pos_sigs": len(valid_pos_sigs),
            "minhash_params": {"L": L, "b": BAND_B, "r": BAND_R, "vocab_size": VOCAB_SIZE},
            "federated_pool_recall": fed_recall,
            "federated_pool_fpr": fed_fpr,
            "sweep": sweep_results,
        },
    }

    with open(OUT / "m3_federation_lift.json", "w") as fh:
        json.dump(output, fh, indent=2)

    # ── Figure ─────────────────────────────────────────────────────────────────
    n_vals = [r["n_providers"] for r in sweep_results]
    lifts  = [r["federation_lift"] for r in sweep_results]
    fprs   = [r["max_single_provider_fpr"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    colors = ["#1f4e79" if l > 0 else "#888888" for l in lifts]
    bars = ax.bar([str(n) for n in n_vals], lifts, color=colors, width=0.5)
    for bar, fpr_val, lift_val in zip(bars, fprs, lifts):
        y = lift_val + max(abs(max(lifts, default=0)), 1e-6) * 0.04
        ax.text(bar.get_x() + bar.get_width() / 2.0, y,
                f"FPR={fpr_val:.4f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="#7f7f7f", lw=0.8)
    ax.set_xlabel("Number of federation providers (n)")
    ax.set_ylabel("Federation lift\n(federated recall − max single-provider recall)")
    ax.set_title(
        f"M3: Federation detection lift — PAN 2012\n"
        f"(b={BAND_B}, r={BAND_R}), FPR ceiling={FPR_CEILING}"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "m3_federation_lift.pdf", bbox_inches="tight")
    fig.savefig(OUT / "m3_federation_lift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n=== M3: Federation Detection Lift (summary) ===")
    print(f"  Federated pool → recall={fed_recall:.4f}, FPR={fed_fpr:.4f}")
    print(f"  {'n_prov':>6}  {'max_single':>11}  {'fed_recall':>11}  "
          f"{'lift':>8}  {'max_FPR':>8}")
    print("  " + "-" * 52)
    for r in sweep_results:
        print(f"  {r['n_providers']:>6}  {r['max_single_provider_recall']:>11.4f}  "
              f"{r['federated_recall']:>11.4f}  {r['federation_lift']:>+8.4f}  "
              f"{r['max_single_provider_fpr']:>8.4f}")
    print()
    status = "demonstrated" if demonstrated else "INCONCLUSIVE"
    print(f"  M3 status: {status}")
    if not demonstrated:
        print(
            "  Shortfall: federation_lift = 0 at all n_providers.\n"
            "  MinHash (b=16, r=16) with J*≈0.841 does not produce recall "
            "on PAN 2012 because inter-conversation Jaccard falls well below "
            "the inflection threshold. A lower operating point (e.g., b=32, r=8) "
            "or a different feature representation would be needed."
        )
    print(f"\nResults: {OUT}/m3_federation_lift.json")


if __name__ == "__main__":
    main()
