#!/usr/bin/env python3
"""
Experiment D3: M3 Federation Lift — Author-Disjoint PAN 2012 Split

Re-runs the M3 federation detection protocol with a split that eliminates
author-level leakage: no predator author appears in both train and test.

The original exp_m3_federation_lift.py splits by conversation_id. Because
PAN 2012 predators appear across multiple conversations, the same author's
stylistic signal can appear on both sides of the split, inflating recall
estimates. This experiment corrects that by splitting at the author level.

Protocol: identical to exp_m3_federation_lift.py — n=8 providers, b=16,
r=16, L=256, FPR ceiling=0.005, SEED=20260514.

Output: experiments/results/m3_author_split.json
Paper: Sec. 12 (M3), Maturity Matrix D3 caveat.
"""

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 20260514

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
OUT = SCRIPT_DIR / "results"
OUT.mkdir(parents=True, exist_ok=True)

DATA_XML = (
    ROOT / "data/pan12/train/"
    "pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
)
DATA_PRED = (
    ROOT / "data/pan12/train/"
    "pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"
)

# ── MinHash / band parameters (identical to exp_m3_federation_lift.py) ─────────
L = 256
BAND_B = 16
BAND_R = 16          # L = BAND_B * BAND_R
VOCAB_SIZE = 4096

FPR_CEILING = 0.005
N_PROVIDERS_SWEEP = [2, 4, 8]


# ── MinHash (verbatim from exp_m3_federation_lift.py) ─────────────────────────

def make_hash_funcs(l: int, vocab_size: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    p = 2**61 - 1
    a = rng.integers(1, p - 1, size=l, dtype=np.int64)
    b = rng.integers(0, p - 1, size=l, dtype=np.int64)
    return a, b, p


def minhash(token_set: np.ndarray, a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    h = (a[:, None] * token_set[None, :] + b[:, None]) % p
    return h.min(axis=1).astype(np.int64)


def build_band_index(sigs: list, b: int, r: int) -> list:
    index: list[dict] = [{} for _ in range(b)]
    for sig in sigs:
        bands = sig.reshape(b, r)
        for bi in range(b):
            index[bi][bands[bi].tobytes()] = True
    return index


def query_band_index(index: list, sig: np.ndarray, b: int, r: int) -> bool:
    bands = sig.reshape(b, r)
    for bi in range(b):
        if bands[bi].tobytes() in index[bi]:
            return True
    return False


# ── Feature extraction ─────────────────────────────────────────────────────────

def text_to_tokens(text: str) -> np.ndarray:
    tokens: set[int] = set()
    for i in range(len(text) - 3):
        gram = text[i:i + 4]
        digest = hashlib.md5(gram.encode("utf-8", errors="replace")).digest()
        tokens.add(int.from_bytes(digest[:4], "little") % VOCAB_SIZE)
    return np.array(sorted(tokens), dtype=np.int64)


def conv_to_sig(messages: list, a_hf: np.ndarray, b_hf: np.ndarray, p: int):
    full_text = " ".join(txt for _, txt in messages)
    tokens = text_to_tokens(full_text)
    if tokens.size == 0:
        return None
    return minhash(tokens, a_hf, b_hf, p)


# ── PAN 2012 I/O ───────────────────────────────────────────────────────────────

def load_predators(path: Path) -> set:
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def parse_pan12(xml_path: Path, predators: set) -> list:
    """
    Returns list of (conv_id, label, messages).
    label=1 if any message author is a known predator.
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


# ── Split strategies ───────────────────────────────────────────────────────────

def conv_id_split(conversations: list, test_frac: float = 0.2, seed: int = SEED):
    """
    Original stratified split by conversation_id.
    Identical to exp_m3_federation_lift.py — used for the 'old' baseline.
    Train list is positives-first so pos_idx maps correctly to shard positions.
    """
    conversations = sorted(conversations, key=lambda x: x[0])
    pos = [c for c in conversations if c[1] == 1]
    neg = [c for c in conversations if c[1] == 0]

    rng = np.random.default_rng(seed)
    pos_perm = rng.permutation(len(pos))
    neg_perm = rng.permutation(len(neg))

    n_pos_tr = int(len(pos) * (1.0 - test_frac))
    n_neg_tr = int(len(neg) * (1.0 - test_frac))

    train = ([pos[i] for i in pos_perm[:n_pos_tr]]
             + [neg[i] for i in neg_perm[:n_neg_tr]])
    test  = ([pos[i] for i in pos_perm[n_pos_tr:]]
             + [neg[i] for i in neg_perm[n_neg_tr:]])
    return train, test


def author_disjoint_split(conversations: list, predators: set,
                          test_frac: float = 0.2, seed: int = SEED):
    """
    Split by predator author ID: no predator author appears in both train and test.

    Positive conversations are assigned by their predator author's partition.
    If a conversation's predators span both partitions (edge case), it is
    assigned conservatively to test to avoid leakage.
    Negative conversations are split 80/20 by conversation_id using the same
    rng (continued after the author permutation for full reproducibility).

    Returns (train, test, train_authors, test_authors, all_pred_authors).
    Train list is positives-first to preserve the shard-mapping convention
    in run_m3.
    """
    # Predator authors per positive conversation
    conv_pred_authors: dict[str, set] = {}
    for conv_id, label, messages in conversations:
        if label == 1:
            conv_pred_authors[conv_id] = {a for a, _ in messages if a in predators}

    # All unique predator authors across all positive conversations, sorted for reproducibility
    all_pred_authors = sorted({a for auths in conv_pred_authors.values() for a in auths})

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_pred_authors))
    n_train_authors = int(len(all_pred_authors) * (1.0 - test_frac))

    train_author_set = {all_pred_authors[i] for i in perm[:n_train_authors]}
    test_author_set  = {all_pred_authors[i] for i in perm[n_train_authors:]}

    # Assign positive conversations; negatives collected separately
    train_pos, test_pos = [], []
    for conv_id, label, messages in sorted(conversations, key=lambda x: x[0]):
        if label != 1:
            continue
        pred_auths = conv_pred_authors.get(conv_id, set())
        # Any predator in test partition → test (conservative, prevents leakage)
        if pred_auths & test_author_set:
            test_pos.append((conv_id, label, messages))
        else:
            train_pos.append((conv_id, label, messages))

    # Negative conversations: 80/20 random split (rng continues from author permutation)
    neg_convs = sorted([c for c in conversations if c[1] == 0], key=lambda x: x[0])
    neg_perm = rng.permutation(len(neg_convs))
    n_neg_tr = int(len(neg_convs) * (1.0 - test_frac))
    train_neg = [neg_convs[i] for i in neg_perm[:n_neg_tr]]
    test_neg  = [neg_convs[i] for i in neg_perm[n_neg_tr:]]

    # Positives-first convention matches original experiment's shard-mapping logic
    train = train_pos + train_neg
    test  = test_pos  + test_neg

    return train, test, train_author_set, test_author_set, all_pred_authors


# ── M3 protocol ────────────────────────────────────────────────────────────────

def evaluate_pool(test_convs: list, pool_index: list,
                  a_hf: np.ndarray, b_hf: np.ndarray, p: int):
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


def run_m3(train: list, test: list, a_hf: np.ndarray, b_hf: np.ndarray, p_hf: int):
    """
    Run the M3 federated detection protocol on a given (train, test) split.
    Assumes train is positives-first (same convention as exp_m3_federation_lift.py).
    Returns (fed_recall, fed_fpr, sweep_results, n_valid_pos_sigs).
    """
    train_pos = [c for c in train if c[1] == 1]

    train_pos_sigs = [conv_to_sig(msgs, a_hf, b_hf, p_hf) for _, _, msgs in train_pos]
    valid_pos_sigs = [s for s in train_pos_sigs if s is not None]

    # Federated pool = union of all training-positive signatures
    fed_index = build_band_index(valid_pos_sigs, BAND_B, BAND_R)
    fed_recall, fed_fpr = evaluate_pool(test, fed_index, a_hf, b_hf, p_hf)

    sweep_results = []
    for n_prov in N_PROVIDERS_SWEEP:
        rng_shard = np.random.default_rng(SEED)
        n_train = len(train)
        perm = rng_shard.permutation(n_train)
        shard_size = n_train // n_prov

        # Map each position in train → shard id
        pos_to_shard: dict[int, int] = {}
        for i_shard in range(n_prov):
            start = i_shard * shard_size
            end = start + shard_size if i_shard < n_prov - 1 else n_train
            for gi in perm[start:end]:
                pos_to_shard[int(gi)] = i_shard

        # Training-positive signatures live at positions 0..len(train_pos)-1 in train
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

        sweep_results.append({
            "n_providers": n_prov,
            "max_single_provider_recall": prov_max_recall,
            "max_single_provider_fpr": prov_max_fpr,
            "federated_recall": fed_recall,
            "federated_fpr": fed_fpr,
            "federation_lift": fed_recall - prov_max_recall,
            "lift_positive": (fed_recall - prov_max_recall) > 0.0,
            "fpr_ceiling_used": FPR_CEILING,
            "provider_details": prov_details,
        })

    return fed_recall, fed_fpr, sweep_results, len(valid_pos_sigs)


# ── Main ───────────────────────────────────────────────────────────────────────

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
          f"{len(predators)} known predator IDs in ground-truth file")

    a_hf, b_hf, p_hf = make_hash_funcs(l=L, vocab_size=VOCAB_SIZE, seed=SEED)

    # ── Old split: conversation-ID-based (identical to existing experiment) ─────
    print("\n[OLD] Running M3 with conversation-ID split …")
    old_train, old_test = conv_id_split(conversations)
    old_fed_r, old_fed_fpr, old_sweep, old_n_valid = run_m3(
        old_train, old_test, a_hf, b_hf, p_hf
    )
    old_at8 = next(r for r in old_sweep if r["n_providers"] == 8)
    print(f"  n=8  fed_recall={old_fed_r:.4f}  fed_fpr={old_fed_fpr:.4f}  "
          f"max_single={old_at8['max_single_provider_recall']:.4f}  "
          f"lift={old_at8['federation_lift']:+.4f}")

    # ── New split: author-disjoint ─────────────────────────────────────────────
    print("\n[NEW] Running M3 with author-disjoint split …")
    new_train, new_test, train_authors, test_authors, all_authors = (
        author_disjoint_split(conversations, predators)
    )
    new_fed_r, new_fed_fpr, new_sweep, new_n_valid = run_m3(
        new_train, new_test, a_hf, b_hf, p_hf
    )
    new_at8 = next(r for r in new_sweep if r["n_providers"] == 8)
    print(f"  n=8  fed_recall={new_fed_r:.4f}  fed_fpr={new_fed_fpr:.4f}  "
          f"max_single={new_at8['max_single_provider_recall']:.4f}  "
          f"lift={new_at8['federation_lift']:+.4f}")

    # ── Split statistics ───────────────────────────────────────────────────────
    old_tr_pos = sum(1 for c in old_train if c[1] == 1)
    old_te_pos = sum(1 for c in old_test  if c[1] == 1)
    old_te_neg = sum(1 for c in old_test  if c[1] == 0)
    new_tr_pos = sum(1 for c in new_train if c[1] == 1)
    new_te_pos = sum(1 for c in new_test  if c[1] == 1)
    new_te_neg = sum(1 for c in new_test  if c[1] == 0)

    # ── Deltas at n=8 ─────────────────────────────────────────────────────────
    delta_fed_r   = new_fed_r   - old_fed_r
    delta_fed_fpr = new_fed_fpr - old_fed_fpr
    delta_single  = (new_at8["max_single_provider_recall"]
                     - old_at8["max_single_provider_recall"])
    delta_lift    = new_at8["federation_lift"] - old_at8["federation_lift"]

    # ── Output JSON ────────────────────────────────────────────────────────────
    output = {
        "experiment": "m3_author_split",
        "split_type": "author_disjoint",
        "dataset": (
            "PAN 2012 Sexual Predator Identification, training XML, "
            "author-disjoint 80/20 split (D3)"
        ),
        "minhash_params": {"L": L, "b": BAND_B, "r": BAND_R, "vocab_size": VOCAB_SIZE},
        "fpr_ceiling": FPR_CEILING,
        "n_total_conversations": len(conversations),
        "n_predator_ids_in_groundtruth": len(predators),
        "n_predator_authors_observed_in_xml": len(all_authors),
        "n_predator_authors_train": len(train_authors),
        "n_predator_authors_test": len(test_authors),
        "comparison_at_n8": {
            "old_conv_split": {
                "split_type": "conversation_id",
                "n_train": len(old_train),
                "n_test": len(old_test),
                "n_train_pos": old_tr_pos,
                "n_test_pos": old_te_pos,
                "n_test_neg": old_te_neg,
                "n_valid_train_pos_sigs": old_n_valid,
                "federated_recall": old_fed_r,
                "federated_fpr": old_fed_fpr,
                "max_single_provider_recall": old_at8["max_single_provider_recall"],
                "max_single_provider_fpr": old_at8["max_single_provider_fpr"],
                "federation_lift": old_at8["federation_lift"],
            },
            "new_author_disjoint": {
                "split_type": "author_disjoint",
                "n_train": len(new_train),
                "n_test": len(new_test),
                "n_train_pos": new_tr_pos,
                "n_test_pos": new_te_pos,
                "n_test_neg": new_te_neg,
                "n_valid_train_pos_sigs": new_n_valid,
                "federated_recall": new_fed_r,
                "federated_fpr": new_fed_fpr,
                "max_single_provider_recall": new_at8["max_single_provider_recall"],
                "max_single_provider_fpr": new_at8["max_single_provider_fpr"],
                "federation_lift": new_at8["federation_lift"],
            },
            "delta_new_minus_old": {
                "federated_recall": delta_fed_r,
                "federated_fpr": delta_fed_fpr,
                "max_single_provider_recall": delta_single,
                "federation_lift": delta_lift,
                "note": (
                    "Negative delta on recall indicates that the original "
                    "conversation-ID split inflated estimates via author-level "
                    "leakage. The author-disjoint result is the corrected baseline."
                ),
            },
        },
        "full_sweep": {
            "old_conv_split": old_sweep,
            "new_author_disjoint": new_sweep,
        },
        "caveats": [
            "Author-disjoint split: no predator author appears in both train and test. "
            "Conversations with predators spanning both author partitions are assigned "
            "conservatively to test.",
            "Negative conversations are split 80/20 by conversation_id (no author-level "
            "leakage concern for negatives).",
            "MinHash at (b=16, r=16) with J*≈0.841 produces low absolute recall on "
            "PAN 2012 because inter-conversation 4-gram Jaccard falls well below the "
            "inflection point (empirical adversarial mean ≪ 0.618 synthetic assumption).",
            "Results are on PAN 2012 (human grooming, 2012). Validation on synthetic "
            "agentic trajectories or pilot deployment data is required before "
            "generalising to the A2 threat class.",
            "The operating-point frontier on PAN 2012 (D4) depends on this split "
            "completing first.",
        ],
    }

    out_path = OUT / "m3_author_split.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)

    # ── Comparison table ───────────────────────────────────────────────────────
    print("\n=== D3: M3 Author-Disjoint Split — Comparison at n=8 providers ===\n")
    hdr = f"  {'Metric':<32} {'Old (conv-split)':>17} {'New (author-disj)':>18} {'Delta':>12}"
    sep = "  " + "-" * 83
    print(hdr)
    print(sep)

    def row(label, old_v, new_v, d):
        sign = "+" if d >= 0 else ""
        print(f"  {label:<32} {old_v:>17.4f} {new_v:>18.4f} {sign}{d:>11.4f}")

    row("Federated recall",        old_fed_r,   new_fed_r,   delta_fed_r)
    row("Federated FPR",           old_fed_fpr, new_fed_fpr, delta_fed_fpr)
    row("Max single-provider recall", old_at8["max_single_provider_recall"],
        new_at8["max_single_provider_recall"], delta_single)
    row("Federation lift",         old_at8["federation_lift"],
        new_at8["federation_lift"], delta_lift)

    print()
    print(f"  Old split:  train={len(old_train):>5}  test={len(old_test):>5}  "
          f"(pos train={old_tr_pos} / pos test={old_te_pos})")
    print(f"  New split:  train={len(new_train):>5}  test={len(new_test):>5}  "
          f"(pos train={new_tr_pos} / pos test={new_te_pos})")
    print(f"  Predator authors: {len(all_authors)} observed in XML → "
          f"{len(train_authors)} train / {len(test_authors)} test partition")
    print()

    if delta_fed_r < -0.001:
        print("  Interpretation: negative delta confirms author-level leakage in the")
        print("  original conv-split; corrected recall is lower. Author-disjoint result")
        print("  is the conservative baseline for Table 1 (M3 claim).")
    elif abs(delta_fed_r) <= 0.001:
        print("  Interpretation: delta ≈ 0. MinHash at (b=16,r=16) produces near-zero")
        print("  recall on both splits; leakage magnitude is undetectable at J*≈0.841.")
        print("  The author-disjoint split remains the methodologically correct baseline.")
    else:
        print("  Interpretation: author-disjoint split yields higher recall than conv-split.")
        print("  This is unexpected; inspect provider shard sizes for distribution shift.")

    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
