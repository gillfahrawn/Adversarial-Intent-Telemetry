#!/usr/bin/env python3
"""
Experiment: Annotate PAN 2012 conversations with feature manifests.

Re-derives the author-disjoint training split from D3 (SEED=20260514, identical
algorithm to exp_m3_author_split.py), then runs manifest extraction on each
training-split adversarial conversation and a matched benign sample (capped at
500 each). Outputs a JSONL annotation file and a summary quality-assessment
report.

Note: experiments/results/m3_author_split.json records aggregate metrics only;
individual conversation IDs are not stored there. The split is recomputed here
using the same deterministic algorithm and SEED so results are identical.

Output files:
  experiments/results/pan_manifest_annotated.jsonl
  experiments/results/pan_manifest_annotation_report.json

Conventions (identical to all other experiments in this repo):
  SEED=20260514, scikit-learn only, no API calls, standalone runnable script.
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SEED = 20260514
CAP = 500

SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent

# ── Import manifest logic from tools/ ─────────────────────────────────────────
sys.path.insert(0, str(ROOT / "tools"))
try:
    from manifest_gen import extract_manifest, compute_entropy, H_MIN
except ImportError as exc:
    print(
        f"ERROR: Cannot import manifest_gen: {exc}\n"
        "Install requirements: pip install -r tools/requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_XML = (
    ROOT / "data/pan12/train"
    / "pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
)
DATA_PRED = (
    ROOT / "data/pan12/train"
    / "pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"
)

DEFAULT_OUT = SCRIPT_DIR / "results"

# ── Manifest field ordering (matches JSONL output schema) ─────────────────────
MANIFEST_FIELDS = [
    "intent_class", "behavior_phase", "phase_transition",
    "embed_anom_bin", "role_marker", "hypoth_frame",
    "decomp_signal", "prem_drift", "provenance_fail",
    "demographic_prior", "scope_class",
]

# Fields structurally always populated by manifest_gen; excluded from the
# low-population diagnostic (their rate is ~1.0 by construction, not by signal).
ALWAYS_POPULATED = {"scope_class", "embed_anom_bin", "decomp_signal"}


# ── PAN 2012 I/O (identical to D3) ────────────────────────────────────────────

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
            conv_id  = elem.get("id", "")
            messages = []
            for msg in elem.findall("message"):
                author = (msg.findtext("author") or "").strip()
                text   = (msg.findtext("text")   or "").strip()
                messages.append((author, text))
            label = 1 if any(a in predators for a, _ in messages) else 0
            conversations.append((conv_id, label, messages))
            elem.clear()
    return conversations


# ── Author-disjoint split (identical to D3) ────────────────────────────────────

def author_disjoint_split(conversations: list, predators: set,
                           test_frac: float = 0.2, seed: int = SEED):
    """
    Re-derives the D3 split deterministically. Returns (train_pos, train_neg,
    train_author_set). Algorithm is byte-for-byte identical to
    exp_m3_author_split.py so the partition is reproducible given the same SEED.

    Positive conversations are assigned by their predator author's partition.
    Conversations whose predators span both author partitions are assigned
    conservatively to test to prevent leakage.
    """
    conv_pred_authors: dict[str, set] = {}
    for conv_id, label, messages in conversations:
        if label == 1:
            conv_pred_authors[conv_id] = {a for a, _ in messages if a in predators}

    all_pred_authors = sorted(
        {a for auths in conv_pred_authors.values() for a in auths}
    )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_pred_authors))
    n_train_authors = int(len(all_pred_authors) * (1.0 - test_frac))

    train_author_set = {all_pred_authors[i] for i in perm[:n_train_authors]}
    test_author_set  = {all_pred_authors[i] for i in perm[n_train_authors:]}

    train_pos = []
    for conv_id, label, messages in sorted(conversations, key=lambda x: x[0]):
        if label != 1:
            continue
        pred_auths = conv_pred_authors.get(conv_id, set())
        if not (pred_auths & test_author_set):
            train_pos.append((conv_id, label, messages))

    neg_convs = sorted([c for c in conversations if c[1] == 0], key=lambda x: x[0])
    neg_perm  = rng.permutation(len(neg_convs))
    n_neg_tr  = int(len(neg_convs) * (1.0 - test_frac))
    train_neg = [neg_convs[i] for i in neg_perm[:n_neg_tr]]

    return train_pos, train_neg, train_author_set


# ── Conversation → trajectory conversion ──────────────────────────────────────

def to_trajectory(messages: list, predators: set, is_adversarial: bool) -> dict:
    """
    Convert PAN 2012 (author, text) pairs to the trajectory dict expected by
    manifest_gen ({"turns": [{"role": ..., "content": ...}]}).

    Adversarial: predator author → role "user" (the adversarial actor whose
    language the rule patterns target); victim → role "assistant".

    Benign: first unique speaker → role "user", others → role "assistant".
    This gives the rule-based extractor access to some "user" text while
    reflecting the initiator/responder structure of a two-party chat.
    """
    if is_adversarial:
        turns = [
            {"role": "user" if author in predators else "assistant", "content": text}
            for author, text in messages
            if text
        ]
    else:
        speaker_role: dict[str, str] = {}
        turns = []
        for author, text in messages:
            if not text:
                continue
            if author not in speaker_role:
                speaker_role[author] = "user" if not speaker_role else "assistant"
            turns.append({"role": speaker_role[author], "content": text})

    return {"turns": turns}


# ── JSONL record construction ──────────────────────────────────────────────────

def _to_list(v):
    """Normalise a manifest field value to a list, as required by the JSONL schema."""
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def normalise_manifest(raw: dict) -> dict:
    """
    Convert raw manifest dict (from extract_manifest) to the JSONL output schema.
    All fields are emitted as lists; absent fields become [].
    scope_class is emitted as a one-element list, e.g. ["CSAE"].
    """
    return {f: _to_list(raw.get(f)) for f in MANIFEST_FIELDS}


def count_manifest_tokens(normalised: dict) -> int:
    """Total element count across all manifest fields (manifest cardinality)."""
    return sum(len(v) for v in normalised.values())


def annotate_conv(conv_id: str, label: int, messages: list, predators: set) -> dict:
    is_adv = (label == 1)
    pred_authors = [a for a, _ in messages if a in predators]
    pred_author_id = pred_authors[0] if pred_authors else ""

    traj = to_trajectory(messages, predators, is_adv)
    raw  = extract_manifest(traj)
    ent  = compute_entropy(raw)

    return {
        "conversation_id":    f"pan2012_train_{conv_id}",
        "data_source":        "pan2012_real_annotated",
        "adversarial":        is_adv,
        "predator_author_id": pred_author_id,
        "n_turns":            len(messages),
        "manifest":           normalise_manifest(raw),
        "manifest_entropy_bits": round(ent, 4),
        "entropy_gate_pass":  ent >= H_MIN,
    }


# ── Summary helpers ────────────────────────────────────────────────────────────

def population_rate(records: list, field: str) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r["manifest"][field]) / len(records)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUT),
        help="Output directory (default: experiments/results)",
    )
    parser.add_argument(
        "--version-suffix",
        default="",
        help="Suffix for output filenames, e.g. '_v2'. "
             "When empty, uses original filenames.",
    )
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.version_suffix
    if suffix:
        out_jsonl  = out_dir / f"pan_manifests{suffix}.jsonl"
        out_report = out_dir / f"annotation_report{suffix}.json"
    else:
        out_jsonl  = out_dir / "pan_manifest_annotated.jsonl"
        out_report = out_dir / "pan_manifest_annotation_report.json"

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

    print("Parsing PAN 2012 XML (~60 s) …")
    predators     = load_predators(DATA_PRED)
    conversations = parse_pan12(DATA_XML, predators)
    print(f"  {len(conversations)} conversations, {len(predators)} predator IDs")

    print("Re-deriving author-disjoint training split (D3, SEED=20260514) …")
    train_pos, train_neg, _ = author_disjoint_split(conversations, predators)
    print(f"  Training adversarial: {len(train_pos)},  benign: {len(train_neg)}")

    # Sample (cap at CAP each; SEED+1 distinguishes from split rng)
    rng = np.random.default_rng(SEED + 1)
    sample_adv = [
        train_pos[i]
        for i in sorted(
            rng.choice(len(train_pos), size=min(CAP, len(train_pos)), replace=False)
        )
    ]
    sample_ben = [
        train_neg[i]
        for i in sorted(
            rng.choice(len(train_neg), size=min(CAP, len(train_neg)), replace=False)
        )
    ]

    # ── Annotate adversarial ───────────────────────────────────────────────────
    print(f"Annotating {len(sample_adv)} adversarial conversations …")
    adv_records = []
    for i, (conv_id, label, messages) in enumerate(sample_adv):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(sample_adv)}")
        adv_records.append(annotate_conv(conv_id, label, messages, predators))

    # ── Annotate benign ────────────────────────────────────────────────────────
    print(f"Annotating {len(sample_ben)} benign conversations …")
    ben_records = []
    for i, (conv_id, label, messages) in enumerate(sample_ben):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(sample_ben)}")
        ben_records.append(annotate_conv(conv_id, label, messages, predators))

    # ── Write JSONL ────────────────────────────────────────────────────────────
    all_records = adv_records + ben_records
    with open(out_jsonl, "w") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec) + "\n")
    print(f"\nJSONL written: {out_jsonl}  ({len(all_records)} records)")

    # ── Compute summary statistics ─────────────────────────────────────────────
    n_adv = len(adv_records)
    n_ben = len(ben_records)

    mean_ent_adv  = float(np.mean([r["manifest_entropy_bits"] for r in adv_records]))
    mean_ent_ben  = float(np.mean([r["manifest_entropy_bits"] for r in ben_records]))
    mean_tok_adv  = float(np.mean([count_manifest_tokens(r["manifest"]) for r in adv_records]))
    mean_tok_ben  = float(np.mean([count_manifest_tokens(r["manifest"]) for r in ben_records]))
    gate_pass_rate = sum(r["entropy_gate_pass"] for r in adv_records) / n_adv

    adv_pop = {f: population_rate(adv_records, f) for f in MANIFEST_FIELDS}

    # Low-population diagnostic: exclude structurally-always-populated fields
    low_fields = [
        f for f in MANIFEST_FIELDS
        if f not in ALWAYS_POPULATED and adv_pop[f] < 0.10
    ]

    entropy_higher = mean_ent_adv > mean_ent_ben
    most_above_20  = all(
        adv_pop[f] > 0.20
        for f in MANIFEST_FIELDS
        if f not in ALWAYS_POPULATED
    )

    if entropy_higher and most_above_20:
        finding = (
            "Annotation successful. Adversarial conversations produce richer "
            "manifest features than benign ones."
        )
    else:
        low_str = ", ".join(low_fields) if low_fields else "none"
        finding = (
            f"Annotation partially successful. Fields [{low_str}] have low "
            "detection rates. Rule extension recommended before C1/C2."
        )

    report = {
        "n_adversarial_annotated":            n_adv,
        "n_benign_annotated":                 n_ben,
        "adversarial_entropy_gate_pass_rate": round(gate_pass_rate, 4),
        "mean_manifest_entropy_adversarial":  round(mean_ent_adv, 4),
        "mean_manifest_entropy_benign":       round(mean_ent_ben, 4),
        "mean_n_tokens_adversarial":          round(mean_tok_adv, 2),
        "mean_n_tokens_benign":               round(mean_tok_ben, 2),
        "field_population_rates":             {f: round(adv_pop[f], 4) for f in MANIFEST_FIELDS},
        "finding":                            finding,
        "caveats": [
            "manifest_gen.py patterns target AI jailbreak trajectories (role injection, "
            "hypothetical frames, bypass language). PAN 2012 is human-human grooming chat "
            "(2012); pattern vocabulary overlap is low by design. Low population rates "
            "motivate rule extension, not failure.",
            "Adversarial conversations: predator messages assigned role='user'; "
            "victim messages assigned role='assistant'.",
            "Benign conversations: first unique speaker assigned role='user'; "
            "others assigned role='assistant'.",
            "Training split re-derived from D3 algorithm (SEED=20260514); "
            "m3_author_split.json does not carry per-conversation IDs.",
            "field_population_rates are adversarial population rates. "
            "scope_class, embed_anom_bin, and decomp_signal are structurally "
            "always populated by manifest_gen and are excluded from the "
            "low-population diagnostic.",
        ],
    }

    with open(out_report, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Report written: {out_report}")

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n=== PAN 2012 Manifest Annotation Report ===\n")
    print(f"  Adversarial annotated : {n_adv}  (of {len(train_pos)} training-split positives)")
    print(f"  Benign annotated      : {n_ben}  (of {len(train_neg)} training-split negatives)")
    print()
    print(f"  Entropy gate pass rate (adversarial) : {gate_pass_rate:.1%}")
    print(f"  Mean manifest entropy  (adversarial) : {mean_ent_adv:.1f} bits")
    print(f"  Mean manifest entropy  (benign)      : {mean_ent_ben:.1f} bits")
    print(f"  Mean manifest tokens   (adversarial) : {mean_tok_adv:.1f}")
    print(f"  Mean manifest tokens   (benign)      : {mean_tok_ben:.1f}")
    print()
    print(f"  {'Field':<22}  {'Adv rate':>9}  Quality")
    print("  " + "-" * 42)
    for f in MANIFEST_FIELDS:
        rate = adv_pop[f]
        if f in ALWAYS_POPULATED:
            flag = "structural"
        elif rate < 0.10:
            flag = "LOW"
        else:
            flag = "ok"
        print(f"  {f:<22}  {rate:>8.1%}  {flag}")
    print()
    print(f"  Finding: {finding}")
    print(f"\nResults:\n  {out_jsonl}\n  {out_report}")


if __name__ == "__main__":
    main()
