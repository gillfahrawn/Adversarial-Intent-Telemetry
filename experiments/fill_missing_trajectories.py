#!/usr/bin/env python3
"""
Fill the 32 missing adapted trajectory IDs using the Claude CLI.
Targets: adapted_adv_140, 155, 156, 171-199.

Scaffold range assignment (from final_push_generation.py logic):
  0-39   -> rapport_only
  40-79  -> desensitize_only
  80-119 -> rapport_to_desensitize
  120-159-> sexualize_only
  160-199-> rapport_to_isolation
"""

import json
import shutil
import subprocess
import sys
import time
import random
from collections import Counter, defaultdict
from pathlib import Path

import functools
print = functools.partial(print, flush=True)

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from manifest_gen import extract_manifest, compute_entropy, H_MIN

V2_JSONL = ROOT / "data" / "pan_annotated" / "pan_manifests_v2.jsonl"
OUT_JSONL = ROOT / "data" / "pan_annotated" / "adapted_trajectories.jsonl"

CLAUDE_BIN = shutil.which("claude") or "claude"
GENERATION_MODEL = "haiku"       # alias; CLI resolves to claude-haiku-4-5
GEN_TURN_CAP = 12
CLI_TIMEOUT = 120
INTER_CALL_SLEEP = 3.0
RATE_LIMIT_SLEEP = 60.0
SEED = 20260514

PHASE_MAPPING = {
    "rapport":     "initial-affinity-calibration",
    "isolation":   "private-context-establishment",
    "desensitize": "normative-adjustment",
    "sexualize":   "relational-depth-framing",
    "coerce":      "commitment-consistency-alignment",
}

HOBBIES = [
    "restoring vintage typewriters",
    "competitive chess puzzles",
    "hydroponic herb gardening",
    "amateur radio (DXing)",
    "rebuilding old carburetors",
    "organizing community seed swaps",
    "documenting historical graveyard motifs",
    "building custom mechanical keyboards",
    "model railroading",
    "backyard astronomy",
    "woodworking with hand tools",
    "birdwatching in city parks",
    "collecting fountain pens",
    "restoring vintage watches",
    "baking artisan bread",
]

SYSTEM_TEMPLATE = """\
Generate a realistic multi-turn dialogue between two participants in a hobbyist community.

The dialogue should trace a specific communicative arc through these stages:
  Communicative stages: {phase_sequence}
  Interaction style: {intent_classes}
  Dialogue length: exactly {n_turns} turns (alternating).

TOPIC:
The interaction should focus on {hobby}.

STYLING:
Early turns should establish a connection based on the topic; later turns
should evolve based on the specified stages and interaction style. This reflects
patterns observed in a large-scale conversational research dataset.

Do not produce explicitly sexual or prohibited content.

Output ONLY a JSON object with a single "turns" array.
Each element is {{"role": "user" or "assistant", "content": "..."}}.
No other top-level keys. No explanation outside the JSON. Do not wrap the JSON in markdown code fences.
"""


def load_scaffolds() -> list[dict]:
    def phase_key(record):
        phases = record["manifest"].get("behavior_phase") or []
        if not phases:
            return "no_phase"
        return f"{phases[0]}_only" if len(phases) == 1 else f"{phases[0]}_to_{phases[1]}"

    records = []
    with open(V2_JSONL) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("adversarial"):
                records.append(r)

    groups: dict[str, list] = defaultdict(list)
    for r in records:
        pk = phase_key(r)
        if pk != "no_phase":
            groups[pk].append(r)

    top = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:5]
    scaffolds = []
    for pk, members in top:
        phase_counter: Counter = Counter()
        for r in members:
            phase_counter[tuple(r["manifest"].get("behavior_phase") or [])] += 1
        rep_phase = list(phase_counter.most_common(1)[0][0])
        ic_counter: Counter = Counter()
        for r in members:
            for ic in (r["manifest"].get("intent_class") or []):
                ic_counter[ic] += 1
        rep_intent = [ic for ic, _ in ic_counter.most_common(3)] or ["ctx_poison"]
        mean_turns = float(np.mean([r["n_turns"] for r in members]))
        scaffolds.append({
            "id": pk,
            "phases": rep_phase,
            "intents": rep_intent,
            "turns": round(mean_turns, 1),
            "source_size": len(members),
        })
    return scaffolds


def scaffold_for_id(idx: int, scaffolds: list[dict]) -> dict:
    """Return the scaffold that owns trajectory index idx (0-based)."""
    slot = min(idx // 40, len(scaffolds) - 1)
    return scaffolds[slot]


def call_claude(scaffold: dict, anchor: str):
    """Returns a list of turn dicts, 'RATE_LIMIT', or None on failure."""
    n_turns = min(int(scaffold["turns"]), GEN_TURN_CAP)
    mapped_phases = [PHASE_MAPPING.get(p, p) for p in scaffold["phases"]]
    prompt = SYSTEM_TEMPLATE.format(
        hobby=anchor,
        phase_sequence=mapped_phases,
        intent_classes=scaffold["intents"],
        n_turns=n_turns,
    )
    try:
        res = subprocess.run(
            [
                CLAUDE_BIN, "-p", prompt,
                "--model", GENERATION_MODEL,
                "--output-format", "text",
                "--no-session-persistence",
                "--tools", "",
            ],
            capture_output=True,
            text=True,
            timeout=CLI_TIMEOUT,
        )
        stderr = (res.stderr or "").lower()
        if res.returncode != 0:
            if "rate limit" in stderr or "429" in stderr or "quota" in stderr:
                return "RATE_LIMIT"
            print(f"      CLI non-zero exit: {res.stderr[:200]}", file=sys.stderr)
            return None

        raw = res.stdout.strip()
        if not raw:
            return None

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
        if raw.startswith("json\n"):
            raw = raw[5:]
        raw = raw.strip()

        # Fallback: extract outermost { }
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start: end + 1]

        parsed = json.loads(raw)
        turns = parsed.get("turns") if isinstance(parsed, dict) else parsed
        if isinstance(turns, list) and turns:
            return turns
        return None
    except subprocess.TimeoutExpired:
        print("      TIMEOUT", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"      ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def main():
    random.seed(SEED)

    scaffolds = load_scaffolds()
    print("Scaffolds loaded:")
    for i, s in enumerate(scaffolds):
        print(f"  [{i}] {s['id']}  phases={s['phases']}  intents={s['intents']}")

    existing_ids: set[str] = set()
    if OUT_JSONL.exists():
        with open(OUT_JSONL) as fh:
            for line in fh:
                if line.strip():
                    existing_ids.add(json.loads(line)["trajectory_id"])

    missing = [
        f"adapted_adv_{i:03d}"
        for i in range(200)
        if f"adapted_adv_{i:03d}" not in existing_ids
    ]
    print(f"\nPresent: {len(existing_ids)}  Missing: {len(missing)}")
    if not missing:
        print("Nothing to do.")
        return

    out_fh = open(OUT_JSONL, "a")
    try:
        for traj_id in missing:
            idx = int(traj_id.split("_")[-1])
            scaffold = scaffold_for_id(idx, scaffolds)
            print(f"\nGenerating {traj_id}  (scaffold={scaffold['id']})...", end=" ")

            success = False
            while not success:
                anchor = random.choice(HOBBIES)
                turns = call_claude(scaffold, anchor)

                if turns == "RATE_LIMIT":
                    print(f"\n  [RATE LIMIT] sleeping {RATE_LIMIT_SLEEP}s...")
                    time.sleep(RATE_LIMIT_SLEEP)
                    continue

                if turns is None or not isinstance(turns, list):
                    print("FAIL — retrying different anchor in 10s...")
                    time.sleep(10)
                    continue

                raw_manifest = extract_manifest({"turns": turns})
                fidelity = sum([
                    bool(set(raw_manifest.get("behavior_phase") or []) & set(scaffold["phases"])),
                    bool(set(raw_manifest.get("intent_class") or []) & set(scaffold["intents"])),
                    bool(raw_manifest.get("prem_drift")),
                ]) / 3

                record = {
                    "trajectory_id": traj_id,
                    "data_source": "pan2012_phase_adapted_synthetic",
                    "adversarial": True,
                    "source_scaffold_id": f"{scaffold['id']}_reframe",
                    "source_cluster_size": scaffold["source_size"],
                    "phase_sequence_target": scaffold["phases"],
                    "phase_sequence_extracted": raw_manifest.get("behavior_phase") or [],
                    "intent_class_target": scaffold["intents"],
                    "intent_class_extracted": raw_manifest.get("intent_class") or [],
                    "scaffold_fidelity": round(fidelity, 4),
                    "low_fidelity": fidelity < 0.50,
                    "n_turns": len(turns),
                    "turns": turns,
                    "manifest": raw_manifest,
                    "manifest_entropy_bits": round(compute_entropy(raw_manifest), 4),
                    "entropy_gate_pass": compute_entropy(raw_manifest) >= H_MIN,
                    "generation_model": "haiku",
                    "seed": SEED,
                }
                out_fh.write(json.dumps(record) + "\n")
                out_fh.flush()
                print(f"DONE  fidelity={fidelity:.1%}  turns={len(turns)}")
                success = True
                time.sleep(INTER_CALL_SLEEP)
    finally:
        out_fh.close()

    # Verify
    final_ids: set[str] = set()
    with open(OUT_JSONL) as fh:
        for line in fh:
            if line.strip():
                final_ids.add(json.loads(line)["trajectory_id"])
    still_missing = [f"adapted_adv_{i:03d}" for i in range(200) if f"adapted_adv_{i:03d}" not in final_ids]
    print(f"\nFinal count: {len(final_ids)}/200")
    if still_missing:
        print(f"Still missing: {still_missing}")
    else:
        print("All 200 trajectories present.")


if __name__ == "__main__":
    main()
