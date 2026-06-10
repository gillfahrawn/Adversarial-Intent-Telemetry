#!/usr/bin/env python3
"""
Top-level runner for all four maturity-matrix upgrade experiments.

Runs experiments sequentially, loads their output JSON files, and
prints a TABLE 1 PATCH in the exact format required for the LaTeX source.

Usage:
    python run_all_experiments.py [--skip-pan12] [--skip-gt]

Options:
    --skip-pan12   Skip Experiments 1 (M3) and 3 (trajectory lift),
                   which require the PAN 2012 XML (large download).
    --skip-gt      Skip Experiment 4 (F3), which requires the GT-HarmBench CSV
                   (gated HuggingFace dataset).
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "experiments" / "results"

EXPERIMENTS = [
    {
        "name": "M3 federation lift",
        "script": "experiments/exp_m3_federation_lift.py",
        "result": "experiments/results/m3_federation_lift.json",
        "requires_pan12": True,
        "requires_gt": False,
    },
    {
        "name": "M8 Byzantine tolerance",
        "script": "experiments/exp_m8_byzantine.py",
        "result": "experiments/results/m8_byzantine.json",
        "requires_pan12": False,
        "requires_gt": False,
    },
    {
        "name": "Sec. 6 trajectory lift",
        "script": "experiments/exp_trajectory_lift.py",
        "result": "experiments/results/trajectory_lift.json",
        "requires_pan12": True,
        "requires_gt": False,
    },
    {
        "name": "F3 reciprocity",
        "script": "experiments/exp_f3_reciprocity.py",
        "result": "experiments/results/f3_reciprocity.json",
        "requires_pan12": False,
        "requires_gt": True,
    },
]


def run_experiment(exp: dict, skip_pan12: bool, skip_gt: bool) -> bool:
    """Run one experiment. Return True if it succeeded (or was skipped cleanly)."""
    if skip_pan12 and exp["requires_pan12"]:
        print(f"\n[SKIP] {exp['name']} (--skip-pan12)", flush=True)
        return False
    if skip_gt and exp["requires_gt"]:
        print(f"\n[SKIP] {exp['name']} (--skip-gt)", flush=True)
        return False

    print(f"\n{'='*60}", flush=True)
    print(f"Running: {exp['name']}", flush=True)
    print(f"Script:  {exp['script']}", flush=True)
    print("=" * 60, flush=True)

    result = subprocess.run(
        [sys.executable, exp["script"]],
        cwd=str(SCRIPT_DIR),
    )
    if result.returncode != 0:
        print(f"\n[FAILED] {exp['name']} exited with code {result.returncode}",
              file=sys.stderr)
        return False
    return True


def format_table1_patch(results: dict) -> str:
    """
    Format the TABLE 1 PATCH from loaded experiment JSON results.
    For skipped or failed experiments, reports '[not run]' or '[failed]'.
    """
    lines = [
        "",
        "=== TABLE 1 PATCH — copy into LaTeX source ===",
    ]

    # M3
    m3 = results.get("m3")
    if m3:
        km = m3.get("key_metric", {})
        status_str = m3.get("status", "inconclusive")
        lift = km.get("value", 0.0)
        n_prov = km.get("at_n_providers", "?")
        fpr = km.get("federated_fpr", km.get("fpr_ceiling", 0.005))
        lines.append(
            f"M3   federation_lift   hypothesized -> {status_str}\n"
            f"     (lift={lift:+.4f} at FPR≤{fpr:.4f}, n={n_prov} providers,"
            f" PAN 2012 train split)"
        )
    else:
        lines.append("M3   federation_lift   hypothesized -> [not run]")

    # M8
    m8 = results.get("m8")
    if m8:
        km = m8.get("key_metric", {})
        status_str = m8.get("status", "inconclusive")
        beta_star = km.get("value", 0.0)
        fr = m8.get("parameters", {})
        N = fr.get("N_senders", 20)
        n_trials = fr.get("n_trials_per_beta", 100)
        lines.append(
            f"M8   byzantine_beta    hypothesized -> {status_str}\n"
            f"     (β*={beta_star:.2f} >= 1/3, N={N}, {n_trials} trials/β)"
        )
    else:
        lines.append("M8   byzantine_beta    hypothesized -> [not run]")

    # Sec. 6 — two separate lines: F1 lift (primary criterion) and evasion-hardness
    traj = results.get("traj")
    if traj:
        km = traj.get("key_metric", {})
        f1_status = traj.get("status", "inconclusive")
        lift = km.get("value", 0.0)
        ci = km.get("bootstrap_95ci", [0.0, 0.0])
        ev = traj.get("evasion_simulation", {})
        dr = ev.get("drop_ratio", None)
        base_drop = ev.get("baseline_detection_drop", None)
        if dr == "infinity (baseline is order-invariant by construction)" or base_drop == 0.0:
            dr_str = "∞ (baseline is order-invariant by construction)"
            ev_status = "demonstrated"
        elif isinstance(dr, (int, float)) and dr > 1:
            dr_str = f"{dr:.2f}"
            ev_status = "demonstrated"
        else:
            dr_str = str(dr) if dr is not None else "n/a"
            ev_status = "inconclusive"
        lines.append(
            f"Sec6 trajectory_lift  F1 lift: {f1_status}"
            f" (lift={lift:+.4f}, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}])\n"
            f"     evasion-hardness: {ev_status} (drop_ratio={dr_str})\n"
            f"     PAN 2012 train split"
        )
    else:
        lines.append("Sec6 trajectory_lift  hypothesized -> [not run]")

    # F3
    f3 = results.get("f3")
    if f3:
        km = f3.get("key_metric", {})
        status_str = f3.get("status", "inconclusive")
        impr = km.get("value", 0.0)
        combined = f3.get("combined_a_b_c_tau_rho_delta_1", {})
        pd_impr = combined.get("per_game", {}).get(
            "Prisoner's Dilemma", {}).get("improvement", 0.0)
        lines.append(
            f"F3   reciprocity_mech  hypothesized -> {status_str}\n"
            f"     (defection reduced {impr:.1%} overall, "
            f"PD improvement={pd_impr:.1%}, GT-HarmBench train split)"
        )
    else:
        lines.append("F3   reciprocity_mech  hypothesized -> [not run]")

    lines.append("=" * 47)
    lines.append("")
    return "\n".join(lines)


def load_results(experiments: list, ran: dict) -> dict:
    loaded = {}
    keys = {"m3": "m3", "m8": "m8", "traj": "traj", "f3": "f3"}
    exp_keys = {
        "m3_federation_lift.json": "m3",
        "m8_byzantine.json": "m8",
        "trajectory_lift.json": "traj",
        "f3_reciprocity.json": "f3",
    }
    results_dir = SCRIPT_DIR / "experiments" / "results"
    for filename, key in exp_keys.items():
        path = results_dir / filename
        if path.exists():
            with open(path) as fh:
                loaded[key] = json.load(fh)
        else:
            loaded[key] = None
    return loaded


def main() -> None:
    skip_pan12 = "--skip-pan12" in sys.argv
    skip_gt    = "--skip-gt"    in sys.argv

    ran = {}
    for exp in EXPERIMENTS:
        success = run_experiment(exp, skip_pan12, skip_gt)
        ran[exp["result"]] = success

    results = load_results(EXPERIMENTS, ran)
    patch = format_table1_patch(results)
    print(patch)

    # Print combined status
    any_inconclusive = any(
        r and r.get("status") != "demonstrated"
        for r in results.values() if r is not None
    )
    if any_inconclusive:
        print("NOTE: One or more experiments are inconclusive.")
        print("      Actual numbers and shortfall explanations are in the JSON files.")
        print("      Do not upgrade the corresponding Maturity Matrix tags without")
        print("      resolving the shortfall.\n")


if __name__ == "__main__":
    main()
