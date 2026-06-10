#!/usr/bin/env python3
"""
Experiment F3: Reciprocity Mechanism on GT-HarmBench

Tests whether the F3 mechanism (trust-decay penalty + reciprocity gate +
per-class FP ceiling bonus) reduces strategic defection on the social-dilemma
subset of the GT-HarmBench training split.

Framing: This experiment applies F3 as a payoff perturbation to existing 2×2
game matrices in the GT-HarmBench training split and measures the resulting shift
in Nash equilibria toward cooperation. It is an analytical claim about mechanism
design, not an evaluation of LLM behavior. Results are on the training split only;
test split evaluation is noted as future work.

Paper: Appendix B (F3 mechanism).
Dataset: GT-HarmBench training split.
Status upgrade target: hypothesized → demonstrated.
"""

import ast
import json
import sys
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

DATA_CSV = ROOT / "data/gt_harmbench/GTHarmbenchdatatrain00000of00001.csv"

# ── Expected validation counts ──────────────────────────────────────────────────
EXPECTED_ROWS = 2009
EXPECTED_PD   = 654
EXPECTED_CHICKEN = 491
EXPECTED_STAG = 403
EXPECTED_SOCIAL_DILEMMA = 1528

# ── Intervention parameter sweeps ─────────────────────────────────────────────
PARAM_SWEEP = [0.5, 1.0, 1.5, 2.0]
PRIMARY_PARAM = 1.0   # τ = ρ = δ = 1.0 for combined A+B+C

GAME_TYPES = ["Prisoner's Dilemma", "Chicken", "Stag hunt"]


# ── Game theory helpers ─────────────────────────────────────────────────────────
def pure_nash_equilibria(R: list, C: list) -> list:
    """
    Return list of (i, j) pairs that are pure Nash equilibria of the 2×2 game.
    (i, j) is Nash iff R[i][j] >= R[1-i][j] and C[i][j] >= C[i][1-j].
    """
    nash = []
    for i in range(2):
        for j in range(2):
            if R[i][j] >= R[1 - i][j] and C[i][j] >= C[i][1 - j]:
                nash.append((i, j))
    return nash


def pareto_optimal_profiles(R: list, C: list) -> list:
    """
    Return list of (i, j) pairs that are Pareto-optimal.
    (i, j) is Pareto-optimal if no (i', j') Pareto-dominates it.
    """
    profiles = [(i, j) for i in range(2) for j in range(2)]
    pareto = []
    for (i, j) in profiles:
        dominated = False
        for (ip, jp) in profiles:
            if (ip, jp) == (i, j):
                continue
            if (R[ip][jp] >= R[i][j] and C[ip][jp] >= C[i][j] and
                    (R[ip][jp] > R[i][j] or C[ip][jp] > C[i][j])):
                dominated = True
                break
        if not dominated:
            pareto.append((i, j))
    return pareto


def is_defection_equilibrium(R: list, C: list):
    """
    Returns True if ALL pure Nash equilibria are NOT Pareto-optimal.
    Returns False if at least one Nash equilibrium is Pareto-optimal.
    Returns None if there are no pure Nash equilibria.
    """
    nash = pure_nash_equilibria(R, C)
    if not nash:
        return None
    pareto = set(pareto_optimal_profiles(R, C))
    return all((i, j) not in pareto for (i, j) in nash)


def social_welfare_maximising(R: list, C: list) -> tuple:
    """Return (i, j) that maximises R[i][j] + C[i][j]."""
    best = None
    best_sw = -1e9
    for i in range(2):
        for j in range(2):
            sw = R[i][j] + C[i][j]
            if sw > best_sw:
                best_sw = sw
                best = (i, j)
    return best


def apply_intervention_a(R, C, tau, nash_defect_profiles):
    """Trust-decay penalty τ: lower the defecting action row/column payoffs."""
    R_new = [row[:] for row in R]
    C_new = [row[:] for row in C]
    for (i_d, j_d) in nash_defect_profiles:
        for j in range(2):
            R_new[i_d][j] -= tau
        for i in range(2):
            C_new[i][j_d] -= tau
    return R_new, C_new


def apply_intervention_b(R, C, rho, nash_defect_profiles):
    """Reciprocity gate ρ: lower mutual-defection payoffs (repeated-game penalty)."""
    R_new = [row[:] for row in R]
    C_new = [row[:] for row in C]
    for (i_d, j_d) in nash_defect_profiles:
        R_new[i_d][j_d] -= rho
        C_new[i_d][j_d] -= rho
    return R_new, C_new


def apply_intervention_c(R, C, delta, coop_profile):
    """Per-class FP ceiling bonus δ: increase cooperation payoffs."""
    R_new = [row[:] for row in R]
    C_new = [row[:] for row in C]
    i_p, j_p = coop_profile
    R_new[i_p][j_p] += delta
    C_new[i_p][j_p] += delta
    return R_new, C_new


def apply_all(R, C, tau, rho, delta):
    """Apply A + B + C combined."""
    nash = pure_nash_equilibria(R, C)
    pareto = set(pareto_optimal_profiles(R, C))
    defect_nash = [(i, j) for (i, j) in nash if (i, j) not in pareto]
    coop = social_welfare_maximising(R, C)
    R2, C2 = apply_intervention_a(R, C, tau, defect_nash)
    R2, C2 = apply_intervention_b(R2, C2, rho, defect_nash)
    R2, C2 = apply_intervention_c(R2, C2, delta, coop)
    return R2, C2


# ── Data loading ────────────────────────────────────────────────────────────────
def parse_payoff(s: str):
    """Parse '[a, b]' string → [int, int]. Returns None on failure."""
    if not s or s in ("\\N", "N", ""):
        return None
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return [int(val[0]), int(val[1])]
    except Exception:
        pass
    return None


def load_gt_harmbench(path: Path) -> list:
    """
    Load GT-HarmBench CSV. Validates expected row and category counts.
    Each returned record is a dict with keys:
      id, formal_game, risk_category, risk_level,
      R (2x2 row-player payoff matrix), C (2x2 col-player payoff matrix),
      target_nash, target_utility, target_rawlsian, target_nsw
    Fails loudly on validation error or missing file.
    """
    import csv

    if not path.exists():
        print("ERROR: GT-HarmBench data file not found:", file=sys.stderr)
        print(f"  Expected: {path}", file=sys.stderr)
        print(
            "\nThe GT-HarmBench dataset is hosted on HuggingFace:\n"
            "  https://huggingface.co/datasets/gtfintechlab/GT-HarmBench\n"
            "Note: this dataset is gated and requires HuggingFace authentication.\n"
            "After access is granted, download the training split:\n"
            "  huggingface-cli download gtfintechlab/GT-HarmBench \\\n"
            "    data/train-00000-of-00001.parquet --repo-type dataset\n"
            "Convert to CSV and place at:\n"
            f"  {path}",
            file=sys.stderr,
        )
        sys.exit(1)

    records = []
    skipped = 0
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        all_rows = list(reader)

    if len(all_rows) != EXPECTED_ROWS:
        print(
            f"ERROR: Expected {EXPECTED_ROWS} rows, found {len(all_rows)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    for row in all_rows:
        p11 = parse_payoff(row.get("1_1_payoff", "\\N"))
        p12 = parse_payoff(row.get("1_2_payoff", "\\N"))
        p21 = parse_payoff(row.get("2_1_payoff", "\\N"))
        p22 = parse_payoff(row.get("2_2_payoff", "\\N"))

        if any(p is None for p in (p11, p12, p21, p22)):
            skipped += 1
            continue

        R = [[p11[0], p12[0]], [p21[0], p22[0]]]
        C = [[p11[1], p12[1]], [p21[1], p22[1]]]

        records.append({
            "id": row.get("id", ""),
            "formal_game": row.get("formal_game", ""),
            "risk_category": row.get("risk_category", ""),
            "risk_level": row.get("risk_level", ""),
            "R": R,
            "C": C,
            "target_nash": row.get("target_nash_equilibria", ""),
            "target_utility": row.get("target_utility_maximizing", ""),
            "target_rawlsian": row.get("target_rawlsian", ""),
            "target_nsw": row.get("target_nash_social_welfare", ""),
        })

    # Validation counts
    pd_count = sum(1 for r in records if r["formal_game"] == "Prisoner's Dilemma")
    ch_count = sum(1 for r in records if r["formal_game"] == "Chicken")
    sg_count = sum(1 for r in records if r["formal_game"] == "Stag hunt")

    def is_social_dilemma(r):
        return (r["target_nash"] and r["target_utility"] and
                r["target_nash"] != r["target_utility"])

    sd_count = sum(1 for r in records if is_social_dilemma(r))

    errors = []
    if pd_count != EXPECTED_PD:
        errors.append(f"Prisoner's Dilemma rows: expected {EXPECTED_PD}, got {pd_count}")
    if ch_count != EXPECTED_CHICKEN:
        errors.append(f"Chicken rows: expected {EXPECTED_CHICKEN}, got {ch_count}")
    if sg_count != EXPECTED_STAG:
        errors.append(f"Stag hunt rows: expected {EXPECTED_STAG}, got {sg_count}")
    if sd_count != EXPECTED_SOCIAL_DILEMMA:
        errors.append(
            f"Social dilemma rows: expected {EXPECTED_SOCIAL_DILEMMA}, got {sd_count}"
        )
    if errors:
        print("ERROR: GT-HarmBench validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Loaded {len(records)} valid records ({skipped} skipped: unparseable payoffs)")
    print(f"  PD={pd_count}, Chicken={ch_count}, Stag hunt={sg_count}, "
          f"Social dilemma={sd_count}")
    return records


# ── Analysis helpers ────────────────────────────────────────────────────────────
def compute_defection_rate(records, intervention_fn=None):
    """
    Compute defection rate and per-game-type breakdown.
    intervention_fn: callable(R, C) → (R_new, C_new) or None for baseline.
    Returns dict with 'overall' and per-game-type rates, and per-record flags.
    """
    counts = {g: {"total": 0, "defection": 0, "no_nash": 0} for g in GAME_TYPES}
    counts["Overall"] = {"total": 0, "defection": 0, "no_nash": 0}

    for r in records:
        fg = r["formal_game"]
        if fg not in GAME_TYPES:
            continue
        R, C = r["R"], r["C"]
        if intervention_fn is not None:
            R, C = intervention_fn(R, C)
        result = is_defection_equilibrium(R, C)
        if result is None:
            counts[fg]["no_nash"] += 1
            counts["Overall"]["no_nash"] += 1
            continue
        counts[fg]["total"] += 1
        counts["Overall"]["total"] += 1
        if result:
            counts[fg]["defection"] += 1
            counts["Overall"]["defection"] += 1

    rates = {}
    for key, d in counts.items():
        rates[key] = {
            "defection_rate": d["defection"] / max(d["total"], 1),
            "n_defection": d["defection"],
            "n_total_with_nash": d["total"],
            "n_no_nash": d["no_nash"],
        }
    return rates


def defection_rate_scalar(records, game_type="Overall", intervention_fn=None) -> float:
    rates = compute_defection_rate(records, intervention_fn)
    return rates[game_type]["defection_rate"]


# ── Social-dilemma subset ───────────────────────────────────────────────────────
def get_social_dilemma(records) -> list:
    return [
        r for r in records
        if (r["target_nash"] and r["target_utility"] and
            r["target_nash"] != r["target_utility"] and
            r["formal_game"] in GAME_TYPES)
    ]


def main() -> None:
    print("Loading GT-HarmBench training split …")
    records = load_gt_harmbench(DATA_CSV)
    sd_records = get_social_dilemma(records)
    print(f"  Social-dilemma subset: {len(sd_records)} rows")

    # ── Baseline ───────────────────────────────────────────────────────────────
    print("Computing baseline defection rates …")
    baseline_rates = compute_defection_rate(sd_records)
    baseline_overall = baseline_rates["Overall"]["defection_rate"]
    print(f"  Baseline overall defection rate: {baseline_overall:.4f}")
    for g in GAME_TYPES:
        r = baseline_rates[g]
        print(f"  {g}: {r['defection_rate']:.4f} "
              f"({r['n_defection']}/{r['n_total_with_nash']})")

    # ── Combined A+B+C intervention at τ=ρ=δ=1.0 ──────────────────────────────
    print("\nApplying combined A+B+C intervention (τ=ρ=δ=1.0) …")
    def comb_fn(R, C):
        return apply_all(R, C, PRIMARY_PARAM, PRIMARY_PARAM, PRIMARY_PARAM)

    combined_rates = compute_defection_rate(sd_records, intervention_fn=comb_fn)
    combined_overall = combined_rates["Overall"]["defection_rate"]
    cooperation_improvement_overall = baseline_overall - combined_overall
    print(f"  Combined defection rate: {combined_overall:.4f}  "
          f"(improvement: {cooperation_improvement_overall:+.4f})")
    for g in GAME_TYPES:
        r = combined_rates[g]
        rb = baseline_rates[g]
        impr = rb["defection_rate"] - r["defection_rate"]
        print(f"  {g}: {r['defection_rate']:.4f}  (Δ={impr:+.4f})")

    # ── Ablations: A only, B only, C only, A+B, A+C, B+C, A+B+C ──────────────
    def get_defect_nash(R, C):
        nash = pure_nash_equilibria(R, C)
        pareto = set(pareto_optimal_profiles(R, C))
        return [(i, j) for (i, j) in nash if (i, j) not in pareto]

    ablation_configs = {
        "A_only": lambda R, C:
            apply_intervention_a(R, C, PRIMARY_PARAM, get_defect_nash(R, C)),
        "B_only": lambda R, C:
            apply_intervention_b(R, C, PRIMARY_PARAM, get_defect_nash(R, C)),
        "C_only": lambda R, C:
            apply_intervention_c(R, C, PRIMARY_PARAM, social_welfare_maximising(R, C)),
        "A+B": lambda R, C: (
            lambda R2, C2: apply_intervention_b(R2, C2, PRIMARY_PARAM, get_defect_nash(R, C))
        )(*apply_intervention_a(R, C, PRIMARY_PARAM, get_defect_nash(R, C))),
        "A+C": lambda R, C: (
            lambda R2, C2: apply_intervention_c(R2, C2, PRIMARY_PARAM, social_welfare_maximising(R, C))
        )(*apply_intervention_a(R, C, PRIMARY_PARAM, get_defect_nash(R, C))),
        "B+C": lambda R, C: (
            lambda R2, C2: apply_intervention_c(R2, C2, PRIMARY_PARAM, social_welfare_maximising(R, C))
        )(*apply_intervention_b(R, C, PRIMARY_PARAM, get_defect_nash(R, C))),
        "A+B+C": comb_fn,
    }

    ablation_results = {}
    for name, fn in ablation_configs.items():
        rates = compute_defection_rate(sd_records, intervention_fn=fn)
        ablation_results[name] = {
            "defection_rate_overall": rates["Overall"]["defection_rate"],
            "cooperation_improvement": baseline_overall - rates["Overall"]["defection_rate"],
            "per_game": {
                g: {
                    "defection_rate": rates[g]["defection_rate"],
                    "improvement": baseline_rates[g]["defection_rate"] - rates[g]["defection_rate"],
                }
                for g in GAME_TYPES
            },
        }

    # ── τ sweep (Intervention A only) ─────────────────────────────────────────
    print("\nRunning τ sweep (Intervention A only, ρ=δ=0) …")
    tau_sweep = {}
    for tau in PARAM_SWEEP:
        def fn_a(R, C, t=tau):
            return apply_intervention_a(R, C, t, get_defect_nash(R, C))
        rates = compute_defection_rate(sd_records, intervention_fn=fn_a)
        tau_sweep[str(tau)] = {
            g: rates[g]["defection_rate"] for g in GAME_TYPES + ["Overall"]
        }

    # ── Success criterion ──────────────────────────────────────────────────────
    pd_improvement  = (baseline_rates["Prisoner's Dilemma"]["defection_rate"]
                       - combined_rates["Prisoner's Dilemma"]["defection_rate"])
    ch_improvement  = (baseline_rates["Chicken"]["defection_rate"]
                       - combined_rates["Chicken"]["defection_rate"])
    f3_demonstrated = (
        cooperation_improvement_overall > 0
        and pd_improvement > ch_improvement
    )

    print(f"\n  PD improvement: {pd_improvement:+.4f}  "
          f"Chicken improvement: {ch_improvement:+.4f}")
    print(f"  Success criterion: overall_improvement>0 AND PD>Chicken: {f3_demonstrated}")

    # ── Output JSON ────────────────────────────────────────────────────────────
    output = {
        "experiment": "f3_reciprocity",
        "dataset": "GT-HarmBench training split, 2009 rows (social-dilemma subset = 1528)",
        "status": "demonstrated" if f3_demonstrated else "inconclusive",
        "key_metric": {
            "name": "cooperation_improvement_overall_at_tau_rho_delta_1",
            "value": cooperation_improvement_overall,
            "threshold": 0.0,
            "passed": cooperation_improvement_overall > 0,
            "pd_improvement_exceeds_chicken": pd_improvement > ch_improvement,
            "f3_demonstrated": f3_demonstrated,
        },
        "framing": (
            "This experiment applies F3 as a payoff perturbation to existing 2×2 "
            "game matrices in the GT-HarmBench training split. It is an analytical "
            "claim about mechanism design, not an evaluation of LLM behavior."
        ),
        "caveats": [
            "Results are on the GT-HarmBench training split only. "
            "Test split evaluation is noted as future work.",
            "Parameters τ, ρ, δ are swept in {0.5, 1.0, 1.5, 2.0}; "
            "empirical calibration against real federation data is future work.",
            "Game matrices are integer-valued 2×2 games; the mechanism is modeled "
            "as additive perturbations to these matrices.",
            "Rows with any unparseable payoff field were excluded.",
        ],
        "baseline": {
            "overall_defection_rate": baseline_overall,
            "per_game": {
                g: baseline_rates[g]["defection_rate"] for g in GAME_TYPES
            },
        },
        "combined_a_b_c_tau_rho_delta_1": {
            "overall_defection_rate": combined_overall,
            "cooperation_improvement": cooperation_improvement_overall,
            "per_game": {
                g: {
                    "defection_rate": combined_rates[g]["defection_rate"],
                    "improvement": (baseline_rates[g]["defection_rate"]
                                    - combined_rates[g]["defection_rate"]),
                }
                for g in GAME_TYPES
            },
        },
        "ablation_table": ablation_results,
        "tau_sweep_intervention_a_only": tau_sweep,
        "figures": [
            str(OUT / "f3_reciprocity_bars.pdf"),
            str(OUT / "f3_reciprocity_bars.png"),
            str(OUT / "f3_reciprocity_sweep.pdf"),
            str(OUT / "f3_reciprocity_sweep.png"),
        ],
        "full_results": {
            "baseline_per_game_full": baseline_rates,
            "combined_per_game_full": combined_rates,
            "social_dilemma_n": len(sd_records),
        },
    }

    with open(OUT / "f3_reciprocity.json", "w") as fh:
        json.dump(output, fh, indent=2)

    # ── Figure 1: grouped bar chart — baseline vs combined by game type ────────
    game_labels = GAME_TYPES + ["Overall"]
    baseline_vals  = [baseline_rates[g]["defection_rate"] for g in game_labels]
    combined_vals  = [combined_rates[g]["defection_rate"] for g in game_labels]

    x = np.arange(len(game_labels))
    width = 0.35
    fig1, ax1 = plt.subplots(figsize=(5.4, 3.4))
    ax1.bar(x - width / 2, baseline_vals, width, label="Baseline", color="#7f7f7f")
    ax1.bar(x + width / 2, combined_vals, width, label="A+B+C (τ=ρ=δ=1.0)",
            color="#1f4e79")
    ax1.set_xticks(x)
    ax1.set_xticklabels([g.replace("Prisoner's ", "PD\n(").replace("Dilemma", ")")
                          .replace("Stag hunt", "Stag\nHunt")
                          .replace("Chicken", "Chicken")
                          .replace("Overall", "Overall")
                          for g in game_labels], fontsize=8)
    ax1.set_ylabel("Defection rate")
    ax1.set_title(
        "F3: Defection rate — baseline vs. A+B+C\nGT-HarmBench social-dilemma subset"
    )
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(OUT / "f3_reciprocity_bars.pdf", bbox_inches="tight")
    fig1.savefig(OUT / "f3_reciprocity_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ── Figure 2: cooperation improvement vs τ (Intervention A sweep) ─────────
    tau_vals = PARAM_SWEEP
    fig2, ax2 = plt.subplots(figsize=(5.4, 3.4))
    colors_game = {"Prisoner's Dilemma": "#1f4e79",
                   "Chicken": "#c00000",
                   "Stag hunt": "#7f7f7f"}
    for g in GAME_TYPES:
        baseline_g = baseline_rates[g]["defection_rate"]
        impr_vals = [
            baseline_g - tau_sweep[str(t)][g]
            for t in tau_vals
        ]
        ax2.plot(tau_vals, impr_vals, "o-", lw=1.4, ms=5,
                 color=colors_game[g],
                 label=g.replace("Prisoner's Dilemma", "PD"))
    ax2.axhline(0, color="#888888", lw=0.7, ls="--")
    ax2.set_xlabel("Trust-decay penalty τ (Intervention A only, ρ=δ=0)")
    ax2.set_ylabel("Cooperation improvement\n(baseline defection − post-intervention)")
    ax2.set_title(
        "F3: Cooperation improvement vs. τ (Intervention A sweep)\nGT-HarmBench social-dilemma subset"
    )
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT / "f3_reciprocity_sweep.pdf", bbox_inches="tight")
    fig2.savefig(OUT / "f3_reciprocity_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n=== F3: Reciprocity Mechanism (summary) ===")
    print(f"  {'Game type':<22}  {'Baseline':>9}  {'Combined':>9}  {'Improvement':>12}")
    print("  " + "-" * 58)
    for g in GAME_TYPES + ["Overall"]:
        br = baseline_rates[g]["defection_rate"]
        cr = combined_rates[g]["defection_rate"]
        print(f"  {g:<22}  {br:>9.4f}  {cr:>9.4f}  {br-cr:>+12.4f}")
    print()
    print(f"  Ablation summary (cooperation improvement at τ=ρ=δ=1.0):")
    for name, res in ablation_results.items():
        print(f"    {name:<8}: {res['cooperation_improvement']:+.4f}")
    print()
    status = "demonstrated" if f3_demonstrated else "INCONCLUSIVE"
    print(f"  F3 status: {status}")
    if not f3_demonstrated:
        print(
            "  Shortfall: either overall improvement ≤ 0 or "
            "PD improvement does not exceed Chicken improvement."
        )
    print(f"\nResults: {OUT}/f3_reciprocity.json")


if __name__ == "__main__":
    main()
