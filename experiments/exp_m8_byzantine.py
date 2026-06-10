#!/usr/bin/env python3
"""
Experiment M8: Byzantine Tolerance β* ≥ 1/3

Validates empirically that the trust-decay EMA + Hoeffding isolation mechanism
maintains network precision above baseline − ε*(n, δ) when a fraction β of
federation senders are Byzantine (miscalibrated). Pure simulation; no external
data required.

Paper: Sec. 9, Eq. (4)–(6).
Parameters validated: λ=0.1 (EMA smoothing), n=500, δ=1e-3 → ε*≈0.083,
operating target β* ≥ 1/3.
Status upgrade target: hypothesized → demonstrated.
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 20260514

# ── Output ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
OUT = SCRIPT_DIR / "results"
OUT.mkdir(parents=True, exist_ok=True)

# ── Protocol parameters (Sec. 9) ───────────────────────────────────────────────
LAMBDA = 0.1              # EMA smoothing factor (Eq. 4)
N_SENDERS = 20            # total senders in the network
P_HONEST = 0.85           # honest sender precision
M_OBS = 10                # observations per sender per timestep
T_TOTAL = 600             # total timesteps
T_WARMUP = 100            # warm-up: no isolation before this timestep

N_HOEFFDING = 500         # n in Hoeffding bound (Eq. 5)
DELTA_HOEFFDING = 1e-3    # δ in Hoeffding bound (Eq. 5)

N_TRIALS = 100            # independent trials per β value
BETA_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# ── Derived constants ───────────────────────────────────────────────────────────
# ε*(n; δ) = sqrt(ln(1/δ) / 2n)   (Eq. 6)
EPS_STAR = math.sqrt(math.log(1.0 / DELTA_HOEFFDING) / (2.0 * N_HOEFFDING))
# A trial degrades if mean network precision falls below this level
PREC_THRESHOLD = P_HONEST - EPS_STAR   # 0.85 − 0.083 = 0.767


def run_trial(n_byzantine: int, rng: np.random.Generator) -> dict:
    """
    Simulate one trial: N_SENDERS senders (n_byzantine Byzantine, rest honest).
    Byzantine senders draw their true precision from Uniform(0.20, 0.50) once
    per trial.

    Returns a dict with:
      mean_prec           : mean network precision over timesteps [T_WARMUP, T_TOTAL]
      byz_isolation_steps : list of length n_byzantine — the timestep at which each
                            Byzantine sender was first isolated, or T_TOTAL if never.
    """
    # True precisions: Byzantine senders first, honest last
    p_true = np.empty(N_SENDERS)
    if n_byzantine > 0:
        p_true[:n_byzantine] = rng.uniform(0.20, 0.50, size=n_byzantine)
    p_true[n_byzantine:] = P_HONEST

    w = np.full(N_SENDERS, 0.5)
    isolated = np.zeros(N_SENDERS, dtype=bool)
    # Track first isolation timestep for each Byzantine sender; None = not yet isolated
    byz_iso_step: list[int | None] = [None] * n_byzantine
    prec_hist: list[float] = []

    for t in range(T_TOTAL):
        # EMA update for non-isolated senders
        for j in range(N_SENDERS):
            if not isolated[j]:
                batch = rng.binomial(1, p_true[j], size=M_OBS)
                p_hat_j = batch.mean()
                w[j] = (1.0 - LAMBDA) * w[j] + LAMBDA * p_hat_j

        # Hoeffding isolation after warm-up (Eq. 5–6)
        if t >= T_WARMUP:
            active = ~isolated
            if active.any():
                mean_w_active = w[active].mean()
                for j in range(N_SENDERS):
                    if not isolated[j] and w[j] < mean_w_active - EPS_STAR:
                        isolated[j] = True
                        if j < n_byzantine and byz_iso_step[j] is None:
                            byz_iso_step[j] = t   # record first isolation timestep

        # Network precision after warm-up (over non-isolated senders)
        if t >= T_WARMUP:
            active = ~isolated
            if active.any():
                w_sum = w[active].sum()
                if w_sum > 0.0:
                    prec = (w[active] * p_true[active]).sum() / w_sum
                    prec_hist.append(float(prec))

    # Replace None (never isolated) with T_TOTAL
    byz_iso_step_final = [s if s is not None else T_TOTAL for s in byz_iso_step]
    return {
        "mean_prec": float(np.mean(prec_hist)) if prec_hist else 0.0,
        "byz_isolation_steps": byz_iso_step_final,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)

    results_by_beta: dict[float, dict] = {}
    for beta in BETA_SWEEP:
        n_byz = math.floor(beta * N_SENDERS)
        trials = [run_trial(n_byz, rng) for _ in range(N_TRIALS)]
        mean_precs = np.array([t["mean_prec"] for t in trials])
        degraded = int((mean_precs < PREC_THRESHOLD).sum())

        # Mean isolation timestep: average over all Byzantine senders in all trials
        all_iso_steps: list[int] = []
        for t in trials:
            all_iso_steps.extend(t["byz_isolation_steps"])
        mean_iso_step = float(np.mean(all_iso_steps)) if all_iso_steps else float("nan")

        results_by_beta[beta] = {
            "beta": beta,
            "n_byzantine": n_byz,
            "degradation_rate": degraded / N_TRIALS,
            "mean_prec_net": float(mean_precs.mean()),
            "std_prec_net": float(mean_precs.std()),
            "n_degraded_trials": degraded,
            "mean_isolation_step": mean_iso_step,
        }

    # β* = largest β where degradation_rate < 0.05
    beta_star = 0.0
    for beta in BETA_SWEEP:
        if results_by_beta[beta]["degradation_rate"] < 0.05:
            beta_star = beta

    m8_demonstrated = beta_star >= 1.0 / 3.0

    # ── Output JSON ────────────────────────────────────────────────────────────
    output = {
        "experiment": "m8_byzantine",
        "dataset": "synthetic simulation — no external data",
        "status": "demonstrated" if m8_demonstrated else "inconclusive",
        "key_metric": {
            "name": "empirical_beta_star",
            "value": beta_star,
            "threshold": 1.0 / 3.0,
            "passed": m8_demonstrated,
            "note": "largest β where degradation_rate < 0.05",
        },
        "parameters": {
            "N_senders": N_SENDERS,
            "lambda_ema": LAMBDA,
            "p_honest": P_HONEST,
            "m_obs_per_timestep": M_OBS,
            "T_total": T_TOTAL,
            "T_warmup": T_WARMUP,
            "n_hoeffding": N_HOEFFDING,
            "delta_hoeffding": DELTA_HOEFFDING,
            "eps_star": EPS_STAR,
            "prec_threshold": PREC_THRESHOLD,
            "n_trials_per_beta": N_TRIALS,
            "seed": SEED,
        },
        "caveats": [
            "Parameters λ=0.1, n=500, δ=1e-3 are hypothesized in the paper (Sec. 9); "
            "this experiment validates the mechanism under those exact parameters.",
            "Byzantine senders draw precision from Uniform(0.20, 0.50); "
            "adversarial Byzantine behavior in a real deployment may differ "
            "(e.g., adaptive, coordinated).",
            "Network topology is fully connected (star: all senders → one receiver); "
            "real deployments have partial observability.",
        ],
        "figures": [
            str(OUT / "m8_byzantine.pdf"),
            str(OUT / "m8_byzantine.png"),
        ],
        "full_results": {
            "eps_star": EPS_STAR,
            "prec_threshold": PREC_THRESHOLD,
            "empirical_beta_star": beta_star,
            "beta_star_geq_one_third": m8_demonstrated,
            "beta_sweep": list(results_by_beta.values()),
            "isolation_diagnostic": {
                "description": (
                    "mean_isolation_step is the mean timestep at which a Byzantine "
                    "sender is first isolated, averaged over all Byzantine senders "
                    "across all 100 trials. Values near T_WARMUP (100) indicate "
                    "fast isolation; values near T_TOTAL (600) indicate the sender "
                    "was never isolated."
                ),
                "per_beta": [
                    {"beta": b, "mean_isolation_step": results_by_beta[b]["mean_isolation_step"]}
                    for b in BETA_SWEEP
                ],
            },
        },
    }

    with open(OUT / "m8_byzantine.json", "w") as fh:
        json.dump(output, fh, indent=2)

    # ── Figure ─────────────────────────────────────────────────────────────────
    betas = BETA_SWEEP
    deg_rates  = [results_by_beta[b]["degradation_rate"]   for b in betas]
    iso_steps  = [results_by_beta[b]["mean_isolation_step"] for b in betas]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))

    # Primary axis: degradation rate
    color_deg = "#1f4e79"
    ax.plot(betas, deg_rates, "o-", color=color_deg, lw=1.5, ms=5,
            label="degradation rate (left axis)")
    ax.axvline(1.0 / 3.0, color="#c00000", lw=1.0, ls="--", alpha=0.8,
               label="β = 1/3 (paper target)")
    ax.axhline(0.05, color="#7f7f7f", lw=0.8, ls="--", alpha=0.7,
               label="degradation threshold = 0.05")
    if m8_demonstrated:
        ax.axvline(beta_star, color=color_deg, lw=0.9, ls=":",
                   label=f"β* = {beta_star:.2f} (empirical)")
    ax.set_xlabel("Byzantine fraction β")
    ax.set_ylabel("Degradation rate", color=color_deg)
    ax.tick_params(axis="y", labelcolor=color_deg)
    ax.set_xlim(-0.01, 0.55)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)

    # Secondary axis: mean isolation timestep
    color_iso = "#c07000"
    ax2 = ax.twinx()
    # Filter out nan (β=0 has no Byzantine senders)
    valid_iso = [(b, s) for b, s in zip(betas, iso_steps) if not math.isnan(s)]
    if valid_iso:
        bx, sx = zip(*valid_iso)
        ax2.plot(bx, sx, "s--", color=color_iso, lw=1.2, ms=4,
                 label="mean isolation step (right axis)")
        ax2.axhline(T_WARMUP, color=color_iso, lw=0.6, ls=":", alpha=0.5,
                    label=f"T_warmup = {T_WARMUP}")
    ax2.set_ylabel("Mean Byzantine isolation timestep", color=color_iso)
    ax2.tick_params(axis="y", labelcolor=color_iso)
    ax2.set_ylim(T_WARMUP - 20, T_TOTAL + 20)

    # Combined legend
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=6.5, frameon=False,
              loc="upper left")

    ax.set_title(
        f"M8: Hoeffding isolation — Byzantine tolerance + isolation timing\n"
        f"N={N_SENDERS}, λ={LAMBDA}, ε*={EPS_STAR:.3f}, "
        f"{N_TRIALS} trials/β, threshold={PREC_THRESHOLD:.3f}"
    )
    fig.tight_layout()
    fig.savefig(OUT / "m8_byzantine.pdf", bbox_inches="tight")
    fig.savefig(OUT / "m8_byzantine.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Summary table ──────────────────────────────────────────────────────────
    print("=== M8: Byzantine Tolerance (EMA + Hoeffding Isolation) ===")
    print(f"  ε* = sqrt(ln(1/δ) / 2n) = sqrt(ln(1/{DELTA_HOEFFDING:.0e}) / {2*N_HOEFFDING})"
          f" ≈ {EPS_STAR:.4f}")
    print(f"  Precision threshold = {P_HONEST} − {EPS_STAR:.4f} = {PREC_THRESHOLD:.4f}")
    print(f"  N={N_SENDERS} senders, λ={LAMBDA}, {N_TRIALS} trials/β\n")
    print(f"  {'β':>6}  {'n_byz':>6}  {'deg_rate':>10}  {'mean_prec':>10}  {'iso_step':>10}")
    print("  " + "-" * 54)
    for beta in BETA_SWEEP:
        r = results_by_beta[beta]
        iso = r["mean_isolation_step"]
        iso_str = f"{iso:>10.1f}" if not math.isnan(iso) else "       n/a"
        print(f"  {beta:>6.2f}  {r['n_byzantine']:>6}  "
              f"{r['degradation_rate']:>10.4f}  {r['mean_prec_net']:>10.4f}  {iso_str}")
    print()
    print(f"  Empirical β* = {beta_star:.2f}  (paper target: 1/3 = {1/3:.4f})")
    status = "demonstrated" if m8_demonstrated else "INCONCLUSIVE"
    print(f"  M8 status: {status}")
    print(f"\nResults written to: {OUT}/m8_byzantine.json")


if __name__ == "__main__":
    main()
