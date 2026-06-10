#!/usr/bin/env python3
"""
Experiment M8: Sequential Probability Ratio Test (SPRT) for Byzantine Isolation

Validates the SPRT mechanism against Obvious and Stealth adversary models.
Compares SPRT performance with a Hoeffding-bound baseline.

Specification:
H0: p0 = 0.85 (honest)
H1: p1 = 0.767 (Byzantine, p0 - eps_star)
alpha = beta = 0.05
"""

import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
SEED = 20260514
T_TOTAL = 600
N_SENDERS = 500
N_TRIALS = 50

P_BAR = 0.85
EPS_STAR = 0.083
P0 = P_BAR
P1 = P_BAR - EPS_STAR  # 0.767

ALPHA_ERR = 0.05
BETA_ERR = 0.05

# Wald SPRT thresholds
THRESHOLD_LOWER = math.log(BETA_ERR / (1 - ALPHA_ERR))  # log(0.05/0.95) ≈ -2.944
THRESHOLD_UPPER = math.log((1 - BETA_ERR) / ALPHA_ERR)  # log(0.95/0.05) ≈ 2.944

# Hoeffding parameters (from exp_m8_byzantine.py)
LAMBDA = 0.1
T_WARMUP = 100
# Note: exp_m8_byzantine.py uses N_HOEFFDING=500 and DELTA=1e-3 to get EPS_STAR=0.083

# Output paths
SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def llr(x, p0, p1):
    """Log-likelihood ratio for Bernoulli(x)"""
    if x == 1:
        return math.log(p1 / p0)
    else:
        return math.log((1 - p1) / (1 - p0))

def run_simulation(adversary_type, beta, rng, use_sprt=True, use_hoeffding=False):
    """
    Runs a single trial simulation.
    adversary_type: 'obvious' or 'stealth'
    beta: fraction of Byzantine senders
    """
    n_byz = int(beta * N_SENDERS)
    p_true = np.full(N_SENDERS, P_BAR)
    
    if adversary_type == 'obvious':
        p_adv = P_BAR - 2 * EPS_STAR  # 0.684
    elif adversary_type == 'stealth':
        p_adv = P_BAR - EPS_STAR - 0.01  # 0.757
    else:
        raise ValueError("Unknown adversary type")
    
    p_true[:n_byz] = p_adv
    
    # SPRT states
    lambda_cum = np.zeros(N_SENDERS)
    isolated_sprt = np.zeros(N_SENDERS, dtype=bool)
    exonerated_sprt = np.zeros(N_SENDERS, dtype=bool)
    isolation_step_sprt = np.full(N_SENDERS, T_TOTAL)
    
    # Hoeffding states
    w_hoeffding = np.full(N_SENDERS, 0.5)
    isolated_hoeffding = np.zeros(N_SENDERS, dtype=bool)
    isolation_step_hoeffding = np.full(N_SENDERS, T_TOTAL)
    
    prec_hist = []
    
    # Precompute log-likelihood ratios for obs=0 and obs=1
    # We use llr = log(P(Honest)/P(Byzantine)) to match the prompt's thresholds:
    # Lambda increases for honest (Exonerate at +2.944)
    # Lambda decreases for Byzantine (Isolate at -2.944)
    llr_0 = math.log((1 - P0) / (1 - P1))
    llr_1 = math.log(P0 / P1)
    
    for t in range(T_TOTAL):
        # Observations for all senders
        obs = rng.binomial(1, p_true)
        
        # Update SPRT
        if use_sprt:
            active_sprt = ~(isolated_sprt | exonerated_sprt)
            if active_sprt.any():
                # Apply llr based on observations
                current_llrs = np.where(obs == 1, llr_1, llr_0)
                lambda_cum[active_sprt] += current_llrs[active_sprt]
                
                # Check thresholds
                new_isolated = active_sprt & (lambda_cum <= THRESHOLD_LOWER)
                isolated_sprt[new_isolated] = True
                isolation_step_sprt[new_isolated] = t
                
                new_exonerated = active_sprt & (lambda_cum >= THRESHOLD_UPPER)
                exonerated_sprt[new_exonerated] = True
        
        # Update Hoeffding
        if use_hoeffding:
            # EMA update for all non-isolated Hoeffding senders
            active_hoeff = ~isolated_hoeffding
            if active_hoeff.any():
                w_hoeffding[active_hoeff] = (1.0 - LAMBDA) * w_hoeffding[active_hoeff] + LAMBDA * obs[active_hoeff]
            
            if t >= T_WARMUP:
                active_hoeff = ~isolated_hoeffding
                if active_hoeff.any():
                    mean_w = w_hoeffding[active_hoeff].mean()
                    new_isolated_hoeff = active_hoeff & (w_hoeffding < mean_w - EPS_STAR)
                    isolated_hoeffding[new_isolated_hoeff] = True
                    isolation_step_hoeffding[new_isolated_hoeff] = t
        
        # Measure network precision (using SPRT isolation for degradation reporting)
        # Requirement: "network_precision_degradation"
        # We'll use the active senders based on whichever test we are focusing on.
        # If both are used, we report for SPRT.
        active = ~isolated_sprt if use_sprt else ~isolated_hoeffding
        if active.any():
            prec_hist.append(p_true[active].mean())
        else:
            prec_hist.append(0.0)
            
    # Metrics
    def get_metrics(isolated, iso_steps):
        byz_isolated = isolated[:n_byz]
        hon_isolated = isolated[n_byz:]
        
        isolation_rate = byz_isolated.mean() if n_byz > 0 else 0.0
        false_isolation_rate = hon_isolated.mean() if (N_SENDERS - n_byz) > 0 else 0.0
        
        # mean_isolation_step for Byzantine senders
        if n_byz > 0:
            mean_iso_step = iso_steps[:n_byz].mean()
        else:
            mean_iso_step = 0.0
            
        return isolation_rate, false_isolation_rate, mean_iso_step

    sprt_metrics = get_metrics(isolated_sprt, isolation_step_sprt) if use_sprt else (0,0,0)
    hoeff_metrics = get_metrics(isolated_hoeffding, isolation_step_hoeffding) if use_hoeffding else (0,0,0)
    
    avg_prec = np.mean(prec_hist)
    degradation = P_BAR - avg_prec
    
    return {
        "sprt": sprt_metrics,
        "hoeffding": hoeff_metrics,
        "degradation": degradation
    }

def main():
    rng = np.random.default_rng(SEED)
    
    # 1. Obvious Adversary Sweep
    beta_sweep_obvious = np.arange(0.05, 0.55, 0.05)
    results_obvious_sprt = []
    results_obvious_hoeffding = []
    
    print("Running Obvious Adversary Sweep...")
    for beta in beta_sweep_obvious:
        trial_sprt_metrics = []
        trial_hoeff_metrics = []
        trial_degradations = []
        
        for _ in range(N_TRIALS):
            res = run_simulation('obvious', beta, rng, use_sprt=True, use_hoeffding=True)
            trial_sprt_metrics.append(res['sprt'])
            trial_hoeff_metrics.append(res['hoeffding'])
            trial_degradations.append(res['degradation'])
            
        # Aggregate
        def aggregate(metrics_list):
            m = np.array(metrics_list)
            return {
                "beta": float(beta),
                "isolation_rate": float(m[:, 0].mean()),
                "false_isolation_rate": float(m[:, 1].mean()),
                "mean_isolation_step": float(m[:, 2].mean()),
            }
        
        agg_sprt = aggregate(trial_sprt_metrics)
        agg_sprt["network_precision_degradation"] = float(np.mean(trial_degradations))
        results_obvious_sprt.append(agg_sprt)
        
        agg_hoeff = aggregate(trial_hoeff_metrics)
        # Note: degradation is reported based on SPRT isolation in run_simulation if both used.
        # But for Hoeffding baseline, we might want its own degradation. 
        # For simplicity and given the prompt requirements, we'll focus on the comparison of speed and rates.
        results_obvious_hoeffding.append(agg_hoeff)

    # 2. Stealth Adversary Sweep
    beta_sweep_stealth = np.arange(0.05, 0.45, 0.05)
    results_stealth_sprt = []
    
    print("Running Stealth Adversary Sweep...")
    for beta in beta_sweep_stealth:
        trial_sprt_metrics = []
        trial_degradations = []
        
        for _ in range(N_TRIALS):
            res = run_simulation('stealth', beta, rng, use_sprt=True, use_hoeffding=False)
            trial_sprt_metrics.append(res['sprt'])
            trial_degradations.append(res['degradation'])
            
        agg_sprt = aggregate(trial_sprt_metrics)
        agg_sprt["network_precision_degradation"] = float(np.mean(trial_degradations))
        results_stealth_sprt.append(agg_sprt)

    # ── Summary Calculations ───────────────────────────────────────────────────
    # sprt_vs_hoeffding_speedup_factor for obvious adversary
    # We'll take the average speedup across beta sweep where isolation rate > 0.5
    speedups = []
    for s, h in zip(results_obvious_sprt, results_obvious_hoeffding):
        if s['isolation_rate'] > 0.5 and h['mean_isolation_step'] > 0:
            # Lower step is faster
            speedups.append(h['mean_isolation_step'] / s['mean_isolation_step'])
    speedup_factor = float(np.mean(speedups)) if speedups else 0.0
    
    # stealth_operational_beta_star: max beta where isolation_rate > 0.8
    beta_star = 0.0
    for r in results_stealth_sprt:
        if r['isolation_rate'] > 0.8:
            beta_star = r['beta']
            
    # max false isolation rate
    max_fir = max([r['false_isolation_rate'] for r in results_obvious_sprt] + 
                  [r['false_isolation_rate'] for r in results_stealth_sprt])

    output = {
        "sprt_params": {
            "p_bar": P_BAR, "eps_star": EPS_STAR, "p0": P0, "p1": P1,
            "alpha_err": ALPHA_ERR, "beta_err": BETA_ERR,
            "threshold_lower": THRESHOLD_LOWER, "threshold_upper": THRESHOLD_UPPER
        },
        "obvious_adversary": {
            "sprt": results_obvious_sprt,
            "hoeffding": results_obvious_hoeffding
        },
        "stealth_adversary": {
            "sprt": results_stealth_sprt
        },
        "summary": {
            "sprt_vs_hoeffding_speedup_factor": speedup_factor,
            "stealth_operational_beta_star": float(beta_star),
            "sprt_false_isolation_rate_max": float(max_fir)
        }
    }

    with open(OUT_DIR / "m8_sprt.json", "w") as f:
        json.dump(output, f, indent=2)

    # ── Plotting ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Obvious Adversary Speed Comparison
    betas_obv = [r['beta'] for r in results_obvious_sprt]
    steps_sprt = [r['mean_isolation_step'] for r in results_obvious_sprt]
    steps_hoeff = [r['mean_isolation_step'] for r in results_obvious_hoeffding]
    
    ax1.plot(betas_obv, steps_sprt, 'o-', label='SPRT')
    ax1.plot(betas_obv, steps_hoeff, 's--', label='Hoeffding')
    ax1.set_title("Obvious Adversary: Mean Isolation Step")
    ax1.set_xlabel("Byzantine Fraction (beta)")
    ax1.set_ylabel("Mean Step to Isolation")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Stealth Adversary Performance
    betas_stl = [r['beta'] for r in results_stealth_sprt]
    ir_stl = [r['isolation_rate'] for r in results_stealth_sprt]
    fir_stl = [r['false_isolation_rate'] for r in results_stealth_sprt]
    
    ax2.plot(betas_stl, ir_stl, 'o-', label='Isolation Rate')
    ax2.plot(betas_stl, fir_stl, 'x-', label='False Isolation Rate')
    ax2.axhline(0.05, color='red', linestyle=':', label='Alpha=0.05')
    ax2.set_title("Stealth Adversary: SPRT Performance")
    ax2.set_xlabel("Byzantine Fraction (beta)")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(OUT_DIR / "m8_sprt.png")
    fig.savefig(OUT_DIR / "m8_sprt.pdf")
    plt.close()

    print(f"Results saved to {OUT_DIR}/m8_sprt.json")
    print(f"Figures saved to {OUT_DIR}/m8_sprt.png and m8_sprt.pdf")

if __name__ == "__main__":
    main()
