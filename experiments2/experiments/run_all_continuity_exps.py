#!/usr/bin/env python3
"""
Orchestrator for LSH Continuity Recovery Experiments.
Runs generations for multiple sweeps and then executes the evaluation pipeline.
"""

import subprocess
import os
import shutil
from pathlib import Path

# Paths
BASE_DIR = Path("Adversarial-Intent-Telemetry")
FIXTURES_GEN = BASE_DIR / "experiments" / "fixtures" / "generate.py"
EVAL_SCRIPT = BASE_DIR / "experiments" / "exp_xplat_continuity.py"
OUT_DIR = BASE_DIR / "experiments" / "fixtures" / "generated"
RESULTS_DIR = BASE_DIR / "experiments" / "results"

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    # Clear old generated data
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: Generating Sweeps ===")
    
    # 1. Invariant Ratio Sweep
    run_cmd([
        "python3", str(FIXTURES_GEN),
        "--out-dir", str(OUT_DIR),
        "--id-prefix", "sweep_ir",
        "--sweep", "invariant_ratio=0.2,0.4,0.6,0.8,1.0",
        "--n", "10" 
    ])
    
    # 2. Perturbation Magnitude Sweep
    run_cmd([
        "python3", str(FIXTURES_GEN),
        "--out-dir", str(OUT_DIR),
        "--id-prefix", "sweep_pm",
        "--sweep", "perturbation_magnitude=0.0,0.2,0.4,0.6,0.8",
    ])
    
    # 3. Entropy Growth Rate Sweep
    run_cmd([
        "python3", str(FIXTURES_GEN),
        "--out-dir", str(OUT_DIR),
        "--id-prefix", "sweep_eg",
        "--sweep", "entropy_growth_rate=0.0,0.1,0.2,0.3,0.4",
    ])
    
    # 4. 2D Sweep for Heatmap (IR x PM)
    for ir in [0.2, 0.4, 0.6, 0.8]:
        for pm in [0.1, 0.3, 0.5, 0.7]:
            run_cmd([
                "python3", str(FIXTURES_GEN),
                "--out-dir", str(OUT_DIR),
                "--id-prefix", f"hmap_ir{int(ir*10)}_pm{int(pm*10)}",
                "--invariant-ratio", str(ir),
                "--perturbation-magnitude", str(pm),
                "--n", "5"
            ])

    # 5. 2D Sweep for Phase Diagram (IR x EG)
    print("\n=== Phase 1.5: Generating Phase Diagram Sweep (IR x EG) ===")
    for ir in [0.2, 0.4, 0.6, 0.8]:
        for eg in [0.0, 0.1, 0.2, 0.3, 0.4]:
            run_cmd([
                "python3", str(FIXTURES_GEN),
                "--out-dir", str(OUT_DIR),
                "--id-prefix", f"phase_ir{int(ir*10)}_eg{int(eg*10)}",
                "--invariant-ratio", str(ir),
                "--entropy-growth-rate", str(eg),
                "--n", "5"
            ])

    print("\n=== Phase 2: Running Evaluation ===")
    
    # Run evaluation on everything
    # Using b=8, r=4 (L=32)
    run_cmd([
        "python3", str(EVAL_SCRIPT),
        "--fixtures-dir", str(OUT_DIR),
        "--out-dir", str(RESULTS_DIR),
        "--b", "8",
        "--r", "4",
        "--sweep-keys", "invariant_ratio,perturbation_magnitude,entropy_growth_rate"
    ])

    print(f"\nExperiment complete. Artifacts in {RESULTS_DIR}")

if __name__ == "__main__":
    main()
