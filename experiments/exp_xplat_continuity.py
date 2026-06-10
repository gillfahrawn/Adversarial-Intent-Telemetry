#!/usr/bin/env python3
"""
LSH Continuity Recovery Experiment.
Evaluates structural continuity recovery across platform migration using synthetic fixtures.
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Utils ---

def jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    if not sig_a or not sig_b:
        return 0.0
    return sum(1 for x, y in zip(sig_a, sig_b) if x == y) / len(sig_a)

def cosine_sim(v_a: list[float], v_b: list[float]) -> float:
    a = np.array(v_a)
    b = np.array(v_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)

def euclidean_sim(v_a: list[float], v_b: list[float]) -> float:
    # Metric 3: Euclidean-based similarity
    a = np.array(v_a)
    b = np.array(v_b)
    dist = np.linalg.norm(a - b)
    return float(np.exp(-dist))

def any_band_match(sig_a: list[int], sig_b: list[int], b: int, r: int) -> bool:
    L = len(sig_a)
    if L != b * r:
        raise ValueError(f"Signature length {L} does not match b*r = {b}*{r}")
    for i in range(b):
        if sig_a[i*r : (i+1)*r] == sig_b[i*r : (i+1)*r]:
            return True
    return False

# --- Metrics ---

@dataclass
class RecoveryMetrics:
    fixture_id: str
    params: dict
    tau: int
    n_steps: int
    jaccard_vs_t0: list[float] = field(default_factory=list)
    lsh_match_vs_t0: list[bool] = field(default_factory=list)
    cosine_vs_t0: list[float] = field(default_factory=list)
    euclidean_vs_t0: list[float] = field(default_factory=list)
    
    def to_records(self, b: int, r: int):
        records = []
        for t in range(self.n_steps):
            records.append({
                "fixture_id": self.fixture_id,
                "timestep": t,
                "tau": self.tau,
                "is_post_migration": t >= self.tau,
                "jaccard": self.jaccard_vs_t0[t],
                "lsh_match": self.lsh_match_vs_t0[t],
                "cosine": self.cosine_vs_t0[t],
                "euclidean": self.euclidean_vs_t0[t],
                "b": b,
                "r": r,
                **self.params
            })
        return records

# --- Runner ---

class ContinuityExperiment:
    def __init__(self, fixtures_dir: Path, b: int, r: int):
        self.fixtures_dir = fixtures_dir
        self.b = b
        self.r = r
        self.L = b * r
        self.results = []

    def load_realization(self, rel_path: Path):
        with open(rel_path, "r") as f:
            return json.load(f)

    def run_on_manifest(self, manifest_path: Path):
        logger.info(f"Processing manifest: {manifest_path.name}")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        for entry in manifest["entries"]:
            rel_path = self.fixtures_dir / entry["realization"]
            if not rel_path.exists():
                logger.warning(f"Realization file not found: {rel_path}")
                continue
            
            data = self.load_realization(rel_path)
            self.evaluate_realization(data)

    def evaluate_realization(self, data: dict):
        fixture_id = data["fixture_id"]
        params = data["generator_params"]
        tau = data["migration"]["tau"]
        steps = data["steps"]
        n_steps = len(steps)
        
        sig0 = steps[0]["minhash_signature"]
        vec0 = steps[0]["structural_vector"]
        
        metrics = RecoveryMetrics(
            fixture_id=fixture_id,
            params=params,
            tau=tau,
            n_steps=n_steps
        )
        
        for t in range(n_steps):
            sigt = steps[t]["minhash_signature"]
            vect = steps[t]["structural_vector"]
            metrics.jaccard_vs_t0.append(jaccard(sig0, sigt))
            metrics.lsh_match_vs_t0.append(any_band_match(sig0, sigt, self.b, self.r))
            metrics.cosine_vs_t0.append(cosine_sim(vec0, vect))
            metrics.euclidean_vs_t0.append(euclidean_sim(vec0, vect))
            
        self.results.extend(metrics.to_records(self.b, self.r))

    def get_dataframe(self):
        return pd.DataFrame(self.results)

# --- Plotting ---

def plot_recovery_curves(df: pd.DataFrame, out_dir: Path, sweep_key: str):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="timestep", y="lsh_match", hue=sweep_key, marker="o")
    plt.title(f"Continuity Recovery (LSH) vs Timestep (Sweep: {sweep_key})")
    plt.ylabel("Recovery Rate (LSH Match)")
    plt.xlabel("Timestep (t)")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(title=sweep_key, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / f"xplat_continuity_sweep_{sweep_key}.png", dpi=150)
    plt.close()

def plot_jaccard_curves(df: pd.DataFrame, out_dir: Path, sweep_key: str):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="timestep", y="jaccard", hue=sweep_key, marker="o")
    plt.title(f"Mean Jaccard vs Timestep (Sweep: {sweep_key})")
    plt.ylabel("Mean Jaccard Similarity")
    plt.xlabel("Timestep (t)")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(title=sweep_key, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / f"xplat_jaccard_sweep_{sweep_key}.png", dpi=150)
    plt.close()

def plot_heatmap(df: pd.DataFrame, out_dir: Path, x_key: str, y_key: str, t_slice: int = None):
    # If t_slice is None, we just take the mean tau or a fixed step
    if t_slice is None:
        t_slice = int(df["tau"].median())
    
    subset = df[df["timestep"] == t_slice]
    pivot = subset.groupby([y_key, x_key])["lsh_match"].mean().unstack()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title(f"Recovery Rate at t={t_slice} over ({x_key}, {y_key})")
    plt.tight_layout()
    plt.savefig(out_dir / f"xplat_continuity_heatmap_{x_key}_{y_key}.png", dpi=150)
    plt.close()

def plot_comparison_curves(df: pd.DataFrame, out_dir: Path):
    # Compare LSH Match vs Raw Cosine (normalized to [0,1])
    plt.figure(figsize=(10, 6))
    subset = df.groupby("timestep").agg({"lsh_match": "mean", "cosine": "mean"}).reset_index()
    plt.plot(subset["timestep"], subset["lsh_match"], label="LSH Recovery (b=8,r=4)", marker="o")
    plt.plot(subset["timestep"], subset["cosine"], label="Latent Cosine Similarity", marker="s")
    plt.title("LSH Decoupling Diagnostic: LSH Match vs Latent Cosine")
    plt.ylabel("Recovery / Similarity")
    plt.xlabel("Timestep (t)")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "lsh_decoupling_diagnostic.png", dpi=150)
    plt.close()

def generate_summary(df: pd.DataFrame, out_dir: Path):
    # Compute summary metrics per (sweep parameters + b, r)
    group_keys = ["b", "r", "invariant_ratio", "perturbation_magnitude", "entropy_growth_rate", "tau"]
    # Filter to only keys present in df
    group_keys = [k for k in group_keys if k in df.columns]
    
    summary_rows = []
    # FIX 4: tau-conditioned grouping
    for keys, group in df.groupby(group_keys, as_index=False):
        tau = int(group["tau"].iloc[0])
        max_t = int(group["timestep"].max())
        
        pre_mig = group[group["timestep"] == tau - 1]["jaccard"].mean()
        at_mig = group[group["timestep"] == tau]["jaccard"].mean()
        post_mig = group[group["timestep"] == max_t]["jaccard"].mean()
        
        # Explicit Metrics:
        mig_shock_recovery = group[group["timestep"] == tau]["lsh_match"].mean()
        rec_post_mig = group[group["timestep"] == max_t]["lsh_match"].mean()
        drift_decay = mig_shock_recovery - rec_post_mig
        
        row = dict(zip(group_keys, keys)) if len(group_keys) > 1 else {group_keys[0]: keys}
        row.update({
            "mean_jaccard_at_migration": at_mig,
            "migration_shock_recovery": mig_shock_recovery,
            "recovery_rate_stabilized": rec_post_mig,
            "drift_decay": drift_decay,
            "cosine_at_migration": group[group["timestep"] == tau]["cosine"].mean(),
            "cosine_stabilized": group[group["timestep"] == max_t]["cosine"].mean(),
            "euclidean_at_migration": group[group["timestep"] == tau]["euclidean"].mean(),
            "euclidean_stabilized": group[group["timestep"] == max_t]["euclidean"].mean()
        })
        summary_rows.append(row)
        
    summary_df = pd.DataFrame(summary_rows)
    
    # FIX 4: Diagnostics - correlation check
    if "entropy_growth_rate" in summary_df.columns:
        corr = summary_df["entropy_growth_rate"].corr(summary_df["drift_decay"])
        logger.info(f"CAUSALITY DIAGNOSTIC: correlation(entropy_growth_rate, drift_decay) = {corr:.4f}")
        with open(out_dir / "causality_diagnostic.txt", "w") as f:
            f.write(f"correlation(entropy_growth_rate, drift_decay): {corr:.4f}\n")

    summary_df.to_csv(out_dir / "continuity_summary.csv", index=False)
    logger.info(f"Saved summary metrics to {out_dir / 'continuity_summary.csv'}")

def plot_phase_diagram(df: pd.DataFrame, out_dir: Path, x_key: str, y_key: str, metric_key: str = "lsh_match"):
    # Slice at stabilization (max timestep) to show "stability"
    max_t = df["timestep"].max()
    subset = df[df["timestep"] == max_t]
    
    # Average metric across realizations for each (x, y) point
    pivot = subset.groupby([y_key, x_key])[metric_key].mean().unstack()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, cmap="magma", vmin=0, vmax=1)
    plt.title(f"2D Phase Diagram: Stability ({metric_key}) at t={max_t}\nAcross {x_key} × {y_key}")
    plt.tight_layout()
    plt.savefig(out_dir / f"xplat_phase_diagram_{x_key}_{y_key}.png", dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Continuity Recovery Experiment")
    parser.add_argument("--fixtures-dir", type=Path, default=Path("experiments/fixtures/generated"), help="Dir containing realizations")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results"), help="Dir for output artifacts")
    parser.add_argument("--b", type=int, default=8, help="LSH bands")
    parser.add_argument("--r", type=int, default=4, help="LSH rows per band")
    parser.add_argument("--sweep-keys", type=str, default="invariant_ratio,perturbation_magnitude,entropy_growth_rate", help="Comma-separated keys to generate line plots for")
    parser.add_argument("--heatmap-keys", type=str, default="invariant_ratio,perturbation_magnitude", help="Comma-separated keys for default heatmaps")
    
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    exp = ContinuityExperiment(args.fixtures_dir, args.b, args.r)
    
    # Process all manifests in the fixtures directory
    manifests = list(args.fixtures_dir.glob("manifest_*.json"))
    if not manifests:
        logger.error(f"No manifests found in {args.fixtures_dir}")
        return
    
    for m in manifests:
        exp.run_on_manifest(m)
    
    df = exp.get_dataframe()
    if df.empty:
        logger.error("No results generated.")
        return
    
    # Save CSV
    df.to_csv(args.out_dir / "continuity_metrics.csv", index=False)
    logger.info(f"Saved metrics to {args.out_dir / 'continuity_metrics.csv'}")
    
    # Generate Summary
    generate_summary(df, args.out_dir)
    
    # Generate Comparison Plot (Decoupling Diagnostic)
    plot_comparison_curves(df, args.out_dir)
    
    # Generate Plots
    sweep_keys = [k.strip() for k in args.sweep_keys.split(",")]
    for sk in sweep_keys:
        if sk in df.columns and df[sk].nunique() > 1:
            logger.info(f"Generating line plots for sweep: {sk}")
            plot_recovery_curves(df, args.out_dir, sk)
            plot_jaccard_curves(df, args.out_dir, sk)
    
    # Default heatmaps
    h_keys = [k.strip() for k in args.heatmap_keys.split(",")]
    if all(k in df.columns for k in h_keys) and len(h_keys) == 2:
        logger.info(f"Generating default heatmap for {h_keys}")
        tau = int(df["tau"].iloc[0])
        plot_heatmap(df, args.out_dir, h_keys[0], h_keys[1], t_slice=tau)
    
    # New Phase Diagram: invariant_ratio x entropy_growth_rate
    if "invariant_ratio" in df.columns and "entropy_growth_rate" in df.columns:
        if df["invariant_ratio"].nunique() > 1 and df["entropy_growth_rate"].nunique() > 1:
            logger.info("Generating phase diagram for invariant_ratio x entropy_growth_rate")
            plot_phase_diagram(df, args.out_dir, "invariant_ratio", "entropy_growth_rate")



if __name__ == "__main__":
    main()
