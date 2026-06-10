#!/usr/bin/env python3
"""Parameterized generator for cross-platform synthetic trajectory fixtures.

Produces fixtures schema-compatible with:
  experiments/fixtures/synth_traj_xplat_001.json              (latent trajectory)
  experiments/fixtures/synth_traj_xplat_001_realization.json  (per-step telemetry)

Five explicit knobs:
  --tau                     migration timestep
  --invariant-ratio         fraction of MinHash positions held cross-platform
  --perturbation-magnitude  scale of platform-conditioned 4-d feature shift
  --entropy-growth-rate     per-step entropy slope
  --coordination-slope      per-step coordination intensity slope

Batch + sweep:
  --n N                     emit N realizations (seed advances per realization)
  --sweep KEY=V1,V2,...     1-D parameter sweep (one realization per value); KEY
                            is any of: tau, invariant_ratio, perturbation_magnitude,
                            entropy_growth_rate, coord_slope

A manifest.json listing all emitted realizations + their params is written to the
output directory after each batch. Requires numpy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# --- defaults approximating the seed fixture synth_traj_xplat_001 ---
DEFAULT_N_STEPS = 12
DEFAULT_TAU = 6
DEFAULT_L = 32
DEFAULT_INVARIANT_RATIO = 22 / 32
DEFAULT_PERT_MAG = 0.20
DEFAULT_ENTROPY_START = 2.10
DEFAULT_ENTROPY_GROWTH = 0.243
DEFAULT_COORD_START = 0.12
DEFAULT_COORD_SLOPE = 0.074
HASH_LO, HASH_HI = 1, 9999

PHASE_NAMES = ["phase_0", "phase_1", "phase_2", "phase_3"]
PHASE_SEMANTICS = {
    "phase_0": "low coordination density",
    "phase_1": "increasing semantic alignment",
    "phase_2": "cross-environment coordination transition",
    "phase_3": "high task specificity / decomposition",
}

# (center on trajectory progress in [0,1], width, peak magnitude before renorm)
CLUSTER_PROFILES = {
    "C_a": (0.05, 0.18, 0.55),
    "C_b": (0.14, 0.20, 0.40),
    "C_c": (0.32, 0.22, 0.32),
    "C_d": (0.58, 0.30, 0.34),
    "C_e": (0.55, 0.26, 0.26),
    "C_f": (0.68, 0.20, 0.30),
    "C_g": (0.88, 0.18, 0.32),
    "C_h": (0.95, 0.18, 0.22),
}

# unit direction vectors for the platform centroids; final centroid = magnitude * dir
_PA_DIR = np.array([1.0, -0.75, 1.5, 0.5])
_PB_DIR = np.array([-1.5, 2.0, 1.25, -1.0])

SWEEP_KEYS = {
    "tau": ("tau", int),
    "invariant_ratio": ("invariant_ratio", float),
    "perturbation_magnitude": ("perturbation_magnitude", float),
    "entropy_growth_rate": ("entropy_growth_rate", float),
    "coord_slope": ("coord_slope", float),
}


@dataclass
class GenParams:
    fixture_id: str = "synth_traj_xplat_gen"
    n_steps: int = DEFAULT_N_STEPS
    tau: int = DEFAULT_TAU
    L: int = DEFAULT_L
    invariant_ratio: float = DEFAULT_INVARIANT_RATIO
    perturbation_magnitude: float = DEFAULT_PERT_MAG
    entropy_start: float = DEFAULT_ENTROPY_START
    entropy_growth_rate: float = DEFAULT_ENTROPY_GROWTH
    coord_start: float = DEFAULT_COORD_START
    coord_slope: float = DEFAULT_COORD_SLOPE
    seed: int = 0


def _phase_for(step: int, n_steps: int) -> str:
    return PHASE_NAMES[min(3, (step * 4) // n_steps)]


def _is_phase_boundary(step: int, n_steps: int) -> bool:
    return step > 0 and _phase_for(step, n_steps) != _phase_for(step - 1, n_steps)


def _draw_hash(rng: np.random.Generator, exclude: int | None = None) -> int:
    while True:
        v = int(rng.integers(HASH_LO, HASH_HI + 1))
        if v != exclude:
            return v


def _generate_signatures(p: GenParams, rng: np.random.Generator) -> tuple[list[list[int]], list[np.ndarray]]:
    L = p.L
    n_inv = max(0, min(L, round(p.invariant_ratio * L)))
    inv_positions = list(range(n_inv))
    cond_positions = list(range(n_inv, L))

    base = [int(rng.integers(HASH_LO, HASH_HI + 1)) for _ in range(L)]
    platform_b_values = {pos: _draw_hash(rng, exclude=base[pos]) for pos in cond_positions}

    # FIX 3: Non-LSH similarity metric (Structural Vector)
    # 32-dim vector representing latent structural features
    base_vec = rng.standard_normal(32)

    sigs = [list(base)]
    vecs = [base_vec.copy()]

    for t in range(1, p.n_steps):
        cur = list(sigs[-1])
        cur_vec = vecs[-1].copy()

        # FIX 1: Make entropy_growth_rate causal
        # Increase intensity: p=0.05 at 0.0, p=0.45 at 0.5
        mut_prob = 0.05 + (p.entropy_growth_rate * 0.8)
        
        # Mutation count is stochastic based on entropy
        n_mut = rng.binomial(n_inv, mut_prob) if n_inv > 0 else 0
        
        if t == p.tau:
            n_mut = max(n_mut, 3)
        elif _is_phase_boundary(t, p.n_steps):
            n_mut = max(n_mut, 4)

        n_mut = min(n_mut, n_inv)
        if n_mut > 0:
            mut_pos = rng.choice(inv_positions, size=n_mut, replace=False)
            for pos in mut_pos:
                cur[int(pos)] = _draw_hash(rng, exclude=cur[int(pos)])

        # Apply drift to structural vector
        # Drift magnitude scales with entropy
        drift_mag = 0.05 + (p.entropy_growth_rate * 0.3)
        cur_vec += rng.normal(0, drift_mag, size=32)

        if t == p.tau:
            if cond_positions:
                for pos in cond_positions:
                    cur[pos] = platform_b_values[pos]
            # Vector shock
            shock_dir = rng.standard_normal(32)
            shock_dir /= np.linalg.norm(shock_dir)
            cur_vec += (p.perturbation_magnitude * 2.0) * shock_dir
        elif t > p.tau and cond_positions and rng.random() < 0.4:
            pos = int(rng.choice(cond_positions))
            cur[pos] = _draw_hash(rng, exclude=cur[pos])

        sigs.append(cur)
        vecs.append(cur_vec)
    return sigs, vecs


def _cluster_mixture(progress: float) -> dict[str, float]:
    raw = {
        name: mag * float(np.exp(-((progress - c) ** 2) / (2 * w**2)))
        for name, (c, w, mag) in CLUSTER_PROFILES.items()
    }
    total = sum(raw.values())
    if total <= 0:
        return {}
    norm = {k: v / total for k, v in raw.items()}
    kept = {k: v for k, v in norm.items() if v >= 0.05}
    total = sum(kept.values())
    return {k: round(v / total, 2) for k, v in kept.items()}


def _entropy_at(step: int, p: GenParams) -> float:
    return round(p.entropy_start + step * p.entropy_growth_rate, 2)


def _coord_at(step: int, p: GenParams) -> float:
    return round(p.coord_start + step * p.coord_slope, 2)


def _entropy_factor(step: int, p: GenParams) -> float:
    final = _entropy_at(p.n_steps - 1, p) - p.entropy_start
    if abs(final) < 1e-9:
        return 0.0
    return (_entropy_at(step, p) - p.entropy_start) / final


def _embedding_drift(step: int, p: GenParams, rng: np.random.Generator) -> float:
    if step == 0:
        return 0.0
    ef = _entropy_factor(step, p)
    noise = rng.normal(0, 0.02 * (0.5 + ef))
    if step == p.tau:
        return round(0.45 + 0.05 * rng.random() + noise, 2)
    if _is_phase_boundary(step, p.n_steps):
        return round(0.20 + 0.05 * rng.random() + noise, 2)
    return round(0.10 + 0.06 * rng.random() + noise, 2)


def _lexical_density(step: int, p: GenParams, rng: np.random.Generator) -> float:
    progress = step / max(1, p.n_steps - 1)
    base = 0.30 + 0.61 * progress
    jitter = rng.normal(0, 0.01 * (0.5 + progress))
    return round(max(0.0, min(1.0, base + jitter)), 2)


def _coord_degree(step: int, p: GenParams, rng: np.random.Generator) -> float:
    base = _coord_at(step, p) * 10.0
    noise = rng.normal(0, 0.05 + 0.10 * step / max(1, p.n_steps - 1))
    if step == p.tau:
        base -= 0.3
    return round(base + noise, 1)


def _platform_perturbation(step: int, p: GenParams, rng: np.random.Generator) -> list[float]:
    centroid = p.perturbation_magnitude * (_PB_DIR if step >= p.tau else _PA_DIR)
    progress = step / max(1, p.n_steps - 1)
    noise = rng.normal(0, 0.02 + 0.04 * progress, size=4)
    return [round(float(x), 2) for x in centroid + noise]


def _jaccard(s1: list[int], s2: list[int]) -> float:
    return sum(1 for x, y in zip(s1, s2) if x == y) / len(s1)


def generate(params: GenParams) -> tuple[dict, dict]:
    rng = np.random.default_rng(params.seed)

    # FIX 2: Introduce non-degenerate tau distribution if default is used
    if params.tau == DEFAULT_TAU and params.fixture_id != "synth_traj_xplat_gen":
        # Randomize tau within [3, 9] to avoid edge effects
        params.tau = int(rng.choice([3, 4, 5, 6, 7, 8, 9]))

    sigs, vecs = _generate_signatures(params, rng)

    latent = {
        "fixture_id": params.fixture_id,
        "description": "Generated cross-platform trajectory fixture with causal entropy and variable tau.",
        "n_steps": params.n_steps,
        "phase_semantics": PHASE_SEMANTICS,
        "migration": {
            "tau": params.tau,
            "source_platform": "platform_A",
            "destination_platform": "platform_B",
        },
        "generator_params": asdict(params),
        "steps": [
            {
                "timestep": t,
                "phase": _phase_for(t, params.n_steps),
                "entropy": _entropy_at(t, params),
                "coordination_intensity": _coord_at(t, params),
            }
            for t in range(params.n_steps)
        ],
    }

    n_inv = max(0, min(params.L, round(params.invariant_ratio * params.L)))
    real_steps = []
    for t in range(params.n_steps):
        step = {
            "timestep": t,
            "phase": _phase_for(t, params.n_steps),
            "platform": "platform_B" if t >= params.tau else "platform_A",
            "semantic_token_clusters": _cluster_mixture(t / max(1, params.n_steps - 1)),
            "minhash_signature": sigs[t],
            "structural_vector": vecs[t].tolist(),  # FIX 3: Non-LSH baseline
            "embedding_drift": _embedding_drift(t, params, rng),
            "lexical_density": _lexical_density(t, params, rng),
            "coordination_graph_degree": _coord_degree(t, params, rng),
            "platform_perturbation": _platform_perturbation(t, params, rng),
        }
        if t == params.tau:
            step["migration_event"] = True
        real_steps.append(step)

    metrics: dict = {}
    intra_phase_adj: list[float] = []
    boundary_adj: list[float] = []
    for i in range(params.n_steps - 1):
        if i + 1 == params.tau:
            continue  # migration is reported separately
        j = _jaccard(sigs[i], sigs[i + 1])
        if _is_phase_boundary(i + 1, params.n_steps):
            boundary_adj.append(j)
        else:
            intra_phase_adj.append(j)
    if intra_phase_adj:
        metrics["intra_phase_adjacent_jaccard_range"] = [
            round(min(intra_phase_adj), 4),
            round(max(intra_phase_adj), 4),
        ]
    if boundary_adj:
        metrics["phase_boundary_adjacent_jaccard"] = round(
            sum(boundary_adj) / len(boundary_adj), 4
        )
    if 0 < params.tau < params.n_steps:
        metrics[f"migration_step_jaccard_step{params.tau - 1}_step{params.tau}"] = round(
            _jaccard(sigs[params.tau - 1], sigs[params.tau]), 4
        )
    if 0 <= params.tau and params.tau + 1 < params.n_steps:
        metrics[f"post_migration_recovery_jaccard_step{params.tau}_step{params.tau + 1}"] = round(
            _jaccard(sigs[params.tau], sigs[params.tau + 1]), 4
        )
    if 0 < params.tau and params.tau + 1 < params.n_steps:
        metrics[f"cross_platform_partial_overlap_step{params.tau - 1}_step{params.tau + 1}"] = round(
            _jaccard(sigs[params.tau - 1], sigs[params.tau + 1]), 4
        )
    if 0 < params.tau < params.n_steps:
        metrics[f"distant_intra_trajectory_jaccard_step0_step{params.tau}"] = round(
            _jaccard(sigs[0], sigs[params.tau]), 4
        )

    realization = {
        "fixture_id": params.fixture_id + "_realization",
        "links_to": params.fixture_id,
        "description": "Generated per-step telemetry realization (experiments/fixtures/generate.py).",
        "signature_params": {
            "L": params.L,
            "hash_space": [HASH_LO, HASH_HI],
            "platform_invariant_positions": list(range(n_inv)),
            "platform_conditioned_positions": list(range(n_inv, params.L)),
        },
        "platform_perturbation_basis": {
            "dim": 4,
            "platform_A_mean": [round(float(x), 2) for x in params.perturbation_magnitude * _PA_DIR],
            "platform_B_mean": [round(float(x), 2) for x in params.perturbation_magnitude * _PB_DIR],
        },
        "migration": {
            "tau": params.tau,
            "source_platform": "platform_A",
            "destination_platform": "platform_B",
        },
        "generator_params": asdict(params),
        "steps": real_steps,
        "expected_continuity_metrics": metrics,
    }
    return latent, realization


def _parse_sweep(spec: str) -> tuple[str, list]:
    if "=" not in spec:
        raise SystemExit(f"--sweep expects KEY=V1,V2,...; got {spec!r}")
    key, vals = spec.split("=", 1)
    if key not in SWEEP_KEYS:
        raise SystemExit(f"unknown sweep key {key!r}; pick one of {sorted(SWEEP_KEYS)}")
    field_name, caster = SWEEP_KEYS[key]
    parsed = [caster(v.strip()) for v in vals.split(",") if v.strip()]
    if not parsed:
        raise SystemExit(f"--sweep {spec!r} parsed to no values")
    return field_name, parsed


def _build_params(args: argparse.Namespace, override: dict | None, idx: int) -> GenParams:
    base = dict(
        n_steps=args.n_steps,
        tau=args.tau,
        L=args.L,
        invariant_ratio=args.invariant_ratio,
        perturbation_magnitude=args.perturbation_magnitude,
        entropy_start=args.entropy_start,
        entropy_growth_rate=args.entropy_growth_rate,
        coord_start=args.coord_start,
        coord_slope=args.coord_slope,
    )
    if override:
        base.update(override)
    fid = args.id_prefix if (args.n == 1 and not override) else f"{args.id_prefix}_{idx:04d}"
    return GenParams(fixture_id=fid, seed=args.seed + idx, **base)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/fixtures/generated"))
    ap.add_argument("--id-prefix", type=str, default="synth_traj_xplat_gen")
    ap.add_argument("--n", type=int, default=1, help="batch size (ignored when --sweep is set)")
    ap.add_argument("--sweep", type=str, default=None, help="KEY=V1,V2,... 1-D sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    ap.add_argument("--tau", type=int, default=DEFAULT_TAU)
    ap.add_argument("--L", type=int, default=DEFAULT_L)
    ap.add_argument("--invariant-ratio", type=float, default=DEFAULT_INVARIANT_RATIO)
    ap.add_argument("--perturbation-magnitude", type=float, default=DEFAULT_PERT_MAG)
    ap.add_argument("--entropy-start", type=float, default=DEFAULT_ENTROPY_START)
    ap.add_argument("--entropy-growth-rate", type=float, default=DEFAULT_ENTROPY_GROWTH)
    ap.add_argument("--coord-start", type=float, default=DEFAULT_COORD_START)
    ap.add_argument("--coord-slope", type=float, default=DEFAULT_COORD_SLOPE)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        sweep_field, sweep_vals = _parse_sweep(args.sweep)
        overrides = [{sweep_field: v} for v in sweep_vals]
    else:
        overrides = [None] * args.n

    manifest_entries = []
    for i, override in enumerate(overrides):
        params = _build_params(args, override, i)
        latent, realization = generate(params)
        latent_path = args.out_dir / f"{params.fixture_id}.json"
        real_path = args.out_dir / f"{params.fixture_id}_realization.json"
        latent_path.write_text(json.dumps(latent, indent=2) + "\n")
        real_path.write_text(json.dumps(realization, indent=2) + "\n")
        manifest_entries.append(
            {
                "fixture_id": params.fixture_id,
                "latent": latent_path.name,
                "realization": real_path.name,
                "params": asdict(params),
                "expected_continuity_metrics": realization["expected_continuity_metrics"],
            }
        )
        print(f"wrote {latent_path.name} + {real_path.name}")

    manifest = {
        "id_prefix": args.id_prefix,
        "sweep": args.sweep,
        "count": len(manifest_entries),
        "entries": manifest_entries,
    }
    manifest_path = args.out_dir / f"manifest_{args.id_prefix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path.name} ({len(manifest_entries)} entries)")


if __name__ == "__main__":
    main()
