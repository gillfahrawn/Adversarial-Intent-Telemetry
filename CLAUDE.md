# CLAUDE.md — Adversarial Intent Telemetry

Persistent context for Claude Code sessions on this repository. Read fully before
editing any file. Update the **Repository State** table whenever a file is added,
removed, or renamed.

---

## 1. Project Identity

**What this is**: A public empirical-research repository. It evaluates whether a
cross-provider behavioral-signature detection scheme survives real adversarial data
and adaptive perturbation. The scheme — a banded-MinHash signature primitive plus a
trajectory-level sequence model — was originally specified as a deployable protocol
in the design paper. **The paper is the design under test; the repository reports
what the primitives actually do when run.**

**Framing rule (do not drift from this)**: This is an honest empirical study with a
mixed result, not a protocol announcement. The headline is the *decomposition* of
where behavioral detection holds (per-message on clean PAN 2012) and where it breaks
(the signature primitive on real text; the per-message detector under perturbation).
Negative and inconclusive results are reported as such. Never restate the design
paper's `demonstrated`-level protocol claims as if the repository validated them.

**Design paper**: `Decentralized_Telemetry_Adversarial_AI_Intent_v8.1.pdf` (included
as the design under test).
**Author**: Fahrawn Gill, Advisor, AI Governance & Cross-Platform Safety, ACCO.

**What the repository must communicate, in order of importance:**
1. The falsifiable question and the mixed result (results table in README).
2. Which claims are real / simulated / analytical / inconclusive.
3. The integrity infrastructure (truth ledger, validity-boundary statement).
4. What is actually in the repository right now.
5. The design under test (the paper), framed as such.

---

## 2. Target Audience

Primary: empirical AI-safety and adversarial-ML researchers (the kind who clone the
repo and re-run the numbers). Secondary: T&S / detection engineers, AI-governance
staff, technically sophisticated hiring managers.

These readers will clone the repo if the README is compelling, and the repo must
then hold up. They trust epistemic modesty over confident claims, and they will
notice instantly if the README describes results the result files do not support.

---

## 3. Repository State

**This table is the source of truth for what the README may describe as existing.**
Update it on every add/remove/rename.

| Path | Status | Notes |
|---|---|---|
| `README.md` | exists | Empirical-study framing (B). No merge-conflict markers. |
| `CLAUDE.md` | exists | This file. |
| `AGENTS.md` | exists | Session/agent instructions. |
| `LICENSE.md` | exists | AGPL v3. |
| `Decentralized_Telemetry_Adversarial_AI_Intent_v8.1.pdf` | exists | Design under test. |
| `spec/manifest-schema.json` | exists | Feature manifest schema (Appendix A). |
| `examples/trajectory.json` | exists | Synthetic adversarial trajectory example. |
| `tools/manifest_gen.py` | exists | Rule-based manifest extractor (AI-jailbreak patterns; low transfer to PAN 2012). |
| `tools/inject_discourse_noise.py` | exists | Perturbation: retrieval-swap, reciprocity-asymmetry. |
| `tools/audit_human_vs_generated.py`, `tools/audit_structural_diversity.py` | exists | Synthetic-vs-real audits. |
| `tools/pan12_empirical_grounding.py` | exists | PAN 2012 grounding stats. |
| `tools/requirements.txt` | exists | Tooling deps. |
| `validation/synthetic/s_curve.py` | exists | **Illustrative** S-curve on synthetic Beta pairs. Not real-data performance. |
| `validation/synthetic/results/` | exists | s_curve.png, results.json. |
| `experiments/exp_m3_author_split.py` | exists | Author-disjoint PAN split. |
| `experiments/exp_m3_frontier.py` | exists | Operating-point frontier on real PAN 2012. |
| `experiments/exp_m3_federation_lift.py` | exists | Federation lift sweep. |
| `experiments/exp_trajectory_lift.py` | exists | Per-message vs sequence + evasion test (lift negative). |
| `experiments/exp_m8_byzantine.py`, `exp_m8_sprt.py` | exists | Byzantine / SPRT simulation. |
| `experiments/exp_f3_reciprocity.py` | exists | Analytical payoff-perturbation on GT-HarmBench. |
| `experiments/exp_annotate_pan_manifest.py` | exists | PAN manifest annotation (low field population, reported). |
| `experiments/exp_xplat_continuity.py` | exists | Cross-platform continuity sweep. |
| `experiments/exp_generate_*.py`, `regen_*.py`, `fill_missing_*.py`, `final_push_generation.py` | exists | Tier 0/2 trajectory generation utilities. |
| `experiments/results/*.json,*.csv,*.txt` | exists | All result artifacts incl. `truth_ledger.json`, `validity_boundary_statement.txt`. |
| `experiments/results/*.png,*.pdf` | exists | Figures. |
| `data/pan12/` | gitignored | Raw PAN 2012 corpus — NOT redistributed. |
| `data/pan_annotated/`, `data/agentic_ncmec/` | partially tracked | Derived Tier 1/2 jsonl (see data/.gitignore). |

**Hard rule**: The README must never describe, link to, or give commands for a path
not marked `exists` here. The README must never describe a result the corresponding
result file does not support.

---

## 4. Known hygiene items (resolve before public share)

- `.DS_Store` and `.claude/worktrees/` are tracked in Git — remove from tracking and
  add to `.gitignore`.
- `data/.gitignore` is broad (`*.json`, `*.csv`, `data/`) — confirm intended Tier 1/2
  derived files remain tracked; it can silently drop them on future commits.
- Several result JSONs contain absolute local paths (`/Users/fahrawngill/...`) in
  their `figures` fields — strip to repo-relative before sharing.
- `data/README.md` and `data/gt_harmbench/README.md` contain only `placeholder` — fill
  with provenance or remove.

---

## 5. Claim discipline (the Maturity language)

When describing any result, attach one of:
- **real** — measured on PAN 2012 / GT-HarmBench real data (author-disjoint where noted).
- **simulation** — Byzantine / SPRT results; validate a mechanism under stated params, not a deployment.
- **analytical** — F3 reciprocity; mechanism-design claim on game matrices, not LLM behavior.
- **synthetic** — Tier 0 and the s_curve harness; ablation / illustrative only.
- **negative / inconclusive** — e.g. trajectory F1 lift (CI below zero); state plainly.

Headline detection claims rest only on **real** Tier 1 results.
