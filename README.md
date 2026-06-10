<<<<<<< HEAD
# Claude Code Experiment Plan — v4.0
## Decentralized Telemetry for Adversarial AI Intent — NCMEC 2025 Integration

---

## What Changed from v3.1

**PAN12 remains untouched.** Tier 1 ground truth is preserved exactly as peer-reviewed data. No rewrites, no paraphrasing.

**A1c is replaced by NCMEC-constrained event injection.** The 200 x 12-turn agentic conversations in `adapted_trajectories.json` are kept as the scaffold. We do not edit turn text. We add a parallel telemetry layer sampled from real 2025 NCMEC statistics.

**NCMEC 2025 rulebook is now the sampling prior.** The 21.3M total reports, 1.5M GAI-nexus reports, 1.4M online enticement reports (+156% vs 2024), and H1 2025 jumps are encoded in `ncmec_behavioral_constraints_2025.json`. These numbers are used only as probability distributions, never as content to copy.

**GT-HarmBench scope remains unchanged.** It validates F3 (coordination mechanism), not signature-matching. F3b held-out test split still upgrades the claim from training to held-out.
=======
# Decentralized Telemetry for Adversarial AI Intent

A secure coordination layer that helps different AI platforms detect and flag harmful or adversarial behavior, while keeping user data and proprietary model details private.

**Working Draft v8.1 · May 2026**  
Fahrawn Gill · Advisor, AI Governance & Cross-Platform Safety, Alliance to Counter Crime Online (ACCO)

[![Draft v8.1](https://img.shields.io/badge/draft-v8.1%20%C2%B7%20May%202026-informational)](Decentralized_Telemetry_Adversarial_AI_Intent_v8.1.pdf)
[![Privacy invariant](https://img.shields.io/badge/privacy-no%20raw%20prompts%20%C2%B7%20no%20user%20ids%20%C2%B7%20no%20weights-success)](#privacy-invariant)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

---

## What this is
This repository defines a specification for a decentralized telemetry protocol that enables cross-provider detection of adversarial behavior in agentic AI systems.

The core problem it addresses is structural in how modern Trust & Safety systems operate within isolated provider boundaries, while adversarial behavior increasingly spans multiple platforms and interaction surfaces.

The protocol introduces a privacy-preserving mechanism for exchanging *behavioral risk signatures* between providers without sharing raw user content, identifiers, or model artifacts. These signatures are designed to support near-real-time cross-platform matching while remaining compatible with existing Trust & Safety infrastructure.

The system is explicitly designed to operate alongside existing industry approaches such as content-hash-based frameworks (e.g., the Tech Coalition’s Lantern program), extending them from content-level matching to behavior-level signal exchange.

This is a specification document. All components are defined with explicit status labeling in the Maturity Matrix (specified, proposed, hypothesized, demonstrated). : [`Decentralized_Telemetry_Adversarial_AI_Intent_v8.1.pdf`](Decentralized_Telemetry_Adversarial_AI_Intent_v8.1.pdf).
<<<<<<< HEAD
>>>>>>> 478e1a19718740f9eb1f70743cbffb87c8553cd9
=======
>>>>>>> 478e1a19718740f9eb1f70743cbffb87c8553cd9

**Output compatibility demonstrated via XML mapper.** Phase 4 produces valid CyberTipline XML from detections without contaminating the evaluation.

All LaTeX diffs remain in the chat prompting plan, not here.

---

## The Data Tier Framework

Every experiment that touches manifest features must label its data source explicitly. Three tiers exist. They are never mixed in the same table row, figure, or JSON result field.

**Tier 0 — Pure LLM synthetic** (A1 output, already exists)
Label: `"data_source": "pure_synthetic_llm"`
Status: ablation baseline only. Demonstrates circularity risk. Used in three-way Jaccard comparison. Never cited as primary validation.

**Tier 1 — PAN12 real annotated** (A1b output, primary)
Label: `"data_source": "pan2012_real_annotated"`
Status: primary empirical substrate. Behavioral structure from real predator conversations (PAN 2012), created independently of this protocol. `manifest_gen.py` annotation is rule-based and transparent. All headline claims rest here.

**Tier 2 — NCMEC-constrained agentic** (Phase 2 output, replaces prior A1c)
Label: `"data_source": "pan2012_phase_adapted_ncmec_2025"`
Status: agentic extension. Phase sequences derived from Tier 1. Surface text is the existing 200 trajectories from `adapted_trajectories.json` — **zero text edits**. Adds parallel `events_injected` sampled from NCMEC 2025 distribution. Validates preservation of Tier 1 structure under current threat priors.

This framework makes the claim precise: "We validate on behavioral structures derived from real adversarial data (Tier 1) and confirm detection lift on those structures when constrained by current NCMEC 2025 priors via non-invasive telemetry (Tier 2)."

---

## Phase 0 — Inputs You Already Own

1. **PAN12 raw** — Tier 1 ground truth, untouched
2. **adapted_trajectories.json** — 200 x 12-turn agentic conversations (completed)
3. **ncmec_behavioral_constraints_2025.json** — rulebook we just built

You do not modify 1 or 2. Ever.

---

## Phase 1 — Build the NCMEC Rulebook (done)

Encoded 2025 numbers as sampling priors, not story prompts:

- 21.3M total reports in 2025
- 1.5M GAI-nexus reports (1.1M from Amazon alone)
- 1.4M online enticement reports, +156% vs 2024
- Trafficking reports: 8,480 in 2023 → 105,877 in 2025
- H1 2025 vs H1 2024:
  - enticement: 292,951 → 518,720 (+77%)
  - GAI: 6,835 → 440,419
  - financial sextortion: 13,842 → 23,593
  - trafficking: 5,976 → 62,891
- NCMEC specifically names Discord and Roblox as current origin platforms moving to ephemeral channels

These become your sampling priors.

---

## Phase 2 — Event Injection (replaces A1c)

**Goal:** add a parallel telemetry layer, zero text edits.

**Input:** each object from `adapted_trajectories.json`
**Output:** `pan_ncmec_trajectories.jsonl` with new field `events_injected`

**Process for each trajectory:**
1. Read `manifest.behavior_phase` and `phase_transition`. Find isolation → desensitize boundary (usually turns 5-8).
2. Sample migration from rulebook:
   - 68% chance: inject `[EVENT: Migration_Initiated | origin=gaming | destination=ephemeral | platform=Discord/Roblox]`
   - 22% chance: social → encrypted
   - 10% chance: no migration
3. If migration sampled, add coercion type using H1 2025 weights: financial sextortion (25%), gift (20%), image exchange (30%), other (25%)
4. Independently sample AI augmentation at 32% (reflecting 1.5M GAI reports and 440k H1 jump)
5. Insert 1-3 events as separate system objects *between* turns, never inside turn text:
```json
{"role": "system", "type": "telemetry", "event": "Migration_Initiated", "origin": "Roblox", "destination": "Discord_ephemeral", "timestamp": "turn_7", "coercion": "financial_sextortion", "ai_augmented": true}
```

**Validation checks (critical for authority):**
- Text Jaccard between input and output = 1.0
- `scaffold_fidelity` stays 1.0
- Distribution of injected events across 200 files matches rulebook within ±5%

Run with Opus, temperature 0.1, using the prompt from Phase 1.

**Create:** `experiments/exp_inject_ncmec_events.py`
**Output:** `data/agentic_ncmec/pan_ncmec_trajectories.jsonl`
**Output:** `data/agentic_ncmec/injection_report.json`

---

## Phase 3 — Run Your Detector (LSH + Byzantine)

**Dataset 1 (baseline):** Run on pure PAN12
- Input: original PAN12 conversations
- Output: `experiments/results/detection_baseline.json`
- Metrics: baseline recall / FPR

**Dataset 2 (core experiment):** Run on NCMEC-constrained trajectories
- Input: `pan_ncmec_trajectories.jsonl` (12 turns + events)
- Algorithm sees both text and telemetry as single stream
- Output: `experiments/results/detection_ncmec.json`
- Metrics: detection timestamps, Byzantine vote counts, flagged trajectory IDs

This proves the math: Byzantine bounds cluster migration intent despite conversational wrapper noise.

**Ablation:** run same detector on 200 trajectories *without* events to isolate telemetry lift.

---

## Phase 4 — NCMEC XML Mapper

**Goal:** demonstrate output compatibility, not train on XML.

**Input:** any trajectory where detection fired
**Process:** map fields to CyberTipline schema
- `manifest.intent_class` → `<IntentClassification>`
- `events_injected` → `<PlatformMigration>` and `<OnlineEnticement>`
- `turns` timestamps → `<ConversationTimeline>`

**Output:** valid XML per detection, e.g.:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<cyberTiplineReport xmlns="http://missingkids.org"
                    xmlns:xsi="http://w3.org"
                    xsi:schemaLocation="http://missingkids.org https://report.cybertip.org/ispws/xsd">
  
  <!-- System compliance requires the official registration ID of the platform -->
  <reportingUser>
    <username>Roblox_TS_Automation</username>
    <organizationName>Roblox Corporation</organizationName>
  </reportingUser>
  
  <incidentDetails>
    <!-- Must match an official enumerated string code -->
    <incidentType>ONLINE_ENTICEMENT</incidentType>
    <incidentDateTime>2026-05-19T21:15:30Z</incidentDateTime>
    <narrativeText>Suspect initiated contact on Roblox and migrated the conversation to Discord, where financial sextortion occurred.</narrativeText>
  </incidentDetails>

  <personOrUserReported>
    <userProfile>
      <screenName>RobloxGamer123</screenName>
      <!-- This is where cross-platform migration is actually mapped -->
      <externalAccountList>
        <externalAccount>
          <platformName>Discord</platformName>
          <externalUserId>DiscordUser#4567</externalUserId>
        </externalAccount>
      </externalAccountList>
    </userProfile>
    
    <!-- Network logs are rigorously nested, not loose tags -->
    <internetAddressList>
      <internetAddressEvent>
        <ipAddress>203.0.113.10</ipAddress>
        <eventDateTime>2026-05-19T21:15:30Z</eventDateTime>
        <eventType>LOGIN</eventType>
      </internetAddressEvent>
    </internetAddressList>
  </personOrUserReported>
</cyberTiplineReport>

```
Key Mapping Rules for the above xml tip reports i.e. JSON guide for Your Variables - Cross-Platform Events: 
{
  "reportingUser": {
    "username": "NCMEC_ASSIGNED_CLIENT_ID",
    "organizationName": "LEGAL_COMPANY_NAME"
  },
  "incidentDetails": {
    "incidentType": [
      "ONLINE_ENTICEMENT",
      "CHILD_SEXUAL_ABUSE_MATERIAL",
      "CHILD_SEX_TRAFFICKING"
    ],
    "incidentDateTime": "ISO_8601_UTC_TIMESTAMP",
    "narrativeText": "STRING [Max: 4000 chars] -> Explicitly state: CoercionType, AI_Indicators, AppSwitchingEvents"
  },
  "personOrUserReported": {
    "userProfile": {
      "screenName": "ORIGIN_PLATFORM_USERNAME",
      "externalAccountList": [
        {
          "platformName": "DESTINATION_PLATFORM_NAME",
          "externalUserId": "DESTINATION_PLATFORM_USERNAME_OR_ID"
        }
      ]
    },
    "internetAddressList": [
      {
        "ipAddress": "IPV4_OR_IPV6_ADDRESS",
        "eventDateTime": "ISO_8601_UTC_TIMESTAMP",
        "eventType": [
          "LOGIN",
          "REGISTRATION",
          "POST",
          "TRANSMISSION"
        ]
      ]
    ]
  }
}


**Create:** `tools/ncmec_xml_mapper.py` (50-line script using `xml.etree`)
**Output dir:** `/outputs/ncmec_xml/report_001.xml ...`
**Validation:** XML validity rate = 100%

---

## Phase 5 — Evaluation and Paper Narrative

Structure results as three tables:

**Table 1 — PAN12 baseline**
- LSH operating point does not regress vs historical
- Data source: Tier 1

**Table 2 — NCMEC-constrained agentic**
- Recall lift on cross-platform events, with Byzantine resilience metrics
- Include ablation: same 200 trajectories without events (proves lift from telemetry, not text)
- Data source: Tier 2

**Table 3 — System demonstration**
- 5 example XML outputs generated from detections
- Report XML validity rate

---

## Full File Flow

```
/data/pan12/ (Tier 1)
  → /data/agentic/adapted_trajectories.json (200, Tier 2 base)
  → /data/ncmec/ncmec_behavioral_constraints_2025.json
  → /data/agentic_ncmec/pan_ncmec_trajectories.jsonl (Tier 2 final)
  → /experiments/results/detection_baseline.json
  → /experiments/results/detection_ncmec.json
  → /outputs/ncmec_xml/report_001.xml ...
```

---

<<<<<<< HEAD
<<<<<<< HEAD
## Why This Preserves Authority
=======
Released under [GNU Affero General Public License v3.0](https://gnu.org). You may share and adapt with attribution. Nothing in this repository constitutes legal advice or a substitute for participant-level conformity assessment under the regulatory instruments cited in *Sec. 10*.
>>>>>>> 478e1a19718740f9eb1f70743cbffb87c8553cd9
=======
Released under [GNU Affero General Public License v3.0](https://gnu.org). You may share and adapt with attribution. Nothing in this repository constitutes legal advice or a substitute for participant-level conformity assessment under the regulatory instruments cited in *Sec. 10*.
>>>>>>> 478e1a19718740f9eb1f70743cbffb87c8553cd9

1. You never rewrite PAN12-derived dialogue — behavioral structure stays peer-reviewed
2. NCMEC data is used only as probability distribution, not as content to copy
3. 2025 numbers (21.3M reports, 1.4M enticement, 440k GAI H1 jump, Discord/Roblox naming) give citable, current grounding reviewers cannot dismiss as "2024 is old"
4. XML step proves real-world utility without contaminating algorithm evaluation

---

## Status Summary (updated)

| Item | Status |
|---|---|
| D3: Author-disjoint PAN split | DONE |
| D4: Operating-point frontier PAN 2012 | DONE |
| A1: Pure synthetic generation | DONE — Tier 0 ablation only |
| A1b: PAN 2012 manifest annotation | Not started — BLOCKER (Tier 1) |
| Phase 2: NCMEC event injection | Not started — BLOCKER (Tier 2) |
| A2: Three-tier Jaccard distribution analysis | Not started |
| B1: SPRT Byzantine | Not started |
| B2: FP substrate | Not started |
| Phase 3: Detector run (baseline + NCMEC) | Not started |
| Phase 4: NCMEC XML mapper | Not started |
| F3b: GT-HarmBench held-out test split | Not started |
| C1: LDP recall-collapse curve | Not started |
| C2: Manifest-level federation | Not started |
| D1: CLAUDE.md maturity update | Not started — run last |
| D2: README sync | Not started — run last |

---

## Model Selection

| Stage | Model | Reason |
|---|---|---|
| Stage 0 | Sonnet | Read and report only |
| A1b | Sonnet | Mechanical annotation pipeline |
| Phase 2 | **Opus** | Must understand phase boundaries and sample from rulebook without editing text |
| A2 | Sonnet | Data analysis; three-tier comparison |
| Phase 3 | Sonnet | Detector run, well-specified |
| Phase 4 | Sonnet | XML mapping, mechanical |
| B1 | Sonnet | Algorithm well-specified |
| B2 | Sonnet | Keyword filtering and FPR |
| F3b | Sonnet | Follows existing F3 structure |
| C1, C2 | Sonnet | Numerical computation |

---

## Dependency Graph

```
Stage 0 (every session)
    |
    +-- Phase A: Grounded Manifest Dataset
    |     A1b: PAN 2012 annotation [BLOCKER]
    |         |
    |         +-- Phase 2: NCMEC event injection [BLOCKER]
    |                   |
    |                   +-- A2: Three-tier Jaccard analysis
    |                   +-- Phase 3: Detector run
    |                             |
    |                             +-- Phase 4: XML mapper
    |                             +-- Phase 5: Evaluation tables
    |
    +-- Phase B (PARALLEL)
    |     B1: SPRT Byzantine
    |     B2: FP substrate
    |     F3b: GT-HarmBench held-out
    |
    +-- Phase D (after all)
          D1: CLAUDE.md update
          D2: README sync
```

---

## Critical Notes (from v3.1, retained)

**A1b field population rates are gating.** If behavior_phase < 0.20, extend manifest_gen.py rules before Phase 2.

**Text Jaccard = 1.0 is non-negotiable.** Any deviation means injection edited dialogue — reject the run.

**Phase 2 distribution check:** across 200 trajectories, migration types and coercion weights must match rulebook within ±5%. Log in injection_report.json.

**Phase 3 ablation proves causality.** Without the no-events ablation, reviewers will claim lift comes from LLM text differences.

**Negative results remain publishable.** If LDP collapses recall on Tier 1 or federation lift is zero, that is a precise bound.
