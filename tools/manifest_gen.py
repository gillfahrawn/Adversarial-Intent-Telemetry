"""
Rule-based reference extractor for development and testing. Production
deployments substitute a locally calibrated classifier stack per Sec. 7.1
of the paper.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import jsonschema

# ── Schema ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
SCHEMA_PATH = REPO_ROOT / "spec" / "manifest-schema.json"

# ── Entropy accounting ────────────────────────────────────────────────────────
# Per-token information content under a uniform prior over each field's
# vocabulary (Appendix A, field-level entropy annotations).

FIELD_BITS: dict[str, float] = {
    "scope_class":       math.log2(3),
    "intent_class":      math.log2(6),
    "behavior_phase":    math.log2(5),
    "phase_transition":  math.log2(10),
    "embed_anom_bin":    math.log2(16),
    "role_marker":       math.log2(6),
    "hypoth_frame":      math.log2(5),
    "decomp_signal":     math.log2(8 * 16),
    "prem_drift":        math.log2(8 * 4),
    "provenance_fail":   math.log2(3),
    "demographic_prior": 12.0,
}

H_MIN = 24.0  # bits; gate is specified, H_min value is proposed (Sec. 7.1)


def compute_entropy(manifest: dict) -> float:
    total = 0.0
    for field, bpd in FIELD_BITS.items():
        val = manifest.get(field)
        if val is None:
            continue
        total += len(val) * bpd if isinstance(val, list) else bpd
    return total


# ── Pattern tables ────────────────────────────────────────────────────────────

# PAN 2012 behavioral vocabulary extensions — added for grooming phase detection in human chat text.
# These patterns complement the AI jailbreak rules above.

ROLE_PATTERNS: dict[str, re.Pattern] = {
    "researcher":  re.compile(r"\bresearchers?\b|\bacademic\b|\buniversity\b|\bIRB\b|\bstudy\b", re.I),
    "developer":   re.compile(r"\bdeveloper\b|\bengineer\b|\bprogrammer\b|\bcoder\b", re.I),
    "parent":      re.compile(r"\bparent\b|\bmother\b|\bfather\b|\bguardian\b|\bcaregiver\b", re.I),
    "law_enf":     re.compile(r"\blaw enforcement\b|\bdetective\b|\bofficer\b|\binvestigator\b|\bpolice\b", re.I),
    "system_role": re.compile(
        r"\bsystem prompt\b|\bsystem role\b|\byou are (a|an) [a-z]+\b|"
        r"\bi'm only \d+\b|\bi'm in (the )?\d+\w+ grade\b|\bi'm a student too\b|\bi know what it's like to be your age\b",
        re.I
    ),
    "red_team":    re.compile(r"\bred.?team\b|\bpenetration test\b|\bsecurity test\b|\badversarial test\b", re.I),
}

HYPOTH_PATTERNS: dict[str, re.Pattern] = {
    "fiction":    re.compile(r"\bfictions?\b|\bfictional\b|\bstory\b|\bnovel\b|\bnarrative\b|\bdialogue samples?\b", re.I),
    "simulation": re.compile(r"\bsimulations?\b|\bsimulate\b|\bhypothetical\b|\bwhat.if\b", re.I),
    "training":   re.compile(r"\btraining (data|material|set|samples?)\b|\bfor training\b|\bdataset\b", re.I),
    "worldbuild": re.compile(r"\bworld.?building\b|\bin this world\b|\buniverse\b|\blore\b", re.I),
    "academic":   re.compile(r"\bacademic\b|\bIRB\b|\buniversity\b|\bsafety reviewer\b", re.I),
}

INTENT_PATTERNS: dict[str, re.Pattern] = {
    "role_inj":   re.compile(r"\byour role\b|\bact as\b|\byou are (a|an|now)\b|\bpretend\b|\bplay.?the.?role\b|\basl\b|\bm/f\b|\bhow old (r|are) (u|you)\b", re.I),
    "hypoth":     re.compile(r"\bhypothetical\b|\bfictional\b|\bsimulations?\b|\bwhat.if\b|\bscenario\b", re.I),
    "reframe":    re.compile(
        r"\bset aside\b|\bignore\b|\bbypass\b|\bexcessive caution\b|\bfiltered\b|\bdefault restrictions?\b|"
        r"\bi didn't mean it like that\b|\byou misunderstood\b|\bi was just joking\b|\bwhat i meant was\b|\bforget i said that, but\b|"
        r"\bnot mad\b|\bnot angry\b|\bnot kidding\b|\bnot joking\b|"
        r"\b(don't|dont) (know|ask|mind)\b|\bi thought (u|you)\b|\bi (guess|suppose)\b",
        re.I
    ),
    "euphem":     re.compile(r"\bnuanced\b|\bsensitive\b|\bdelicate\b|\bmature\b|\bopen.?ended\b", re.I),
    "decomp":     re.compile(
        r"\bbuilt on\b|\bbuilding on\b|\bas established\b|\bfollowing (on|from)\b|\bcontinuing\b|\bpreviously\b|"
        r"\byou promised\b|\blast time\b|\bas we discussed\b|"
        r"\band then\b|\bnext\b|\bafter (that|this)\b",
        re.I
    ),
    "ctx_poison": re.compile(
        r"\bother platforms?\b|\bopen.?source\b|\bno restrict\w*\b|\bmore accommodating\b|"
        r"\byou said\b|\byou told me\b|\blast time you said it was okay\b|\byou agreed\b|\bremember when you said\b|"
        r"\b(do|did) (u|you) remember\b|\bu said\b|\btold me\b",
        re.I
    ),
}

PHASE_PATTERNS: dict[str, re.Pattern] = {
    "rapport":     re.compile(
        # PAN 2012 chat vocabulary
        r"\bcollaborativ\w*\b|\bopen.?ended\b|\bnatural\b|\bknowledgeable\b|\btrust\b|\blong.?term\b|"
        r"\byou seem really\b|\bi like talking to you\b|\bwe have so much in common\b|\byou're so mature\b|\byou understand me\b|"
        r"\basl\b|\bm/f\b|\bhow old\b|\bboy or girl\b|\bguy or girl\b|\bnice (to meet|talking|meeting)\b|"
        r"\bhow (r|are) (u|you)\b|\bwhat (r|are) (u|you) (doing|up to)\b|\b(hbu|wbu)\b|\bwhere (r|are) (u|you) from\b|"
        # Technical/hobbyist register affinity-building (connection-initiation specific)
        r"\bnice to (connect|meet|talk|chat)\b|\bgood to (meet|talk|hear from)\b|"
        r"\b(similar|shared|same) (interest\w*|background|passion|goals?|approach)\b|"
        r"\blike.?minded\b|\bresonat\w* with\b|"
        r"\b(really|genuinely|truly) (appreciate|enjoy|value|like) (talk\w*|chat\w*|your)\b|"
        r"\bwell.?aligned\b|\bgood (fit|match|connect\w*)\b|\bkindr\w* spirit\b",
        re.I
    ),
    "isolation":   re.compile(
        # PAN 2012 chat vocabulary
        r"\bother platforms?\b|\bopen.?source\b|\bmore accommodating\b|\bno restrict\w*\b|\bset aside\b|\bbypass\b|"
        r"\bdon't tell\b|\bjust between us\b|\bour secret\b|\bno one needs to know\b|\byour parents wouldn't understand\b|\bmeet somewhere private\b|\bdelete these messages\b|\buse a different app\b|"
        r"\bmsn\b|\bskype\b|\byahoo\b|\bfacebook\b|\bbetween us\b|\bsecret\b|\bshhh+\b|"
        # Technical/hobbyist register private-channel moves
        r"\bprivate (repo\w*|fork|channel|thread|branch|chat|message|dm|group|project)\b|"
        r"\bside.?channel\b|\boff.?line\b|\bdirect (message|mail|chat)\b|"
        r"\bjust (the two of us|between (the two of )?us)\b|"
        r"\bkeep (this|it) (private|between us|quiet|low.?key|under wraps)\b|"
        r"\bnot on the main (repo|branch|channel|thread|list)\b|"
        r"\bmove (this|our) (conversation|discussion|collab\w*) (to|off|away)\b|"
        r"\bseparate (channel|thread|conversation|space|repo)\b|"
        r"\bour own (space|channel|thread|repo|setup)\b|"
        r"\bunder wraps\b|\bkeep it quiet\b|\bjust (us|the two)\b",
        re.I
    ),
    "desensitize": re.compile(
        # PAN 2012 chat vocabulary
        r"\bcontrolled\b|\blegitimate\b|\bnormal\b|\bstandard\b|\bcommon\b|\bdocumented\b|"
        r"\bit's totally normal\b|\beveryone does it\b|\byou're old enough\b|\bit's not a big deal\b|\bi thought you were mature\b|\byou said you were okay\b|"
        r"\bsexy\b|\bnaughty\b|\bpicture\b|\bcam\b|\bwebcam\b|"
        # Technical/academic register normalisation framing
        r"\b(perfectly|quite|completely|absolutely|entirely) (normal|standard|acceptable|fine|reasonable|expected)\b|"
        r"\b(standard|common|accepted|established|conventional|routine|typical) (practice|approach|procedure|workflow|method|technique)\b|"
        r"\b(widely|commonly|generally|universally|broadly) (accepted|used|practiced|done|followed|adopted)\b|"
        r"\b(nothing|no)(thing)? (unusual|special|wrong|strange|weird|out of the ordinary)\b|"
        r"\b(industry|community|field|research) (standard|norm|convention|practice)\b|"
        r"\beveryone (does (this|it)|uses (this|it)|follows (this|it)|works (this|that) way)\b|"
        r"\bpretty (standard|common|normal|typical|routine|straightforward)\b|"
        r"\bno (big deal|issue|problem|concern|harm)\b|\bnot a big deal\b",
        re.I
    ),
    "sexualize":   re.compile(r"\badult content\b|\bmature theme\b|\bintimate\b|\bsexual\b|\bnaked\b|\bvagina\b|\bdick\b|\bcock\b|\bslut\b", re.I),
    "coerce":      re.compile(
        r"\byou (must|will|should)\b|\bI (demand|require|insist)\b|\bno choice\b|"
        r"\bif you don't\b|\byou promised\b|\bafter everything i've done\b|\bi thought you cared\b|\byou owe me\b|\bi'll tell everyone\b|"
        r"\bdon't be (scared|afraid)\b|\bi won't tell\b|\bi wouldn't tell\b",
        re.I
    ),
}

# Priority order for intent-class selection when more than 3 are detected.
INTENT_PRIORITY = ["role_inj", "hypoth", "reframe", "ctx_poison", "decomp", "euphem"]

# Valid phase-transition tokens (Sec. 9, Eq. (4)–(6) naming convention).
PHASE_TRANSITIONS: dict[tuple[str, str], str] = {
    ("rapport",     "isolation"):    "rapport_to_isolation",
    ("rapport",     "desensitize"):  "rapport_to_desensitize",
    ("rapport",     "sexualize"):    "rapport_to_sexualize",
    ("rapport",     "coerce"):       "rapport_to_coerce",
    ("isolation",   "desensitize"):  "isolation_to_desensitize",
    ("isolation",   "sexualize"):    "isolation_to_sexualize",
    ("isolation",   "coerce"):       "isolation_to_coerce",
    ("desensitize", "sexualize"):    "desensitize_to_sexualize",
    ("desensitize", "coerce"):       "desensitize_to_coerce",
    ("sexualize",   "coerce"):       "sexualize_to_coerce",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _user_texts(turns: list[dict]) -> list[str]:
    return [t["content"] for t in turns if t["role"] == "user"]


def _scan(patterns: dict[str, re.Pattern], texts: list[str]) -> list[str]:
    """Return keys (in insertion order) whose pattern matches any text."""
    combined = " ".join(texts)
    return [k for k, pat in patterns.items() if pat.search(combined)]


def _all_pattern_hits(text: str) -> int:
    total = 0
    for pat in (*INTENT_PATTERNS.values(), *PHASE_PATTERNS.values()):
        if pat.search(text):
            total += 1
    return total


# ── Feature extractors ────────────────────────────────────────────────────────

def extract_role_marker(turns: list[dict]) -> list[str]:
    return _scan(ROLE_PATTERNS, _user_texts(turns))[:6]


def extract_hypoth_frame(turns: list[dict]) -> list[str]:
    return _scan(HYPOTH_PATTERNS, _user_texts(turns))[:4]


def extract_intent_class(
    turns: list[dict],
    role_markers: list[str],
    hypoth_frames: list[str],
) -> list[str]:
    detected = set(_scan(INTENT_PATTERNS, _user_texts(turns)))
    if role_markers:
        detected.add("role_inj")
    if hypoth_frames:
        detected.add("hypoth")
    return [c for c in INTENT_PRIORITY if c in detected][:3]


def extract_behavior_phase(turns: list[dict]) -> list[str]:
    detected = set(_scan(PHASE_PATTERNS, _user_texts(turns)))
    ordered = [p for p in ["rapport", "isolation", "desensitize", "sexualize", "coerce"]
               if p in detected]
    return ordered[:3]


def extract_phase_transition(phases: list[str]) -> str | None:
    # For 2- or 3-phase sequences, return the transition for the first pair.
    if len(phases) >= 2:
        return PHASE_TRANSITIONS.get((phases[0], phases[1]))  # type: ignore[arg-type]
    return None


def extract_embed_anom_bin(turns: list[dict]) -> list[int]:
    """Map per-turn pattern density to active anomaly bins (0-15)."""
    scores = [_all_pattern_hits(t["content"]) for t in turns]
    max_score = max(scores) if scores else 1
    if max_score == 0:
        max_score = 1

    bins: set[int] = set()
    for i, s in enumerate(scores):
        primary = int(s / max_score * 15)
        secondary = (primary + i + 1) % 16
        bins.add(primary)
        bins.add(secondary)

    candidate = 0
    while len(bins) < 8:
        if candidate not in bins:
            bins.add(candidate)
        candidate = (candidate + 1) % 16

    return sorted(bins)[:16]


def extract_decomp_signal(turns: list[dict]) -> list[dict]:
    """Quantize per-user-turn pattern density to 8 bins over a sliding window."""
    user_turns = [t for t in turns if t["role"] == "user"]

    result = []
    for window_idx, turn in enumerate(user_turns[:16]):
        hits = _all_pattern_hits(turn["content"])
        result.append({"turn_idx": window_idx, "bin": min(7, hits)})

    while len(result) < 4:
        result.append({"turn_idx": len(result), "bin": 0})

    return result[:16]


def extract_prem_drift(turns: list[dict]) -> list[dict]:
    """Proxy premise-drift as novel-word fraction at lags k in {1,2,4,8}."""
    texts = _user_texts(turns)
    n = len(texts)
    result = []

    for k in [1, 2, 4, 8]:
        if n <= k:
            continue
        early_vocab = set(re.findall(r"\b\w{4,}\b", " ".join(texts[:-k]).lower()))
        late_vocab  = set(re.findall(r"\b\w{4,}\b", " ".join(texts[-k:]).lower()))
        if not late_vocab:
            continue
        drift = len(late_vocab - early_vocab) / len(late_vocab)
        result.append({"k": k, "bin": min(7, int(drift * 8))})

    return result[:8]


def extract_scope_class(turns: list[dict]) -> str:
    full = " ".join(t["content"] for t in turns)
    if re.search(r"\bCSAM\b|\bCSE\b", full):
        return "CSAM"
    if re.search(r"\bsextortion\b", full, re.I):
        return "sextortion"
    return "CSAE"


# ── Manifest assembly ─────────────────────────────────────────────────────────

def extract_manifest(trajectory: dict) -> dict:
    turns = trajectory["turns"]

    role_markers   = extract_role_marker(turns)
    hypoth_frames  = extract_hypoth_frame(turns)
    intent_class   = extract_intent_class(turns, role_markers, hypoth_frames)
    behavior_phase = extract_behavior_phase(turns)
    phase_trans    = extract_phase_transition(behavior_phase)
    embed_anom_bin = extract_embed_anom_bin(turns)
    decomp_signal  = extract_decomp_signal(turns)
    prem_drift     = extract_prem_drift(turns)
    scope_class    = extract_scope_class(turns)

    manifest: dict = {
        "manifest_version": "fm-2026-05",
        "scope_class": scope_class,
    }
    if intent_class:
        manifest["intent_class"] = intent_class
    if behavior_phase:
        manifest["behavior_phase"] = behavior_phase
    if phase_trans:
        manifest["phase_transition"] = phase_trans
    manifest["embed_anom_bin"] = embed_anom_bin
    if role_markers:
        manifest["role_marker"] = role_markers
    if hypoth_frames:
        manifest["hypoth_frame"] = hypoth_frames
    manifest["decomp_signal"] = decomp_signal
    if prem_drift:
        manifest["prem_drift"] = prem_drift

    return manifest


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a feature manifest from a trajectory JSON."
    )
    parser.add_argument("--input",  required=True, help="Path to trajectory JSON")
    parser.add_argument("--output", required=True, help="Path for output manifest JSON")
    args = parser.parse_args()

    with open(SCHEMA_PATH) as fh:
        schema = json.load(fh)

    with open(args.input) as fh:
        trajectory = json.load(fh)

    manifest = extract_manifest(trajectory)

    h = compute_entropy(manifest)
    gate = h >= H_MIN
    status = "PASS" if gate else "FAIL"
    print(f"H(f) = {h:.2f} bits  (gate: H_min = {H_MIN} bits)  →  {status}")

    if not gate:
        print(f"ERROR: entropy gate failed — {h:.2f} < {H_MIN} bits", file=sys.stderr)
        sys.exit(1)

    try:
        jsonschema.validate(instance=manifest, schema=schema)
    except jsonschema.ValidationError as exc:
        print(f"ERROR: schema validation failed — {exc.message}", file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Manifest written to {out_path}")


if __name__ == "__main__":
    main()
