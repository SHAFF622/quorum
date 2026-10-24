"""
A simple automatic analyst, for no-AI mode.

It plays the AI scientist when there is no API key: it reads the research
question for keywords, picks matching columns (plus the standard clinical
ones), runs a first-pass survival analysis for each, and writes up every
significant result the way a hurried analyst would: one variable at a time,
no adjustment, overall survival only. That is deliberately naive. It produces
exactly the kind of claim (significant but possibly confounded, or resting on
few deaths) that the referees exist to check.

It is labeled "simple analyst (no AI)" everywhere it appears.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

import pandas as pd

from quorum.claims import Claim, ClaimType, Provenance, Stated
from quorum.survival import AnalysisSpec, Filter, analyse

ALPHA = 0.05
LABELS = {
    "pam50": "subtype", "er_status": "ER status", "her2_status": "HER2 status", "grade": "tumor grade",
    "tumor_size": "tumor size", "nodes_positive": "number of positive lymph nodes", "age": "age",
    "esr1_z": "ESR1 (estrogen receptor) expression", "tp53_mutated": "TP53 mutation", "race": "race",
    "ancestry": "genetic ancestry", "chemotherapy": "chemotherapy", "hormone_therapy": "hormone therapy",
    "histology": "histological type", "menopausal_state": "menopausal state", "stage": "stage",
}
KEYWORDS = [
    (r"estrogen|oestrogen|\ber\b|esr1|hormone", ["esr1_z", "er_status", "hormone_therapy"]),
    (r"age|young|old|menopaus", ["age", "menopausal_state"]),
    (r"subtype|molecular|luminal|basal|pam50", ["pam50"]),
    (r"race|ancestry|black|ethnic", ["race", "ancestry"]),
    (r"her2|trastuzumab", ["her2_status"]),
    (r"tp53|mutation|gene", ["tp53_mutated"]),
    (r"grade|size|node|stage|clinical", ["grade", "tumor_size", "nodes_positive", "stage"]),
]
STANDARD = ["age", "grade", "tumor_size", "nodes_positive", "pam50", "er_status", "her2_status", "tp53_mutated"]
SKIP = {"cohort", "os_months", "os_event", "dss_months", "dss_event"}


def pick_variables(question: str, df: pd.DataFrame) -> list[str]:
    q = question.lower()
    chosen = []
    for pattern, cols in KEYWORDS:
        if re.search(pattern, q):
            chosen += cols
    chosen += STANDARD
    # Uploaded data: also try any other usable column.
    chosen += [c for c in df.columns if c not in SKIP]
    seen, out = set(), []
    for c in chosen:
        if c in df.columns and c not in seen and df[c].notna().mean() > 0.5:
            seen.add(c)
            out.append(c)
    return out


def plans_for(col: str, df: pd.DataFrame, cohort: str, question: str) -> list[AnalysisSpec]:
    s = df[col].dropna()
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 3:
        plans = [AnalysisSpec(cohort=cohort, variable=col)]
        if col == "esr1_z" and "er_status" in df.columns and df["er_status"].notna().any():
            plans.append(AnalysisSpec(cohort=cohort, variable=col,
                                      subgroup=[Filter(column="er_status", op="==", value="Positive")]))
        return plans
    levels = s.astype(str).value_counts()
    levels = levels[levels >= 30]
    if len(levels) < 2 or len(levels) > 8:
        return []
    reference = levels.index[0]
    if col == "pam50" and "LumA" in levels.index:
        reference = "LumA"
    return [AnalysisSpec(cohort=cohort, variable=col, exposed=lvl, reference=reference)
            for lvl in levels.index if lvl != reference]


def _level(v) -> str:
    """'3.0' -> '3'; leaves words alone."""
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(v)
    except (TypeError, ValueError):
        return str(v)


BINARY = {"tp53_mutated": ("TP53 mutation", "none")}


def describe(spec: AnalysisSpec) -> str:
    name = LABELS.get(spec.variable, spec.variable)
    where = ""
    for f in spec.subgroup:
        if f.column == "er_status" and f.op == "==":
            where = f" among ER-{str(f.value).lower()} patients"
        elif f.op == "==":
            where = f" among patients with {f.column.replace('_', ' ')} {f.value}"
    if spec.exposed is not None:
        if spec.variable in BINARY and _level(spec.exposed) == "1":
            label, none = BINARY[spec.variable]
            return f"{label} (vs {none}){where}"
        return f"{name} {_level(spec.exposed)} (vs {_level(spec.reference)}){where}"
    return f"higher {name}{where}"


def explore_simple(question: str, df: pd.DataFrame, cohort: str, run_id: str = "run1",
                   max_analyses: int = 30,
                   on_analysis: Optional[Callable[[AnalysisSpec, dict], None]] = None) -> list[Claim]:
    specs: list[AnalysisSpec] = []
    for col in pick_variables(question, df):
        specs += plans_for(col, df, cohort, question)
    specs = specs[:max_analyses]

    results = []
    for spec in specs:
        try:
            r = analyse(df, spec)
        except (ValueError, KeyError) as e:
            out = {"error": str(e)}
            r = None
        else:
            out = {"hr": round(r.hr, 3), "ci_95": [round(r.ci_low, 3), round(r.ci_high, 3)],
                   "p_value": float(f"{r.p_value:.3g}"), "n": r.n, "deaths": r.events,
                   "deaths_exposed": r.events_exposed, "deaths_reference": r.events_reference}
        if on_analysis:
            on_analysis(spec, out)
        results.append((spec, r))

    tests_run = sum(r is not None for _, r in results)
    claims = []
    for spec, r in results:
        if r is None or r.p_value >= ALPHA:
            continue
        what = describe(spec)
        per = " per standard deviation" if spec.exposed is None else ""
        text = (f"{what[0].upper() + what[1:]} is associated with {r.direction} overall survival "
                f"(HR {r.hr:.2f}{per}, p={r.p_value:.2g}, n={r.n}).")
        claims.append(Claim(
            type=ClaimType.DATA, text=text, spec=spec,
            stated=Stated(direction=r.direction, hr=round(r.hr, 2), p_value=r.p_value, n=r.n,
                          events=r.events, tests_run=tests_run),
            provenance=Provenance(source="simple analyst (no AI)", run_id=run_id, goal_wording=question,
                                  code=spec.model_dump_json()),
        ))
    return claims
