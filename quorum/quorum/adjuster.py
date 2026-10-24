"""
The fair-comparison check: does the effect survive comparing like with like?

The most common way a "significant" survival finding misleads is confounding:
a hidden third factor, usually age, makes two things look linked. Overall
survival adds a second trap: it counts deaths from any cause, so anything that
goes with being older looks deadly.

This referee needs no AI. It re-runs a reproduced claim twice:
1. adjusted for the standard clinical factors (age, grade, tumor size, nodes),
2. counting only breast cancer deaths (where the cohort records cause of death).
If the effect disappears or reverses, the claim gets "contests" with the
numbers, so a human looks before anyone acts on it.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from quorum.claims import Evidence
from quorum.survival import AnalysisSpec, SurvivalResult, analyse

ALPHA = 0.05
STANDARD_ADJUSTMENTS = ["age", "grade", "tumor_size", "nodes_positive"]
# Stage is built from tumor size and node count, so adjusting it for them would
# remove the very thing it measures (over-adjustment, not confounding).
PART_OF = {"stage": {"tumor_size", "nodes_positive"}}


def _fmt(r: SurvivalResult) -> str:
    return f"HR {r.hr:.2f} (95% CI {r.ci_low:.2f} to {r.ci_high:.2f}), p={r.p_value:.2g}"


def fair_comparison(spec: Optional[AnalysisSpec], original: Optional[SurvivalResult],
                    cohorts: dict[str, pd.DataFrame]) -> Evidence:
    if spec is None or original is None or original.p_value >= ALPHA:
        return Evidence(referee="adjuster", outcome="no_evidence", detail="No significant finding to stress-test.")
    df = cohorts[spec.cohort]
    adjust = [a for a in STANDARD_ADJUSTMENTS
              if a in df.columns and df[a].notna().mean() > 0.5 and a != spec.variable
              and a not in spec.adjust_for and not any(f.column == a for f in spec.subgroup)
              and a not in PART_OF.get(spec.variable, set())]
    checks = []
    if adjust:
        checks.append(("compared fairly (same " + ", ".join(a.replace("_", " ") for a in adjust) + ")",
                       spec.model_copy(update={"adjust_for": spec.adjust_for + adjust})))
    has_dss = "dss_event" in df.columns and df["dss_event"].notna().mean() > 0.5
    if has_dss and spec.endpoint == "OS":
        checks.append(("counting only breast cancer deaths, compared fairly",
                       spec.model_copy(update={"adjust_for": spec.adjust_for + adjust, "endpoint": "DSS"})))
    if not checks:
        return Evidence(referee="adjuster", outcome="no_evidence", detail="No adjustment factors available.")

    lines, problems, data = [], [], {"original": {"hr": original.hr, "p_value": original.p_value}}
    for label, s in checks:
        try:
            r = analyse(df, s)
        except (ValueError, KeyError) as e:
            lines.append(f"{label}: not computable ({e})")
            continue
        data[label] = {"hr": round(r.hr, 3), "ci": [round(r.ci_low, 3), round(r.ci_high, 3)],
                       "p_value": r.p_value, "n": r.n, "events": r.events, "spec": s.model_dump(mode="json")}
        lines.append(f"{label}: {_fmt(r)}")
        if r.p_value < ALPHA and r.direction != original.direction:
            problems.insert(0, f"the effect reverses when {label}")
        elif r.p_value >= ALPHA:
            problems.append(f"the effect disappears when {label}")

    detail = "; ".join(lines) + "."
    if problems:
        return Evidence(referee="adjuster", outcome="contests",
                        detail=f"Likely confounded: {problems[0]}. {detail}", data=data)
    return Evidence(referee="adjuster", outcome="supports", detail=f"Holds up. {detail}", data=data)
