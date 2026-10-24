"""
The Replicator: does a METABRIC finding hold in different patients (TCGA)?

Kosmos can't fetch outside data to cross-check itself. This referee re-runs
the same analysis, with the same shared column names, on TCGA breast cancer.

TCGA is a weaker test: follow-up is short (median about 27 months), there are
few deaths (151 overall, 83 from breast cancer), it has no ER IHC status, and
gene expression was measured by RNA sequencing instead of microarrays (so we
compare per-standard-deviation effects, never raw values). So:
- same direction and significant -> "replicated" (supports)
- same direction, not significant -> "not refuted, too few deaths" (no evidence)
- opposite direction and significant -> contests (a human should look)
It never refutes on its own: a different cohort, era and assay can genuinely differ.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from quorum.claims import Evidence
from quorum.survival import AnalysisSpec, Filter, SurvivalResult, analyse

ALPHA = 0.05

# METABRIC-only columns and the closest TCGA stand-in (or None: can't replicate).
TCGA_STAND_INS = {
    "er_status": None,
    "her2_status": None,
    "chemotherapy": None,
    "hormone_therapy": None,
    "menopausal_state": None,
    "tumor_size": None,
    "nodes_positive": None,
}
# ER-positive in METABRIC ~ luminal subtypes in TCGA (luminal tumors are ER-positive by definition).
ER_POSITIVE_PROXY = Filter(column="pam50", op="in", value=["LumA", "LumB"])


def to_tcga(spec: AnalysisSpec) -> tuple[Optional[AnalysisSpec], list[str]]:
    """Translate a METABRIC analysis to TCGA, noting every substitution."""
    notes = []
    if spec.variable in TCGA_STAND_INS:
        return None, [f"{spec.variable} is not recorded in TCGA."]
    if spec.variable == "pam50" and "claudin-low" in (spec.exposed, spec.reference):
        return None, ["TCGA has no claudin-low subtype."]
    subgroup = []
    for f in spec.subgroup:
        if f.column == "er_status" and f.op == "==" and f.value == "Positive":
            subgroup.append(ER_POSITIVE_PROXY)
            notes.append("ER-positive approximated by luminal A/B subtype in TCGA.")
        elif f.column in TCGA_STAND_INS:
            return None, [f"Subgroup uses {f.column}, which TCGA doesn't record."]
        else:
            subgroup.append(f)
    adjust = [a for a in spec.adjust_for if a not in TCGA_STAND_INS]
    dropped = sorted(set(spec.adjust_for) - set(adjust))
    if dropped:
        notes.append(f"Could not adjust for {', '.join(dropped)} in TCGA.")
    return spec.model_copy(update={"cohort": "tcga", "subgroup": subgroup, "adjust_for": adjust}), notes


def replicate(spec: Optional[AnalysisSpec], original: Optional[SurvivalResult],
              cohorts: dict[str, pd.DataFrame]) -> Evidence:
    if spec is None or original is None or spec.cohort != "metabric" or "tcga" not in cohorts:
        return Evidence(referee="replicator", outcome="no_evidence", detail="No METABRIC finding to replicate.")
    if original.p_value >= ALPHA:
        return Evidence(referee="replicator", outcome="no_evidence",
                        detail="Original result not significant; nothing to replicate.")
    tcga_spec, notes = to_tcga(spec)
    if tcga_spec is None:
        return Evidence(referee="replicator", outcome="no_evidence", detail=" ".join(notes))
    tcga = cohorts["tcga"]
    usable = [a for a in tcga_spec.adjust_for if a in tcga.columns and tcga[a].notna().mean() > 0.5]
    missing = sorted(set(tcga_spec.adjust_for) - set(usable))
    if missing:
        notes.append(f"TCGA lacks usable {', '.join(missing)}; not adjusted for.")
        tcga_spec = tcga_spec.model_copy(update={"adjust_for": usable})
    try:
        r = analyse(cohorts["tcga"], tcga_spec)
    except (ValueError, KeyError) as e:
        return Evidence(referee="replicator", outcome="no_evidence", detail=f"TCGA analysis failed: {e}")

    found = (f"TCGA: HR {r.hr:.2f} (95% CI {r.ci_low:.2f} to {r.ci_high:.2f}), p={r.p_value:.2g}, "
             f"n={r.n}, deaths={r.events}.")
    caveat = (" " + " ".join(notes)) if notes else ""
    data = {"hr": round(r.hr, 3), "ci": [round(r.ci_low, 3), round(r.ci_high, 3)], "p_value": r.p_value,
            "n": r.n, "events": r.events, "spec": tcga_spec.model_dump(mode="json")}
    same_direction = r.direction == original.direction
    if r.p_value < ALPHA and same_direction:
        return Evidence(referee="replicator", outcome="supports", detail=f"Replicated. {found}{caveat}", data=data)
    if r.p_value < ALPHA:
        return Evidence(referee="replicator", outcome="contests",
                        detail=f"Opposite direction in TCGA. {found}{caveat}", data=data)
    lean = "same" if same_direction else "opposite"
    return Evidence(referee="replicator", outcome="no_evidence",
                    detail=f"Not replicated, not refuted: TCGA leans the {lean} way but has too few deaths. "
                           f"{found}{caveat}", data=data)
