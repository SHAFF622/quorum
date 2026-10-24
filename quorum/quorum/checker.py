"""
Check one finding: the website's main tool.

A user builds a finding as a sentence ("Among ER-positive patients, higher
ESR1 is linked to worse survival") and asks whether it holds up. This turns
that into a claim, runs every referee, and returns the verdict with the
evidence a chart needs. No AI is needed: the sentence is built from choices,
so the analysis to re-run is known exactly. (With AI credits, a free-text
sentence can go through the blind re-checker instead.)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import pandas as pd

from quorum.claims import Claim, ClaimType, Provenance, Stated
from quorum.data import load
from quorum.panel import judge
from quorum.pipeline import REFEREE_JOBS, REFEREE_NAMES
from quorum.simple_explorer import LABELS, _level
from quorum.codegen import code_bundle
from quorum.survival import AnalysisSpec, Filter

PHRASINGS = {
    "linked": "is linked to",
    "causes": "causes",
    "driver": "is a major driver of",
}


@lru_cache(maxsize=4)
def cohort(name: str) -> pd.DataFrame:
    return load(name)


def cohorts_for(dataset: str) -> dict[str, pd.DataFrame]:
    if dataset == "tcga":
        return {"tcga": cohort("tcga")}
    return {"metabric": cohort("metabric"), "tcga": cohort("tcga")}


# Who the finding is about. Each entry: id -> (label, filters).
def subgroups(dataset: str) -> dict[str, tuple[str, list[Filter]]]:
    groups = {"all": ("all patients", [])}
    if dataset == "metabric":
        groups["er_pos"] = ("ER-positive patients", [Filter(column="er_status", op="==", value="Positive")])
        groups["er_neg"] = ("ER-negative patients", [Filter(column="er_status", op="==", value="Negative")])
    groups["under40"] = ("patients under 40", [Filter(column="age", op="<", value=40)])
    groups["40plus"] = ("patients 40 and over", [Filter(column="age", op=">=", value=40)])
    groups["under55"] = ("patients under 55", [Filter(column="age", op="<", value=55)])
    groups["55plus"] = ("patients 55 and over", [Filter(column="age", op=">=", value=55)])
    for sub, label in (("LumA", "Luminal A"), ("LumB", "Luminal B"), ("Basal", "basal-like"), ("Her2", "HER2-enriched")):
        groups[f"pam50_{sub}"] = (f"{label} tumors", [Filter(column="pam50", op="==", value=sub)])
    if dataset == "tcga":
        groups["black"] = ("Black patients", [Filter(column="race", op="==", value="Black or African American")])
        groups["white"] = ("White patients", [Filter(column="race", op="==", value="White")])
    return groups


FRIENDLY_LEVELS = {
    "pam50": {"LumA": "Luminal A", "LumB": "Luminal B", "Basal": "basal-like", "Her2": "HER2-enriched",
              "Normal": "normal-like", "claudin-low": "claudin-low"},
    "tp53_mutated": {"1": "mutated", "0": "not mutated"},
}
SKIP = {"cohort", "os_months", "os_event", "dss_months", "dss_event", "histology", "stage", "ancestry"}


def options(dataset: str) -> dict:
    """Everything the sentence builder needs, read from the data itself."""
    df = cohort("tcga" if dataset == "tcga" else "metabric")
    factors = []
    for col in df.columns:
        if col in SKIP or df[col].notna().mean() < 0.5:
            continue
        s = df[col].dropna()
        friendly = FRIENDLY_LEVELS.get(col, {})
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 3:
            factors.append({"id": col, "label": LABELS.get(col, col), "kind": "number"})
            continue
        counts = s.astype(str).map(_level).value_counts()
        levels = [lvl for lvl, n in counts.items() if n >= 30]
        if 2 <= len(levels) <= 8:
            factors.append({"id": col, "label": LABELS.get(col, col), "kind": "category",
                            "levels": [{"id": lvl, "label": friendly.get(lvl, lvl)} for lvl in levels]})
    order = ["esr1_z", "pam50", "grade", "tumor_size", "nodes_positive", "age", "er_status", "her2_status",
             "tp53_mutated", "race", "chemotherapy", "hormone_therapy", "menopausal_state"]
    factors.sort(key=lambda f: order.index(f["id"]) if f["id"] in order else 99)
    return {"dataset": dataset, "factors": factors,
            "subgroups": [{"id": k, "label": v[0]} for k, v in subgroups(dataset).items()]}


def _raw_level(df: pd.DataFrame, col: str, level: str) -> str:
    """The builder sends '3'; the data may hold '3.0'. Find the stored spelling."""
    for v in df[col].dropna().astype(str).unique():
        if _level(v) == level:
            return v
    return level


def sentence(subgroup_label: str, factor: dict, exposed: Optional[str], reference: Optional[str],
             phrasing: str, direction: str) -> str:
    if factor["kind"] == "number":
        what = f"higher {factor['label']}"
    else:
        names = {lv["id"]: lv["label"] for lv in factor.get("levels", [])}
        what = f"{factor['label']} {names.get(exposed, exposed)} (compared with {names.get(reference, reference)})"
    who = "" if subgroup_label == "all patients" else f"Among {subgroup_label}, "
    text = f"{who}{what} {PHRASINGS.get(phrasing, 'is linked to')} {direction} survival."
    return text[0].upper() + text[1:]


def check(params: dict) -> dict:
    dataset = "tcga" if params.get("dataset") == "tcga" else "metabric"
    df = cohort(dataset)
    opts = options(dataset)
    factor = next((f for f in opts["factors"] if f["id"] == params.get("factor")), None)
    if factor is None:
        raise ValueError("Pick a factor to check.")
    group_label, filters = subgroups(dataset).get(params.get("subgroup", "all"), ("all patients", []))
    direction = "better" if params.get("direction") == "better" else "worse"
    exposed = reference = None
    if factor["kind"] == "category":
        exposed, reference = params.get("exposed"), params.get("reference")
        if not exposed or not reference or exposed == reference:
            raise ValueError("Pick two different groups to compare.")
        exposed, reference = _raw_level(df, factor["id"], exposed), _raw_level(df, factor["id"], reference)

    spec = AnalysisSpec(cohort=dataset, variable=factor["id"], exposed=exposed, reference=reference,
                        subgroup=filters)
    text = sentence(group_label, factor, _level(exposed) if exposed else None,
                    _level(reference) if reference else None, params.get("phrasing", "linked"), direction)
    claim = Claim(type=ClaimType.DATA, text=text, spec=spec, stated=Stated(direction=direction, tests_run=1),
                  provenance=Provenance(source="sentence builder"))

    evidence = []
    verdict = judge(claim, cohorts_for(dataset), None, on_evidence=evidence.append)
    from quorum.viz import finding_curves
    names = {lv["id"]: lv["label"] for lv in factor.get("levels", [])}
    names = {raw: names.get(_level(raw), raw) for raw in (exposed, reference) if raw is not None}
    return {
        "text": text,
        "label": verdict.label.value,
        "triage": verdict.triage,
        "reasons": verdict.reasons,
        "curves": finding_curves(df, spec, names),
        "code": code_bundle(spec, verdict.evidence, names),
        "checks": [{"referee": e.referee, "name": REFEREE_NAMES.get(e.referee, e.referee),
                    "job": REFEREE_JOBS.get(e.referee, ""), "outcome": e.outcome, "detail": e.detail,
                    "data": e.data} for e in evidence],
    }
