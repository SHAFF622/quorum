"""
Science brief: the research question, answered by plain code and refereed by Quorum's own rules.

Research question: which clinical and molecular features best explain
differences in overall survival in breast cancer, and in which subgroups?

This is not the AI scientist. It's the reference analysis a human
biostatistician would run first, scripted so every number is reproducible.
It covers the three open gaps the public data can address:
- Gap 5 (headline): estrogen receptor expression as a continuous score (ESR1)
- Gap 1: race and genetic ancestry (TCGA only)
- Gap 3: patients under 40
Each result is passed through quorum.rules, so the brief flags its own weak
spots (too few deaths, and so on) exactly as the referees would.

    python -m quorum.brief      -> runs/science-brief.md and .json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from quorum import rules
from quorum.claims import Claim, ClaimType, Stated
from quorum.data import load
from quorum.survival import AnalysisSpec, Filter, SurvivalResult, analyse

OUT = Path(__file__).resolve().parent.parent / "runs"
ER_POS = Filter(column="er_status", op="==", value="Positive")
CLINICAL = ["age", "grade", "tumor_size", "nodes_positive"]


@dataclass
class Row:
    section: str
    question: str
    spec: AnalysisSpec
    result: SurvivalResult | None
    error: str = ""

    def flags(self, tests_run: int = 1) -> list[str]:
        if self.result is None:
            return []
        claim = Claim(type=ClaimType.DATA, text=self.question, spec=self.spec,
                      stated=Stated(direction=self.result.direction, tests_run=tests_run))
        return [e.detail for e in rules.check(claim, self.result) if e.data.get("rule") != "changed_endpoint"]


def run(section: str, question: str, df: pd.DataFrame, **spec) -> Row:
    s = AnalysisSpec(**spec)
    try:
        return Row(section, question, s, analyse(df, s))
    except (ValueError, KeyError) as e:
        return Row(section, question, s, None, str(e))


def with_quartiles(df: pd.DataFrame, col: str, within: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out[f"{col}_quartile"] = pd.Series(pd.NA, index=out.index, dtype="object")
    q = pd.qcut(out.loc[within, col], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    out.loc[within, f"{col}_quartile"] = q.astype(str)
    return out


def main() -> None:
    m, t = load("metabric"), load("tcga")
    rows: list[Row] = []
    extra: dict = {}

    # ---- Gap 5: continuous ER ---------------------------------------------
    g5 = "Gap 5: ESR1 expression within ER-positive tumors"
    for ep in ("OS", "DSS"):
        for adj, tag in (([], "unadjusted"), (CLINICAL, "adjusted for age, grade, size, nodes")):
            rows.append(run(g5, f"Higher ESR1, {ep}, {tag}", m, variable="esr1_z",
                            subgroup=[ER_POS], adjust_for=adj, endpoint=ep))
    erpos = m["er_status"] == "Positive"
    mq = with_quartiles(m, "esr1_z", erpos & m["esr1_z"].notna())
    for q in ("Q1", "Q2", "Q3"):
        rows.append(run(g5, f"ESR1 quartile {q} (lowest = Q1) vs Q4, breast cancer deaths, adjusted", mq,
                        variable="esr1_z_quartile", exposed=q, reference="Q4", subgroup=[ER_POS],
                        adjust_for=CLINICAL, endpoint="DSS"))
    for sub in ("LumA", "LumB"):
        rows.append(run(g5, f"Higher ESR1 within {sub}, breast cancer deaths, adjusted", m, variable="esr1_z",
                        subgroup=[ER_POS, Filter(column="pam50", op="==", value=sub)],
                        adjust_for=CLINICAL, endpoint="DSS"))
    for lo, hi, tag in ((0, 55, "under 55"), (55, 200, "55 and over")):
        rows.append(run(g5, f"Higher ESR1, {tag}, breast cancer deaths, adjusted", m, variable="esr1_z",
                        subgroup=[ER_POS, Filter(column="age", op=">=", value=lo),
                                  Filter(column="age", op="<", value=hi)],
                        adjust_for=CLINICAL, endpoint="DSS"))
    # Why the overall-survival result points the wrong way: who dies of what, by quartile.
    sub = mq[erpos & mq["esr1_z_quartile"].notna() & mq["os_event"].notna()]
    extra["esr1_quartile_profile"] = (
        sub.groupby("esr1_z_quartile").agg(
            patients=("age", "size"), median_age=("age", "median"),
            breast_cancer_deaths=("dss_event", "sum"), all_deaths=("os_event", "sum"))
        .assign(other_cause_deaths=lambda d: d.all_deaths - d.breast_cancer_deaths)
        .round(1).reset_index().to_dict(orient="records"))
    rows.append(run(g5, "Replication in TCGA: higher ESR1 in luminal A/B, breast cancer deaths, age-adjusted", t,
                    cohort="tcga", variable="esr1_z", subgroup=[Filter(column="pam50", op="in", value=["LumA", "LumB"])],
                    adjust_for=["age"], endpoint="DSS"))

    # ---- Gap 1: ancestry (TCGA) --------------------------------------------
    g1 = "Gap 1: race and genetic ancestry (TCGA)"
    for ep in ("OS", "DSS"):
        rows.append(run(g1, f"Black vs White patients, {ep}, age-adjusted", t, cohort="tcga", variable="race",
                        exposed="Black or African American", reference="White", adjust_for=["age"], endpoint=ep))
    rows.append(run(g1, "Black vs White patients within Luminal A, OS, age-adjusted", t, cohort="tcga",
                    variable="race", exposed="Black or African American", reference="White",
                    subgroup=[Filter(column="pam50", op="==", value="LumA")], adjust_for=["age"]))
    afr = t["ancestry"].isin(["AFR", "AFR_ADMIX"])
    eur = t["ancestry"] == "EUR"
    seq = t["tp53_mutated"].notna()
    a = [[int(t.loc[afr & seq, "tp53_mutated"].sum()), int((afr & seq).sum() - t.loc[afr & seq, "tp53_mutated"].sum())],
         [int(t.loc[eur & seq, "tp53_mutated"].sum()), int((eur & seq).sum() - t.loc[eur & seq, "tp53_mutated"].sum())]]
    odds, p = fisher_exact(a)
    extra["tp53_by_ancestry"] = {
        "AFR_rate": round(a[0][0] / sum(a[0]), 3), "AFR_n": sum(a[0]),
        "EUR_rate": round(a[1][0] / sum(a[1]), 3), "EUR_n": sum(a[1]),
        "odds_ratio": round(odds, 2), "p_value": float(f"{p:.3g}")}
    basal = t["pam50"] == "Basal"
    extra["basal_share_by_ancestry"] = {
        "AFR": round(float(basal[afr].mean()), 3), "EUR": round(float(basal[eur].mean()), 3)}

    # ---- Gap 3: under 40 ----------------------------------------------------
    g3 = "Gap 3: patients under 40 (METABRIC)"
    young = Filter(column="age", op="<", value=40)
    rows.append(run(g3, "Under 40 vs 40 and over, breast cancer deaths, adjusted for grade, size, nodes", m.assign(
        under40=np.where(m["age"] < 40, "yes", "no")), variable="under40", exposed="yes", reference="no",
        adjust_for=["grade", "tumor_size", "nodes_positive"], endpoint="DSS"))
    rows.append(run(g3, "Under 40: HER2-positive vs negative, breast cancer deaths", m, variable="her2_status",
                    exposed="Positive", reference="Negative", subgroup=[young], endpoint="DSS"))
    rows.append(run(g3, "Under 40: basal-like vs luminal A, breast cancer deaths", m, variable="pam50",
                    exposed="Basal", reference="LumA", subgroup=[young], endpoint="DSS"))
    extra["subtype_mix"] = {
        "under_40": m.loc[m["age"] < 40, "pam50"].value_counts(normalize=True).round(3).to_dict(),
        "40_and_over": m.loc[m["age"] >= 40, "pam50"].value_counts(normalize=True).round(3).to_dict()}

    write(rows, extra)


def fmt(r: Row) -> str:
    if r.result is None:
        return f"not computable: {r.error}"
    x = r.result
    return f"HR {x.hr:.2f} ({x.ci_low:.2f} to {x.ci_high:.2f}), p={x.p_value:.2g}, n={x.n}, deaths={x.events}"


def write(rows: list[Row], extra: dict) -> None:
    lines = ["# Science brief: what explains breast cancer survival, and where", "",
             "Reference analyses by plain code on public METABRIC and TCGA data, each checked by Quorum's "
             "statistics rules. HR above 1 means the named group (or higher values) does worse. "
             "OS = death from any cause; DSS = death from breast cancer only.", ""]
    k = sum(r.result is not None for r in rows)
    lines.insert(4, f"This brief ran {k} survival analyses, so each p-value is also checked against a "
                    f"Bonferroni threshold of 0.05/{k} = {0.05 / k:.4f}.")
    lines.insert(5, "")
    section = None
    for r in rows:
        if r.section != section:
            section = r.section
            lines += ["", f"## {section}", "", "| Question | Result | Rule flags |", "| --- | --- | --- |"]
        flags = "; ".join(r.flags(k)) or "none"
        lines.append(f"| {r.question} | {fmt(r)} | {flags} |")
    lines += ["", "## Supporting tables", "", "```json", json.dumps(extra, indent=2), "```"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "science-brief.md").write_text("\n".join(lines) + "\n")
    (OUT / "science-brief.json").write_text(json.dumps({
        "rows": [{"section": r.section, "question": r.question, "spec": r.spec.model_dump(mode="json"),
                  "result": None if r.result is None else r.result.__dict__, "error": r.error,
                  "flags": r.flags(sum(x.result is not None for x in rows))} for r in rows],
        "extra": extra}, indent=2, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
