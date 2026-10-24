"""
Readable code for every statistical test behind a verdict.

For each analysis a referee ran, this writes a short, standalone Python script
that redoes it: load the data, pick the patients, fit the Cox model, print the
hazard ratio, its 95% confidence interval and the p-value. It spells the
statistics out with statsmodels instead of calling Quorum's own helper, so a
reader can audit every step. A test runs the generated code and checks it
prints the same numbers the website shows.
"""

from __future__ import annotations

from typing import Optional

from quorum.survival import ENDPOINTS, AnalysisSpec

ENDPOINT_WORDS = {"OS": "overall survival (death from any cause)", "DSS": "breast cancer deaths only"}


def _filter_lines(spec: AnalysisSpec) -> list[str]:
    lines = []
    for f in spec.subgroup:
        col = f'df["{f.column}"]'
        if f.op == "in":
            lines.append(f"df = df[{col}.isin({list(f.value)!r})]")
        elif isinstance(f.value, str):
            lines.append(f"df = df[{col} {f.op} {f.value!r}]")
        else:
            lines.append(f"df = df[{col} {f.op} {f.value!r}]")
    return lines or ["# all patients"]


def analysis_code(spec: AnalysisSpec, title: str = "") -> str:
    time_col, event_col = ENDPOINTS[spec.endpoint]
    v = spec.variable
    out = [
        f"# {title}" if title else "# Reproduce this analysis",
        "# Needs: pip install pandas numpy statsmodels   (run inside the quorum project for `load`)",
        "import numpy as np",
        "import pandas as pd",
        "from statsmodels.duration.hazard_regression import PHReg",
        "from statsmodels.duration.survfunc import survdiff",
        "from quorum.data import load  # public cBioPortal data, cached on disk",
        "",
        f'df = load("{spec.cohort}")',
        "",
        "# 1. The patients this finding is about",
        *_filter_lines(spec),
        "",
        f"# 2. Survival: {ENDPOINT_WORDS[spec.endpoint]}",
        f'df = df.dropna(subset=["{time_col}", "{event_col}", "{v}"])',
        f'df = df[df["{time_col}"] > 0]',
        "",
    ]
    if spec.exposed is not None:
        out += [
            f"# 3. Compare {spec.exposed!s} with {spec.reference!s}",
            f'df = df[df["{v}"].astype(str).isin([{str(spec.exposed)!r}, {str(spec.reference)!r}])]',
            f'x = (df["{v}"].astype(str) == {str(spec.exposed)!r}).astype(float)  # 1 = {spec.exposed}, 0 = {spec.reference}',
        ]
    else:
        out += [
            f"# 3. {v} as a continuous score, scaled so the hazard ratio is per one standard deviation",
            f'x = (df["{v}"] - df["{v}"].mean()) / df["{v}"].std()',
        ]
    out.append(f'exog = pd.DataFrame({{"{v}": x}})')
    if spec.adjust_for:
        out.append("# Compare like with like: adjust for " + ", ".join(spec.adjust_for))
        for a in spec.adjust_for:
            out.append(f'exog["{a}"] = df["{a}"].astype(float)')
    out += [
        "exog = exog.dropna()",
        "df = df.loc[exog.index]",
        "",
        "# 4. Cox proportional hazards model",
        f'fit = PHReg(df["{time_col}"].to_numpy(), exog.to_numpy(), status=df["{event_col}"].to_numpy()).fit()',
        "beta, se = fit.params[0], fit.bse[0]",
        "hr = np.exp(beta)",
        "ci_low, ci_high = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)",
        "p = fit.pvalues[0]",
        f'deaths = int(df["{event_col}"].sum())',
        'print(f"HR {hr:.2f} (95% CI {ci_low:.2f} to {ci_high:.2f}), p={p:.2g}, n={len(df)}, deaths={deaths}")',
    ]
    if spec.exposed is not None:
        out += [
            "",
            "# 5. Log-rank test: do the two survival curves differ at all?",
            f'_, logrank_p = survdiff(df["{time_col}"], df["{event_col}"], x.loc[df.index].astype(int))',
            'print(f"log-rank p={logrank_p:.2g}")',
        ]
    return "\n".join(out)


def plot_code(spec: AnalysisSpec, names: Optional[dict] = None, title: str = "") -> str:
    """The code that draws the two survival-curve charts shown with a finding."""
    names = names or {}
    v = spec.variable
    if spec.exposed is not None:
        groups = (f'groups = {{\n    {names.get(spec.exposed, str(spec.exposed))!r}: df[df["{v}"].astype(str) == {str(spec.exposed)!r}],\n'
                  f'    {names.get(spec.reference, str(spec.reference))!r}: df[df["{v}"].astype(str) == {str(spec.reference)!r}],\n}}')
    else:
        groups = (f'median = df["{v}"].median()\n'
                  f'groups = {{"higher (above median)": df[df["{v}"] > median],\n'
                  f'          "lower (at or below median)": df[df["{v}"] <= median]}}')
    return "\n".join([
        f"# {title}" if title else "# Survival curves for this finding",
        "# Needs: pip install pandas statsmodels matplotlib",
        "import matplotlib.pyplot as plt",
        "from statsmodels.duration.survfunc import SurvfuncRight",
        "from quorum.data import load",
        "",
        f'df = load("{spec.cohort}")',
        *_filter_lines(spec),
        f'df = df[df["{v}"].notna()]',
        groups,
        "",
        "# Left: deaths from any cause. Right: breast cancer deaths only.",
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)",
        'for ax, (time, event, heading) in zip(axes, [("os_months", "os_event", "Counting all deaths"),',
        '                                            ("dss_months", "dss_event", "Counting breast cancer deaths only")]):',
        "    for label, g in groups.items():",
        "        g = g.dropna(subset=[time, event])",
        "        g = g[g[time] > 0]",
        "        sf = SurvfuncRight(g[time].to_numpy(), g[event].to_numpy())  # Kaplan-Meier estimate",
        '        ax.step([0, *(sf.surv_times / 12)], [1, *sf.surv_prob], where="post", label=f"{label} (n={len(g)})")',
        "    ax.set_title(heading)",
        '    ax.set_xlabel("years since diagnosis")',
        "    ax.set_ylim(0, 1.02)",
        "    ax.legend()",
        'axes[0].set_ylabel("share of patients")',
        "plt.tight_layout()",
        "plt.show()",
    ])


def code_bundle(claim_spec: Optional[AnalysisSpec], evidence: list, names: Optional[dict] = None) -> list[dict]:
    """Every test behind a verdict, as {title, kind, code}, in the order the referees ran them."""
    out = []
    planned = None
    for e in evidence:
        if e.referee == "reanalyst" and e.data.get("planned_spec"):
            planned = AnalysisSpec.model_validate(e.data["planned_spec"])
    spec = planned or claim_spec
    if spec is None:
        return out
    out.append({"title": "As claimed", "kind": "test",
                "code": analysis_code(spec, "Re-checker: the analysis as the finding states it")})
    for e in evidence:
        if e.referee == "adjuster":
            for label, d in e.data.items():
                if isinstance(d, dict) and d.get("spec"):
                    short = "Cancer deaths only" if "breast cancer" in label else "Comparable patients"
                    out.append({"title": short, "kind": "test",
                                "code": analysis_code(AnalysisSpec.model_validate(d["spec"]), f"Fair-comparison check: {label}")})
        if e.referee == "replicator" and isinstance(e.data.get("spec"), dict):
            out.append({"title": "Second study (TCGA)", "kind": "test",
                        "code": analysis_code(AnalysisSpec.model_validate(e.data["spec"]),
                                              "Second opinion: the same analysis on TCGA's separate patients")})
    out.append({"title": "Survival plots", "kind": "plot", "code": plot_code(spec, names)})
    return out
