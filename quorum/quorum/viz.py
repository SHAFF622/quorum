"""
Numbers for the website's charts.

- Kaplan-Meier curves: the share of patients still alive (or not yet dead of
  breast cancer) over time, one line per group. The classic picture in cancer
  research: if one group's line drops faster, that group does worse.
- Dataset overview counts, in the style of cBioPortal's study summary.

All curves are thinned to at most ~120 points so the browser draws them fast.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.duration.survfunc import SurvfuncRight

from quorum.survival import ENDPOINTS, AnalysisSpec, apply_filters

MAX_POINTS = 120


def km(times: pd.Series, events: pd.Series) -> dict:
    """One Kaplan-Meier curve as step points: months -> share surviving."""
    ok = times.notna() & events.notna() & (times > 0)
    t, e = times[ok].to_numpy(float), events[ok].to_numpy(float)
    if len(t) < 5:
        return {"t": [], "s": [], "n": int(len(t)), "events": int(e.sum()) if len(e) else 0}
    sf = SurvfuncRight(t, e)
    xs, ys = sf.surv_times, sf.surv_prob
    if len(xs) > MAX_POINTS:
        idx = np.unique(np.linspace(0, len(xs) - 1, MAX_POINTS).astype(int))
        xs, ys = xs[idx], ys[idx]
    return {"t": [0.0] + [round(float(x), 1) for x in xs], "s": [1.0] + [round(float(y), 4) for y in ys],
            "n": int(len(t)), "events": int(e.sum()), "max_t": round(float(t.max()), 1)}


def groups_for(df: pd.DataFrame, spec: AnalysisSpec, names: Optional[dict] = None) -> list[tuple[str, pd.DataFrame]]:
    """Split the patients a finding is about into the groups it compares."""
    data = apply_filters(df, spec.subgroup)
    data = data[data[spec.variable].notna()]
    if spec.exposed is not None:
        names = names or {}
        col = data[spec.variable].astype(str)
        return [(names.get(spec.exposed, str(spec.exposed)), data[col == str(spec.exposed)]),
                (names.get(spec.reference, str(spec.reference)), data[col == str(spec.reference)])]
    median = data[spec.variable].median()
    return [(f"higher (above median)", data[data[spec.variable] > median]),
            (f"lower (at or below median)", data[data[spec.variable] <= median])]


def finding_curves(df: pd.DataFrame, spec: AnalysisSpec, names: Optional[dict] = None) -> dict:
    """Curves for both ways of counting deaths, so the page can show them side by side."""
    out = {}
    for endpoint, (tcol, ecol) in ENDPOINTS.items():
        if tcol not in df.columns or ecol not in df.columns or df[ecol].notna().mean() < 0.5:
            continue
        out[endpoint] = [{"label": label, **km(g[tcol], g[ecol])} for label, g in groups_for(df, spec, names)]
    return out


def _counts(s: pd.Series, top: int = 8, rename: Optional[dict] = None) -> list[dict]:
    s = s.dropna().astype(str)
    if rename:
        s = s.map(lambda v: rename.get(v, v))
    vc = s.value_counts()
    rows = [{"label": k, "n": int(v)} for k, v in vc.head(top).items()]
    if len(vc) > top:
        rows.append({"label": "other", "n": int(vc.iloc[top:].sum())})
    return rows


def overview(df: pd.DataFrame) -> dict:
    surv = df.dropna(subset=["os_months", "os_event"])
    deaths = int(surv["os_event"].sum())
    out = {
        "patients": int(len(df)),
        "with_survival": int(len(surv)),
        "deaths": deaths,
        "median_follow_up_months": round(float(surv["os_months"].median()), 1) if len(surv) else None,
        "km_overall": km(df["os_months"], df["os_event"]),
        "age_hist": [],
        "panels": [],
    }
    if "dss_event" in df.columns and df["dss_event"].notna().mean() > 0.5:
        cancer = int(df.loc[df["os_event"] == 1, "dss_event"].sum())
        out["deaths_by_cause"] = [{"label": "breast cancer", "n": cancer}, {"label": "other causes", "n": deaths - cancer}]
        out["km_dss"] = km(df["dss_months"], df["dss_event"])
    if "age" in df.columns:
        ages = df["age"].dropna()
        edges = list(range(20, 101, 5))
        hist, _ = np.histogram(ages.clip(20, 99.9), bins=edges)
        out["age_hist"] = [{"from": a, "to": a + 5, "n": int(n)} for a, n in zip(edges[:-1], hist)]
    friendly = {"LumA": "Luminal A", "LumB": "Luminal B", "Basal": "basal-like", "Her2": "HER2-enriched",
                "Normal": "normal-like", "1.0": "mutated", "0.0": "not mutated", "1": "grade 1", "2": "grade 2",
                "3": "grade 3"}
    for col, title in (("er_status", "ER status"), ("pam50", "Molecular subtype"), ("grade", "Tumor grade"),
                       ("her2_status", "HER2 status"), ("tp53_mutated", "TP53 mutation"),
                       ("hormone_therapy", "Hormone therapy"), ("race", "Race"), ("ancestry", "Genetic ancestry")):
        if col in df.columns and df[col].notna().mean() > 0.3:
            vals = df[col]
            if col == "grade":
                vals = vals.dropna().astype(int).astype(str)
            out["panels"].append({"title": title, "rows": _counts(vals, rename=friendly)})
    return out
