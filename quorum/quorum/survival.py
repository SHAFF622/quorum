"""
Reference survival analysis: the trusted, plain-code way to test a claim.

The Re-analyst referee calls these functions instead of trusting the
explorer's own code (the "recompute every reported number" rule from Google's
aptamer paper). Each function returns the same small result record, so a claim
and its re-check can be compared field by field.

Two tests:
- Cox proportional hazards: a hazard ratio (HR). HR > 1 means the exposed
  group dies sooner; HR < 1 means it survives longer.
- Log-rank: whether two survival curves differ at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import survdiff


# Endpoint -> (time column, event column).
# OS: death from any cause. DSS: death from breast cancer only.
ENDPOINTS = {"OS": ("os_months", "os_event"), "DSS": ("dss_months", "dss_event")}


class Filter(BaseModel):
    """One condition that selects a patient subgroup, e.g. pam50 == LumA."""

    column: str
    op: Literal["==", "!=", "<", "<=", ">", ">=", "in"]
    value: object


class AnalysisSpec(BaseModel):
    """
    Exactly what was compared, so anyone can re-run it.

    For a categorical variable, `exposed` and `reference` name the two levels
    compared (e.g. pam50: Basal vs LumA). For a continuous variable, leave both
    empty: the HR is then per one standard deviation of that variable.
    """

    cohort: str = "metabric"  # metabric, tcga, or uploaded
    variable: str
    exposed: Optional[str] = None
    reference: Optional[str] = None
    subgroup: list[Filter] = Field(default_factory=list)
    adjust_for: list[str] = Field(default_factory=list)
    endpoint: Literal["OS", "DSS"] = "OS"


@dataclass
class SurvivalResult:
    hr: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    events: int
    events_exposed: Optional[int] = None
    events_reference: Optional[int] = None
    logrank_p: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    @property
    def direction(self) -> str:
        """'worse' if the exposed group (or higher values) dies sooner."""
        return "worse" if self.hr > 1 else "better"


def apply_filters(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    out = df
    for f in filters:
        col = out[f.column]
        if f.op == "==":
            out = out[col == f.value]
        elif f.op == "!=":
            out = out[col != f.value]
        elif f.op == "in":
            out = out[col.isin(list(f.value))]
        else:
            v = float(f.value)
            out = out[{"<": col < v, "<=": col <= v, ">": col > v, ">=": col >= v}[f.op]]
    return out


def _numeric_covariate(df: pd.DataFrame, name: str) -> pd.Series:
    s = df[name]
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    raise ValueError(
        f"Adjustment variable {name!r} is not numeric; add it as its own comparison instead."
    )


def analyse(df: pd.DataFrame, spec: AnalysisSpec) -> SurvivalResult:
    """Run the Cox model (and log-rank for two groups) that a spec describes."""
    time_col, event_col = ENDPOINTS[spec.endpoint]
    data = apply_filters(df, spec.subgroup)
    if time_col not in data.columns or event_col not in data.columns:
        raise ValueError(f"Endpoint {spec.endpoint} is not available in this cohort.")
    data = data.dropna(subset=[time_col, event_col, spec.variable])
    data = data[data[time_col] > 0]
    notes: list[str] = []

    categorical = spec.exposed is not None or spec.reference is not None
    if categorical:
        if spec.exposed is None or spec.reference is None:
            raise ValueError("A categorical comparison needs both exposed and reference levels.")
        values = data[spec.variable].astype(str)
        data = data[values.isin([str(spec.exposed), str(spec.reference)])]
        x = (data[spec.variable].astype(str) == str(spec.exposed)).astype(float)
        x.name = spec.variable
    else:
        raw = data[spec.variable].astype(float)
        sd = raw.std()
        if not sd or np.isnan(sd):
            raise ValueError(f"{spec.variable!r} has no variation in this subgroup.")
        x = (raw - raw.mean()) / sd
        notes.append("HR is per one standard deviation.")

    exog = pd.DataFrame({spec.variable: x})
    for adj in spec.adjust_for:
        exog[adj] = _numeric_covariate(data, adj)
    exog = exog.dropna()
    data = data.loc[exog.index]

    n = len(data)
    events = int(data[event_col].sum())
    if n < 10 or events < 5:
        raise ValueError(f"Too few patients to analyse: n={n}, deaths={events}.")

    fit = PHReg(data[time_col].to_numpy(), exog.to_numpy(), status=data[event_col].to_numpy()).fit()
    beta, se = fit.params[0], fit.bse[0]
    result = SurvivalResult(
        hr=float(np.exp(beta)),
        ci_low=float(np.exp(beta - 1.96 * se)),
        ci_high=float(np.exp(beta + 1.96 * se)),
        p_value=float(fit.pvalues[0]),
        n=n,
        events=events,
        notes=notes,
    )

    if categorical:
        exposed = x.loc[data.index] == 1.0
        result.events_exposed = int(data.loc[exposed, event_col].sum())
        result.events_reference = int(data.loc[~exposed, event_col].sum())
        _, p = survdiff(data[time_col], data[event_col], exposed.astype(int))
        result.logrank_p = float(p)
    return result
