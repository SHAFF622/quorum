"""
The code shown on the website must reproduce the website's numbers.

Each generated script is executed against synthetic patients (standing in for
`quorum.data.load`) and its printed hazard ratio, confidence interval and
p-value are compared with Quorum's own analysis of the same data.
"""

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

import quorum.data
from quorum.codegen import analysis_code, plot_code
from quorum.survival import AnalysisSpec, Filter, analyse


@pytest.fixture
def cohort(monkeypatch):
    rng = np.random.default_rng(11)
    n = 900
    df = pd.DataFrame({
        "cohort": "metabric",
        "pam50": rng.choice(["LumA", "Basal"], size=n),
        "er_status": rng.choice(["Positive", "Negative"], size=n, p=[0.7, 0.3]),
        "age": rng.uniform(30, 85, size=n),
        "grade": rng.choice([1.0, 2.0, 3.0], size=n),
        "esr1_z": rng.normal(size=n),
    })
    rate = np.where(df["pam50"] == "Basal", 1.8, 1.0) * np.exp(0.02 * (df["age"] - 60)) / 80
    t = rng.exponential(1 / rate)
    c = rng.uniform(0, 150, n)
    df["os_months"] = np.minimum(t, c)
    df["os_event"] = (t <= c).astype(float)
    df["dss_months"] = df["os_months"]
    df["dss_event"] = df["os_event"] * (rng.uniform(size=n) < 0.6)
    monkeypatch.setattr(quorum.data, "load", lambda name: df.copy())
    return df


def run(code: str) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "<generated>", "exec"), {})
    return out.getvalue()


@pytest.mark.parametrize("spec", [
    AnalysisSpec(variable="pam50", exposed="Basal", reference="LumA"),
    AnalysisSpec(variable="esr1_z", subgroup=[Filter(column="er_status", op="==", value="Positive")]),
    AnalysisSpec(variable="esr1_z", subgroup=[Filter(column="er_status", op="==", value="Positive")],
                 adjust_for=["age", "grade"], endpoint="DSS"),
    AnalysisSpec(variable="pam50", exposed="Basal", reference="LumA",
                 subgroup=[Filter(column="age", op="<", value=50)]),
])
def test_generated_code_reproduces_the_numbers(cohort, spec):
    printed = run(analysis_code(spec, "test"))
    r = analyse(cohort, spec)
    expected = f"HR {r.hr:.2f} (95% CI {r.ci_low:.2f} to {r.ci_high:.2f}), p={r.p_value:.2g}, n={r.n}, deaths={r.events}"
    assert expected in printed
    if spec.exposed is not None:
        assert f"log-rank p={r.logrank_p:.2g}" in printed


def test_plot_code_is_valid_python():
    spec = AnalysisSpec(variable="esr1_z", subgroup=[Filter(column="er_status", op="==", value="Positive")])
    compile(plot_code(spec), "<plot>", "exec")
