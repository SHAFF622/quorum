"""Tests for the replicator, the cancer-death endpoint, adapters and the HTML board."""

import json

import numpy as np
import pandas as pd
import pytest

from quorum.adapters import from_jsonl, from_kosmos
from quorum.claims import Claim, ClaimType, Stated
from quorum.html import render
from quorum.replicator import replicate, to_tcga
from quorum.rules import check
from quorum.survival import AnalysisSpec, Filter, analyse


def make_cohort(name, n, hr, seed):
    rng = np.random.default_rng(seed)
    pam50 = rng.choice(["LumA", "Basal"], size=n)
    rate = np.where(pam50 == "Basal", hr, 1.0) / 60
    t = rng.exponential(1 / rate)
    c = rng.uniform(0, 120, n)
    died = (t <= c).astype(float)
    # Half of deaths are "from other causes": they count for OS, not DSS.
    cancer = died * (rng.uniform(size=n) < 0.5)
    return pd.DataFrame({"cohort": name, "pam50": pam50, "age": rng.uniform(30, 80, n),
                         "os_months": np.minimum(t, c), "os_event": died,
                         "dss_months": np.minimum(t, c), "dss_event": cancer})


SPEC = AnalysisSpec(variable="pam50", exposed="Basal", reference="LumA")


def test_dss_counts_only_cancer_deaths():
    df = make_cohort("metabric", 800, 2.0, 1)
    os_ = analyse(df, SPEC)
    dss = analyse(df, SPEC.model_copy(update={"endpoint": "DSS"}))
    assert dss.events < os_.events
    assert dss.direction == "worse"


def test_replicator_supports_when_tcga_agrees():
    cohorts = {"metabric": make_cohort("metabric", 800, 2.0, 1), "tcga": make_cohort("tcga", 800, 2.0, 2)}
    ev = replicate(SPEC, analyse(cohorts["metabric"], SPEC), cohorts)
    assert ev.outcome == "supports"


def test_replicator_contests_when_tcga_disagrees():
    cohorts = {"metabric": make_cohort("metabric", 800, 2.0, 1), "tcga": make_cohort("tcga", 800, 0.4, 2)}
    ev = replicate(SPEC, analyse(cohorts["metabric"], SPEC), cohorts)
    assert ev.outcome == "contests"


def test_replicator_never_refutes_and_skips_missing_columns():
    spec, notes = to_tcga(AnalysisSpec(variable="er_status", exposed="Negative", reference="Positive"))
    assert spec is None and "not recorded" in notes[0]
    spec, notes = to_tcga(AnalysisSpec(variable="esr1_z", subgroup=[Filter(column="er_status", op="==", value="Positive")]))
    assert spec.subgroup[0].column == "pam50" and "luminal" in notes[0]


def test_rules_dss_is_a_different_question_not_an_error():
    c = Claim(type=ClaimType.DATA, text="Basal tumors have worse breast cancer survival.",
              spec=SPEC.model_copy(update={"endpoint": "DSS"}), stated=Stated(direction="worse"))
    found = {e.data["rule"]: e.outcome for e in check(c, declared_endpoint="OS")}
    assert found["changed_endpoint"] == "contests"
    # Without a declared endpoint, counting cancer deaths is a fair answer: no flag.
    assert "changed_endpoint" not in {e.data["rule"] for e in check(c)}


def test_rules_catch_cancer_specific_wording_on_overall_survival():
    c = Claim(type=ClaimType.DATA, text="Basal tumors have worse cancer-specific survival.",
              spec=SPEC, stated=Stated(direction="worse"))
    assert any(e.data["rule"] == "endpoint_mismatch" and e.outcome == "refutes" for e in check(c))


def test_kosmos_adapter_splits_data_and_interpretation(tmp_path):
    d = tmp_path / "run_a" / "cycle_1"
    d.mkdir(parents=True)
    (d / "task_0_finding.json").write_text(json.dumps({
        "finding_id": "cycle1_task0", "summary": "Basal tumors have worse overall survival.",
        "statistics": {"hazard_ratio": 1.31, "p_value": 0.011, "n": 909, "significant": True},
        "interpretation": "Basal biology drives early mortality.",
        "citations": [{"pmid": "22522925"}], "code_provenance": {"code_snippet": "cph.fit(df)"},
    }))
    claims = from_kosmos(tmp_path / "run_a")
    assert [c.type for c in claims] == [ClaimType.DATA, ClaimType.INTERPRETATION]
    assert "hazard_ratio=1.31" in claims[0].text and "significant" not in claims[0].text
    assert claims[0].citations == ["22522925"] and claims[0].provenance.run_id == "run_a"


def test_jsonl_adapter(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"text": "A claim.", "type": "interpretation", "run_id": "r2"}\n\n')
    (c,) = from_jsonl(p)
    assert c.type == ClaimType.INTERPRETATION and c.provenance.run_id == "r2"


def test_html_board_renders_and_escapes(tmp_path):
    item = {"claim": {"type": "data", "text": "<script>x</script> Basal worse", "provenance": {"run_id": "run1"}},
            "evidence": [{"referee": "reanalyst", "outcome": "supports", "detail": "Reproduced."}],
            "verdict": {"label": "verified", "reasons": ["Reproduced from raw data."]}}
    (tmp_path / "claims.json").write_text(json.dumps([item]))
    (tmp_path / "summary.json").write_text(json.dumps({"claims": 1, "wordings": ["w"], "llm": {"spent_usd": 0.1}}))
    page = render(tmp_path).read_text()
    assert "act on (1)" in page and "<script>x" not in page and "&lt;script&gt;" in page


def test_librarian(monkeypatch):
    from types import SimpleNamespace
    from quorum.librarian import check_citations
    from quorum.llm import LLM

    class Client:
        def __init__(self, obj): self.obj = obj; self.messages = self
        def create(self, **kw):
            u = SimpleNamespace(input_tokens=10, output_tokens=10, cache_read_input_tokens=0, cache_creation_input_tokens=0)
            return SimpleNamespace(model="claude-sonnet-5-5", stop_reason="end_turn", usage=u,
                                   content=[SimpleNamespace(type="text", text=json.dumps(self.obj))])

    papers = {"1": {"pmid": "1", "title": "t", "abstract": "a"}}
    fetch = lambda pmid: papers.get(pmid)
    claim = Claim(text="x", citations=["1"])
    weak = {"papers": [{"pmid": "1", "judgement": "unrelated_or_weak", "reason": "off topic"}]}
    assert check_citations(claim, LLM(budget_usd=1, client=Client(weak)), fetch).outcome == "contests"
    ok = {"papers": [{"pmid": "1", "judgement": "supports", "reason": "yes"}]}
    assert check_citations(claim, LLM(budget_usd=1, client=Client(ok)), fetch).outcome == "supports"
    fake = Claim(text="x", citations=["999"])
    assert check_citations(fake, LLM(budget_usd=1, client=Client(ok)), fetch).outcome == "refutes"
    assert check_citations(Claim(text="x"), LLM(budget_usd=1, client=Client(ok)), fetch).outcome == "no_evidence"
