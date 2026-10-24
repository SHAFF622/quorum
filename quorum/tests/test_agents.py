"""
The AI parts, tested with a fake Claude so nothing costs money.

The fake client returns scripted responses: tool calls for the explorer, and
JSON for the referees. This proves the plumbing (tool loop, blind re-analysis,
verdicts, budget stop) without judging the model itself.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quorum.agents import blind_reanalysis, explore
from quorum.claims import Claim, ClaimType, Label
from quorum.llm import LLM, BudgetExceeded
from quorum.panel import blind_copy, judge


def usage():
    return SimpleNamespace(input_tokens=1000, output_tokens=200,
                           cache_read_input_tokens=0, cache_creation_input_tokens=0)


def text_response(obj):
    return SimpleNamespace(model="claude-sonnet-5-5", stop_reason="end_turn", usage=usage(),
                           content=[SimpleNamespace(type="text", text=json.dumps(obj))])


def tool_response(name, inp, uid="t1"):
    return SimpleNamespace(model="claude-sonnet-5-5", stop_reason="tool_use", usage=usage(),
                           content=[SimpleNamespace(type="tool_use", name=name, input=inp, id=uid)])


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.script.pop(0)


@pytest.fixture(scope="module")
def cohorts():
    rng = np.random.default_rng(3)
    n = 600
    pam50 = rng.choice(["LumA", "Basal"], size=n)
    rate = np.where(pam50 == "Basal", 2.0, 1.0) / 60
    t = rng.exponential(1 / rate)
    c = rng.uniform(0, 120, n)
    df = pd.DataFrame({"cohort": "metabric", "pam50": pam50, "age": rng.uniform(30, 80, n),
                       "os_months": np.minimum(t, c), "os_event": (t <= c).astype(float)})
    df.index.name = "patient_id"
    return {"metabric": df}


SPEC = {"cohort": "metabric", "variable": "pam50", "exposed": "Basal", "reference": "LumA",
        "subgroup": [], "adjust_for": []}


def test_explorer_runs_tools_then_submits(cohorts):
    client = FakeClient([
        tool_response("run_survival_analysis", SPEC),
        tool_response("submit_claims", {"claims": [
            {"text": "Basal tumors have worse overall survival than Luminal A (HR 2.0).",
             "type": "data", "analysis_index": 0},
            {"text": "This suggests basal biology drives early death.", "type": "interpretation",
             "analysis_index": -1},
        ]}, uid="t2"),
    ])
    llm = LLM(budget_usd=1, client=client, use_fallbacks=False)
    claims = explore("goal", cohorts, llm, run_id="run1")
    assert [c.type for c in claims] == [ClaimType.DATA, ClaimType.INTERPRETATION]
    assert claims[0].stated.tests_run == 1
    assert "Basal" in claims[0].provenance.code  # the explorer's analysis is kept for audit
    # The tool result went back to the model.
    # (messages = goal, tool call, tool result, ...; the list is shared, so index it)
    assert client.requests[1]["messages"][2]["content"][0]["type"] == "tool_result"


def test_blind_copy_hides_the_explorers_analysis():
    c = Claim(text="x", stated={"tests_run": 5, "hr": 2.0})
    seen = blind_copy(c)
    assert seen.spec is None and seen.stated.hr is None and seen.stated.tests_run == 5


def test_blind_reanalysis_catches_a_flipped_claim(cohorts):
    plan = {"checkable": True, "reason": "", "spec": SPEC,
            "stated": {"direction": "better", "hr": 0.5, "p_value": -1, "n": -1}}
    llm = LLM(budget_usd=1, client=FakeClient([text_response(plan)]), use_fallbacks=False)
    claim = Claim(text="Basal tumors have better survival than Luminal A (HR 0.5).")
    ev, _ = blind_reanalysis(claim, cohorts, llm)
    assert ev.outcome == "refutes"


def test_panel_skeptic_can_lower_a_reproduced_claim(cohorts):
    plan = {"checkable": True, "reason": "", "spec": SPEC,
            "stated": {"direction": "worse", "hr": -1, "p_value": -1, "n": -1}}
    skeptic = {"outcome": "contests", "problem": "confounding", "reason": "Not adjusted for age."}
    llm = LLM(budget_usd=1, client=FakeClient([text_response(plan), text_response(skeptic)]),
              use_fallbacks=False)
    v = judge(Claim(text="Basal tumors have worse overall survival than Luminal A."), cohorts, llm)
    assert v.label == Label.CONTESTED


def test_budget_stops_calls_before_they_are_sent(cohorts):
    client = FakeClient([text_response({})] * 5)
    llm = LLM(budget_usd=0.000001, client=client, use_fallbacks=False)
    llm.spent_usd = 1.0
    with pytest.raises(BudgetExceeded):
        llm.create(model="claude-sonnet-5-5", system="s", messages=[], prompt_version="v")
    assert client.requests == []


def test_fallbacks_are_requested_by_default():
    client = FakeClient([text_response({"a": 1})])
    LLM(budget_usd=1, client=client).create(model="claude-sonnet-5-5", system="s",
                                            messages=[{"role": "user", "content": "hi"}], prompt_version="v")
    req = client.requests[0]
    assert req["extra_body"] == {"fallbacks": "default"}
    assert req["extra_headers"]["anthropic-beta"] == "server-side-fallback-2026-07-01"
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
