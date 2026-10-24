"""
Core tests on synthetic patients, so they run offline in a second.

The fake cohort has a known truth: group "B" dies about twice as fast as "A",
and the column "noise" has no effect at all.
"""

import numpy as np
import pandas as pd
import pytest

from quorum.board import ClaimBoard
from quorum.claims import Claim, ClaimType, Evidence, Label, Stated
from quorum.reanalyst import reanalyse
from quorum.rules import check
from quorum.survival import AnalysisSpec, Filter, analyse
from quorum.verdict import RULE_VERSION, decide


@pytest.fixture(scope="module")
def cohort():
    rng = np.random.default_rng(7)
    n = 800
    group = rng.choice(["A", "B"], size=n)
    rate = np.where(group == "B", 2.0, 1.0) / 60  # B: hazard ratio 2
    time = rng.exponential(1 / rate)
    censor = rng.uniform(0, 120, size=n)
    return pd.DataFrame({
        "cohort": "metabric",
        "group": group,
        "noise": rng.normal(size=n),
        "age": rng.uniform(30, 80, size=n),
        "os_months": np.minimum(time, censor),
        "os_event": (time <= censor).astype(float),
    })


def spec(**kw):
    return AnalysisSpec(variable="group", exposed="B", reference="A", **kw)


def test_survival_recovers_known_hazard_ratio(cohort):
    r = analyse(cohort, spec())
    assert 1.6 < r.hr < 2.5
    assert r.p_value < 1e-6
    assert r.direction == "worse"
    assert r.events_exposed + r.events_reference == r.events


def test_survival_finds_no_effect_for_noise(cohort):
    r = analyse(cohort, AnalysisSpec(variable="noise"))
    assert r.ci_low < 1 < r.ci_high


def test_subgroup_filter(cohort):
    r = analyse(cohort, spec(subgroup=[Filter(column="age", op="<", value=50)]))
    assert r.n < len(cohort)


def _claim(text, **stated):
    return Claim(type=ClaimType.DATA, text=text, spec=spec(), stated=Stated(**stated))


def test_reanalyst_supports_a_true_claim(cohort):
    ev, _ = reanalyse(_claim("B has worse survival.", direction="worse"), {"metabric": cohort})
    assert ev.outcome == "supports"


def test_reanalyst_refutes_flipped_direction(cohort):
    ev, _ = reanalyse(_claim("B has better survival.", direction="better"), {"metabric": cohort})
    assert ev.outcome == "refutes"
    assert "Direction" in ev.detail


def test_reanalyst_refutes_inflated_hr(cohort):
    ev, _ = reanalyse(_claim("B has worse survival (HR 6).", direction="worse", hr=6.0), {"metabric": cohort})
    assert ev.outcome == "refutes"


def test_reanalyst_refutes_noise_presented_as_signal(cohort):
    c = Claim(type=ClaimType.DATA, text="Noise predicts worse survival.",
              spec=AnalysisSpec(variable="noise"), stated=Stated(direction="worse", p_value=0.001))
    ev, _ = reanalyse(c, {"metabric": cohort})
    assert ev.outcome == "refutes"


def test_reanalyst_cannot_judge_interpretation():
    c = Claim(type=ClaimType.INTERPRETATION, text="This suggests endocrine resistance.")
    ev, ref = reanalyse(c, {})
    assert ev.outcome == "no_evidence" and ref is None


@pytest.mark.parametrize("text,rule", [
    ("Group B causes worse survival.", "causal_language"),
    ("Group B drives mortality and is associated with worse survival.", "causal_language"),
])
def test_rules_flag_causal_words(text, rule):
    found = {e.data["rule"] for e in check(_claim(text, direction="worse"))}
    assert rule in found


def test_rules_flag_strong_words_on_small_effect():
    found = check(_claim("B is a major driver of worse survival.", direction="worse", hr=1.1))
    assert any(e.data["rule"] == "overclaim_small_effect" for e in found)


def test_rules_leave_a_plain_claim_alone(cohort):
    c = _claim("B has worse overall survival than A.", direction="worse")
    _, ref = reanalyse(c, {"metabric": cohort})
    assert check(c, ref) == []


def test_rules_flag_multiple_testing():
    found = check(_claim("B stood out among 40 features.", direction="worse", p_value=0.01, tests_run=40))
    assert any(e.data["rule"] == "multiple_testing" for e in found)


def test_rules_flag_underpowered(cohort):
    small = cohort.head(60)
    c = _claim("B has worse survival.", direction="worse")
    _, ref = reanalyse(c, {"metabric": small})
    if ref is not None:
        assert any(e.data["rule"] == "too_few_events" for e in check(c, ref))


def test_verdict_model_can_lower_but_never_raise_trust():
    claim = _claim("x")
    # An AI referee saying "supports" alone is not enough to verify.
    v = decide(claim, [Evidence(referee="skeptic", outcome="supports")])
    assert v.label == Label.UNVERIFIABLE
    # Data reproduction verifies; an AI objection then pulls it down.
    v = decide(claim, [Evidence(referee="reanalyst", outcome="supports"),
                       Evidence(referee="skeptic", outcome="contests")])
    assert v.label == Label.CONTESTED


def test_verdict_robust_needs_two_runs():
    claim = _claim("x")
    ev = [Evidence(referee="reanalyst", outcome="supports")]
    assert decide(claim, ev, runs_found_in=1).label == Label.VERIFIED
    assert decide(claim, ev, runs_found_in=2).label == Label.ROBUST


def test_verdict_records_frozen_rule_version():
    v = decide(_claim("x"), [Evidence(referee="reanalyst", outcome="refutes", detail="no")])
    assert v.label == Label.REFUTED and v.rule_version == RULE_VERSION


def test_board_is_append_only_and_tracks_unjudged():
    board = ClaimBoard()
    c = board.post_claim(_claim("B has worse survival.", direction="worse"))
    assert [x.claim_id for x in board.unjudged()] == [c.claim_id]
    board.post_evidence(c.claim_id, Evidence(referee="reanalyst", outcome="supports"))
    board.post_verdict(decide(c, board.evidence(c.claim_id)))
    assert list(board.unjudged()) == []
    assert board.latest_verdict(c.claim_id).label == Label.VERIFIED
    rid = board.post_request("Check in ER-positive only", claim_id=c.claim_id)
    assert board.open_requests()[0]["id"] == rid
    board.close_request(rid)
    assert board.open_requests() == []


def test_content_hash_ignores_storage_details():
    a = _claim("same", direction="worse")
    b = a.model_copy(update={"claim_id": "other", "created_at": "later"})
    assert a.content_hash() == b.content_hash()
