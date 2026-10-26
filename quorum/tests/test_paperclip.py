"""
The Published-evidence checker, offline: Paperclip's replies are faked so the
rule can be tested. It can push a finding down, never up, and only when the
published record clearly leans against it.
"""

import pytest

from quorum import paperclip
from quorum.claims import Claim, ClaimType


def _ev(pmid, weight, retracted=False):
    return {"evidence": f"passage from {pmid}", "confidence": 0.9, "weight": weight, "section": "Results",
            "paper": {"pmid": pmid, "year": 2020, "title": "t", "design": "cohort", "retracted": retracted}}


@pytest.fixture
def fake(monkeypatch):
    reply = {}
    monkeypatch.setattr(paperclip, "connected", lambda: True)
    monkeypatch.setattr(paperclip, "_post", lambda path, body, timeout=90: reply)
    return reply


CLAIM = Claim(type=ClaimType.DATA, text="Basal-like tumors have worse survival than Luminal A.")


def test_clear_lean_against_contests(fake):
    fake.update({"supports": [_ev("1", 0.2)], "refutes": [_ev("2", 0.9), _ev("3", 0.8), _ev("4", 0.7)],
                 "counts": {"supports": 1, "refutes": 3, "retrieved": 40}})
    e = paperclip.claim_evidence(CLAIM)
    assert e.outcome == "contests" and "PMID 2" in e.detail


def test_agreement_supports_but_cannot_raise_trust(fake):
    fake.update({"supports": [_ev("1", 0.9)], "refutes": [], "counts": {"supports": 4, "refutes": 0, "retrieved": 40}})
    assert paperclip.claim_evidence(CLAIM).outcome == "supports"


def test_one_contrary_paper_is_not_enough(fake):
    fake.update({"supports": [], "refutes": [_ev("2", 0.9)], "counts": {"supports": 0, "refutes": 1, "retrieved": 40}})
    assert paperclip.claim_evidence(CLAIM).outcome == "no_evidence"


def test_retracted_papers_do_not_count(fake):
    fake.update({"supports": [], "refutes": [_ev(str(i), 0.9, retracted=True) for i in range(3)],
                 "counts": {"supports": 0, "refutes": 3, "refutes_retracted": 3, "retrieved": 40}})
    assert paperclip.claim_evidence(CLAIM).outcome == "no_evidence"


def test_not_connected_is_harmless(monkeypatch):
    monkeypatch.setattr(paperclip, "connected", lambda: False)
    assert paperclip.claim_evidence(CLAIM).outcome == "no_evidence"


def test_search_maps_hits_to_papers(monkeypatch):
    monkeypatch.setattr(paperclip, "_post", lambda path, body, timeout=90: {"papers": [
        {"id": "PMC1", "object": "paper", "title": "A paper.", "url": "https://paperclip.gxl.ai/papers/PMC1",
         "pmid": "123", "year": 2021, "authors": ["Lee A", "Kim B"], "identifiers": {"pmc": "PMC1"},
         "abstract": "Survival differs."},
        {"id": "NCT1", "object": "trial", "title": "A trial", "url": "u"}]})
    papers = paperclip.search("survival", n=2)
    assert len(papers) == 1 and papers[0]["pmid"] == "123" and papers[0]["pmcid"] == "PMC1"
    assert papers[0]["authors"] == "Lee A et al." and papers[0]["links"]["paperclip"].endswith("PMC1")
