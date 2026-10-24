"""
The referee panel: every referee looks at a claim, then plain code decides.

Order matters only for what each referee is allowed to see:
1. Blind Re-analyst: claim text only -> plans a check -> plain code re-runs it.
2. Stats rules: plain code, on the claim and the re-analysis numbers.
3. Replicator: re-runs a reproduced METABRIC finding on TCGA's separate patients.
4. Librarian: if the claim cites papers, checks each abstract supports it.
5. Skeptic: the claim, the re-analysis numbers and the rule flags (never the
   explorer's own analysis or reasoning).
Then `verdict.decide` turns the evidence into a label.

Consensus (merging repeated runs) happens afterwards, across the whole board.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from quorum import rules
from quorum.agents import blind_reanalysis, skeptic
from quorum.claims import Claim, Evidence, Stated, Verdict
from quorum.llm import LLM
from quorum.adjuster import fair_comparison
from quorum.librarian import check_citations
from quorum.replicator import replicate
from quorum.reanalyst import reanalyse
from quorum.verdict import decide


def blind_copy(claim: Claim) -> Claim:
    """What referees may see: the text and run metadata, not the explorer's analysis."""
    return claim.model_copy(update={
        "spec": None,
        "stated": Stated(tests_run=claim.stated.tests_run),
    })


def judge(claim: Claim, cohorts: dict[str, pd.DataFrame], llm: LLM | None, use_skeptic: bool = True,
          on_evidence=None, declared_endpoint: str | None = None, context: list[str] | None = None) -> Verdict:
    """
    Referee one claim. With `llm=None` (no API key) the AI referees are skipped:
    the re-checker uses the claim's own stated analysis instead of planning one
    blind, and there is no skeptic or librarian. Everything else is identical.
    """
    seen = blind_copy(claim)
    evidence: list[Evidence] = []

    def add(e: Evidence) -> None:
        evidence.append(e)
        if on_evidence:
            on_evidence(e)

    if llm is None:
        reanalysis, ref = reanalyse(claim, cohorts)
        reanalysis.data["mode"] = "given the explorer's stated analysis (no-AI mode)"
        if claim.spec is not None:
            reanalysis.data["planned_spec"] = claim.spec.model_dump(mode="json")
        planned_claim = claim
    else:
        reanalysis, planned = blind_reanalysis(seen, cohorts, llm)
        planned_claim, ref = planned if planned else (seen, None)
    add(reanalysis)

    flags = rules.check(planned_claim, ref, declared_endpoint=declared_endpoint)
    for f in flags:
        add(f)
    if ref is not None and reanalysis.outcome == "supports":
        add(fair_comparison(planned_claim.spec, ref, cohorts))
        add(replicate(planned_claim.spec, ref, cohorts))
    if claim.citations and llm is not None:
        add(check_citations(seen, llm))
    if use_skeptic and llm is not None:
        add(skeptic(seen, reanalysis, flags, llm, context=context if claim.type.value == "interpretation" else None))

    return decide(seen, evidence)


def consensus_key(verdict: Verdict) -> tuple | None:
    """Two claims are 'the same finding' if the re-analyst planned the same check."""
    for e in verdict.evidence:
        if e.referee == "reanalyst" and "planned_spec" in e.data and e.outcome == "supports":
            s = e.data["planned_spec"]
            subgroup = tuple(sorted((f["column"], f["op"], str(f["value"])) for f in s.get("subgroup", [])))
            return (s["cohort"], s["variable"], s.get("exposed"), s.get("reference"), subgroup,
                    tuple(sorted(s.get("adjust_for", []))), e.data.get("hr", 1) > 1)
    return None


def apply_consensus(claims: list[Claim], verdicts: dict[str, Verdict]) -> dict[str, Verdict]:
    """Re-decide each verified claim knowing how many independent runs found it."""
    runs_by_key: dict[tuple, set] = defaultdict(set)
    for c in claims:
        key = consensus_key(verdicts[c.claim_id])
        if key is not None:
            runs_by_key[key].add(c.provenance.run_id)
    out = {}
    for c in claims:
        v = verdicts[c.claim_id]
        key = consensus_key(v)
        runs = len(runs_by_key[key]) if key is not None else 1
        evidence = [e for e in v.evidence if e.referee != "consensus"]
        evidence.append(Evidence(referee="consensus", outcome="supports" if runs >= 2 else "no_evidence",
                                 detail=f"Found in {runs} independent run(s).", data={"runs": runs}))
        out[c.claim_id] = decide(blind_copy(c), evidence, runs_found_in=runs)
    return out


def agreement(claims: list[Claim], verdicts: dict[str, Verdict]) -> dict:
    """How much independent runs agree, before and after refereeing (Jaccard overlap)."""
    by_run_all: dict[str, set] = defaultdict(set)
    by_run_verified: dict[str, set] = defaultdict(set)
    for c in claims:
        key = consensus_key(verdicts[c.claim_id])
        if key is None:
            continue
        by_run_all[c.provenance.run_id].add(key)
        if verdicts[c.claim_id].label.value in ("verified", "robust"):
            by_run_verified[c.provenance.run_id].add(key)

    def mean_jaccard(groups: dict[str, set]) -> float | None:
        runs = list(groups.values())
        pairs = [(a, b) for i, a in enumerate(runs) for b in runs[i + 1:]]
        if not pairs:
            return None
        return sum(len(a & b) / len(a | b) if a | b else 0 for a, b in pairs) / len(pairs)

    return {"checkable_findings": mean_jaccard(by_run_all), "verified_findings": mean_jaccard(by_run_verified)}
