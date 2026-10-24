"""
The verdict rule: plain code that turns referees' evidence into a label.

Two promises, written into the code below:

1. A model can lower trust but never raise it. Only the Re-analyst's
   reproduction from raw data (plain code) can make a claim "verified".
   AI referees (Skeptic, Librarian) can only push a claim down.
2. The rule is frozen. Its version is the hash of this file, recorded on every
   verdict, so an evaluation can prove the rule didn't change mid-way (the
   frozen-metric idea from Google's epitope-excision paper).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from quorum.claims import Claim, ClaimType, Evidence, Label, Verdict

RULE_VERSION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
ROBUST_MIN_RUNS = 2


def decide(claim: Claim, evidence: list[Evidence], runs_found_in: int = 1) -> Verdict:
    refutes = [e for e in evidence if e.outcome == "refutes"]
    contests = [e for e in evidence if e.outcome == "contests"]
    reproduced = any(e.referee == "reanalyst" and e.outcome == "supports" for e in evidence)
    literature_ok = any(e.referee == "librarian" and e.outcome == "supports" for e in evidence)

    skeptic_ok = any(e.referee == "skeptic" and e.outcome == "supports" for e in evidence)

    if refutes:
        label = Label.REFUTED
        reasons = [f"{e.referee}: {e.detail}" for e in refutes]
    elif claim.type == ClaimType.INTERPRETATION and not reproduced:
        # Meaning can't be re-derived from data. At best a reviewer finds it a
        # fair reading, which still needs a human: never "verified".
        label = Label.CONTESTED
        reasons = ([f"{e.referee}: {e.detail}" for e in contests] if contests else
                   ["A fair reading of the evidence, but interpretation can't be verified against data."
                    if skeptic_ok else "Interpretation with no independent review."])
    elif not reproduced and not literature_ok:
        label = Label.UNVERIFIABLE
        reasons = ["No referee could reproduce or source this claim."]
    elif contests:
        label = Label.CONTESTED
        reasons = [f"{e.referee}: {e.detail}" for e in contests]
    elif reproduced and runs_found_in >= ROBUST_MIN_RUNS:
        label = Label.ROBUST
        reasons = [f"Reproduced from raw data and found in {runs_found_in} independent runs."]
    elif reproduced:
        label = Label.VERIFIED
        reasons = ["Reproduced from raw data."]
    else:
        # Literature support alone, with no data reproduction.
        label = Label.CONTESTED
        reasons = ["Supported by literature only; not reproduced from data."]

    if label in (Label.VERIFIED, Label.ROBUST):
        reasons += [f"replicator: {e.detail}" for e in evidence
                    if e.referee == "replicator" and e.outcome == "supports"]

    return Verdict(
        claim_id=claim.claim_id, label=label, reasons=reasons,
        evidence=evidence, rule_version=RULE_VERSION,
    )
