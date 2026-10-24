"""
Statistics rules: plain-code checks for the mistakes Kosmos makes most.

Kosmos is right only 57.9% of the time on interpretation statements, mostly by
treating "statistically significant" as "scientifically important". These
rules catch the common versions of that, without asking a model:

- too few deaths to trust a survival difference
- a tiny effect described with strong words
- cause-and-effect words on observational data
- significance that disappears once you count how many things were tested
- a success measure that quietly changed (from Google's frozen-metric rule)
- HER2 claims on METABRIC, whose patients mostly predate trastuzumab

Each rule returns a finding with a severity: "refutes" (the claim is wrong as
written), "contests" (it may be right but needs a human look), or nothing.
"""

from __future__ import annotations

import math
import re
from typing import Optional

from quorum.claims import Claim, Evidence
from quorum.survival import SurvivalResult

MIN_EVENTS_PER_GROUP = 20
MIN_EVENTS_TOTAL = 50
SMALL_HR = 1.2  # an HR between 1/1.2 and 1.2 is a small effect

STRONG_WORDS = re.compile(
    r"\b(strong(ly)?|major|dramatic(ally)?|key driver|dominant|critical|profound(ly)?|"
    r"striking(ly)?|substantial(ly)?|powerful)\b", re.I)
CAUSAL_WORDS = re.compile(
    r"\b(causes?|caused|causing|drives?|driving|leads? to|results? in|"
    r"responsible for|determines?|because of|due to)\b", re.I)


def _effect(claim: Claim, ref: Optional[SurvivalResult]) -> Optional[float]:
    if ref is not None:
        return ref.hr
    return claim.stated.hr


def check(claim: Claim, ref: Optional[SurvivalResult] = None,
          declared_endpoint: Optional[str] = None) -> list[Evidence]:
    out: list[Evidence] = []
    text = claim.text
    asserts_effect = bool(claim.stated.direction) or bool(re.search(r"\b(worse|better|associated|predicts?|higher risk|lower risk)\b", text, re.I))

    # 1. Too few deaths.
    if ref is not None:
        smallest = min(x for x in (ref.events_exposed, ref.events_reference, ref.events) if x is not None)
        if ref.events < MIN_EVENTS_TOTAL or smallest < MIN_EVENTS_PER_GROUP:
            out.append(Evidence(
                referee="rules", outcome="contests",
                detail=f"Underpowered: only {smallest} deaths in the smallest group ({ref.events} in total).",
                data={"rule": "too_few_events"},
            ))

    # 2. Small effect, strong words.
    hr = _effect(claim, ref)
    if hr and STRONG_WORDS.search(text) and 1 / SMALL_HR < hr < SMALL_HR:
        out.append(Evidence(
            referee="rules", outcome="refutes",
            detail=f"Strong wording for a small effect (HR {hr:.2f}).",
            data={"rule": "overclaim_small_effect"},
        ))

    # 3. Causal words on observational data.
    m = CAUSAL_WORDS.search(text)
    if m and asserts_effect:
        out.append(Evidence(
            referee="rules", outcome="refutes",
            detail=f"Causal language (\"{m.group(0)}\") on observational survival data.",
            data={"rule": "causal_language"},
        ))

    # 4. Multiple testing (Bonferroni).
    k = claim.stated.tests_run
    p = ref.p_value if ref is not None else claim.stated.p_value
    if k and k > 1 and p is not None and p < 0.05 and p * k >= 0.05:
        out.append(Evidence(
            referee="rules", outcome="refutes",
            detail=f"Not significant after testing {k} hypotheses (p={p:.2g}, corrected {min(1, p * k):.2g}).",
            data={"rule": "multiple_testing"},
        ))

    # 5. Changed success measure. Not wrong in itself, but it answers a
    # different question from the one the run was set, so a human should look.
    if declared_endpoint and claim.spec is not None and claim.spec.endpoint != declared_endpoint:
        out.append(Evidence(
            referee="rules", outcome="contests",
            detail=f"Uses {claim.spec.endpoint} (breast cancer deaths only); this run asked about overall survival.",
            data={"rule": "changed_endpoint"},
        ))
    if claim.spec is not None and claim.spec.endpoint == "OS" and re.search(r"\b(cancer-specific|disease-specific|died of (breast )?cancer)\b", text, re.I):
        out.append(Evidence(
            referee="rules", outcome="refutes",
            detail="Text claims a cancer-specific result, but the analysis counts deaths from any cause.",
            data={"rule": "endpoint_mismatch"},
        ))
    if re.search(r"\b(recurrence|relapse|disease-free|progression)\b", text, re.I) and claim.spec is not None:
        out.append(Evidence(
            referee="rules", outcome="contests",
            detail="Text talks about recurrence, but the analysis is overall survival.",
            data={"rule": "endpoint_wording"},
        ))

    # 6. Treatment era.
    if claim.spec is not None and claim.spec.cohort == "metabric" and (
        claim.spec.variable == "her2_status" or claim.spec.exposed == "Her2"
    ):
        out.append(Evidence(
            referee="rules", outcome="contests",
            detail="METABRIC patients were mostly treated before trastuzumab; HER2 survival today differs.",
            data={"rule": "treatment_era"},
        ))
    return out
