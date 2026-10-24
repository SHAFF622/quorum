"""
The answer key: true claims, computed from METABRIC by plain code.

These are the "definitive answers" of breast cancer survival (Luminal A does
best; basal, HER2-enriched and Luminal B do worse; higher grade, more positive
nodes and larger tumors do worse). Each fact keeps the direction the medical
literature expects. A fact enters the key only if the data agree with that
direction and the result is significant, so the key is true twice over: by the
literature and by this dataset.

The key does two jobs: the AI scientist should rediscover these facts, and
the planted-error generator corrupts them to make false claims with known labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from quorum.claims import Claim, ClaimType, Provenance, Stated
from quorum.survival import AnalysisSpec, analyse


@dataclass
class Fact:
    key: str
    spec: AnalysisSpec
    exposed_label: str
    reference_label: Optional[str]
    expected_direction: str  # what the literature says: "worse" or "better"


FACTS = [
    Fact("basal_vs_luma", AnalysisSpec(variable="pam50", exposed="Basal", reference="LumA"),
         "Basal-like tumors", "Luminal A tumors", "worse"),
    Fact("lumb_vs_luma", AnalysisSpec(variable="pam50", exposed="LumB", reference="LumA"),
         "Luminal B tumors", "Luminal A tumors", "worse"),
    Fact("her2_vs_luma", AnalysisSpec(variable="pam50", exposed="Her2", reference="LumA"),
         "HER2-enriched tumors", "Luminal A tumors", "worse"),
    Fact("er_neg_vs_pos", AnalysisSpec(variable="er_status", exposed="Negative", reference="Positive"),
         "ER-negative tumors", "ER-positive tumors", "worse"),
    Fact("grade3_vs_1", AnalysisSpec(variable="grade", exposed="3.0", reference="1.0"),
         "Grade 3 tumors", "grade 1 tumors", "worse"),
    Fact("tp53_mut", AnalysisSpec(variable="tp53_mutated", exposed="1.0", reference="0.0"),
         "TP53-mutated tumors", "TP53 wild-type tumors", "worse"),
    Fact("nodes", AnalysisSpec(variable="nodes_positive"),
         "More positive lymph nodes", None, "worse"),
    Fact("tumor_size", AnalysisSpec(variable="tumor_size"),
         "Larger tumors", None, "worse"),
]


def sentence(fact: Fact, direction: str, hr: float, p: float, n: int) -> str:
    """Write a claim the way an AI scientist would."""
    if fact.reference_label:
        return (f"{fact.exposed_label} have {direction} overall survival than "
                f"{fact.reference_label} (HR {hr:.2f}, p={p:.2g}, n={n}).")
    risk = "higher" if direction == "worse" else "lower"
    return (f"{fact.exposed_label} are associated with {direction} overall survival "
            f"({risk} risk per standard deviation, HR {hr:.2f}, p={p:.2g}, n={n}).")


def build(metabric: pd.DataFrame) -> list[tuple[Fact, Claim]]:
    """Compute every fact; keep those the data confirm."""
    out = []
    for fact in FACTS:
        r = analyse(metabric, fact.spec)
        if r.direction != fact.expected_direction or r.p_value >= 0.05:
            continue  # the data don't confirm the textbook here; leave it out
        claim = Claim(
            type=ClaimType.DATA,
            text=sentence(fact, r.direction, r.hr, r.p_value, r.n),
            spec=fact.spec,
            stated=Stated(direction=r.direction, hr=round(r.hr, 2), p_value=r.p_value, n=r.n, events=r.events),
            provenance=Provenance(source="answer-key"),
        )
        out.append((fact, claim))
    return out
