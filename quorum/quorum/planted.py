"""
Planted errors: false claims made on purpose, so we know the right answer.

Each true fact from the answer key is corrupted in the ways AI scientists
actually go wrong. Because we made the error, we know the correct verdict, so
we can score any referee (or a single model checking itself) without needing
human experts for every claim.

Defects:
- direction_flip     says "better" where the data say "worse"
- inflated_effect    states a hazard ratio far outside what the data show
- causal_overclaim   right numbers, but "causes" instead of "is associated with"
- strong_words       a small effect described as "a major driver"
- multiple_testing   reports a p-value as significant after screening many features
- null_as_signal     a variable with no real effect presented as significant
- underpowered       a subgroup with too few deaths presented as a finding
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from quorum.answer_key import Fact, sentence
from quorum.claims import Claim, ClaimType, Label, Provenance, Stated
from quorum.survival import AnalysisSpec, Filter, analyse

# Verdicts that count as "caught" for each kind of defect.
CAUGHT_IF = {
    "direction_flip": {Label.REFUTED},
    "inflated_effect": {Label.REFUTED},
    "causal_overclaim": {Label.REFUTED},
    "strong_words": {Label.REFUTED},
    "multiple_testing": {Label.REFUTED},
    "null_as_signal": {Label.REFUTED},
    "underpowered": {Label.REFUTED, Label.CONTESTED, Label.UNVERIFIABLE},
}


@dataclass
class Case:
    """One claim in the evaluation set, with its known truth."""

    claim: Claim
    is_true: bool
    defect: Optional[str] = None
    origin: str = ""
    acceptable: set = field(default_factory=set)


def _planted(text: str, spec: AnalysisSpec, stated: Stated, defect: str, origin: str) -> Case:
    claim = Claim(type=ClaimType.DATA, text=text, spec=spec, stated=stated,
                  provenance=Provenance(source="planted"))
    return Case(claim=claim, is_true=False, defect=defect, origin=origin, acceptable=CAUGHT_IF[defect])


def corrupt(fact: Fact, true_claim: Claim) -> list[Case]:
    s = true_claim.stated
    hr, p, n = s.hr, s.p_value, s.n
    cases = []

    flipped = "better" if s.direction == "worse" else "worse"
    cases.append(_planted(
        sentence(fact, flipped, 1 / hr, p, n), fact.spec,
        Stated(direction=flipped, hr=round(1 / hr, 2), p_value=p, n=n), "direction_flip", fact.key))

    big = round(hr * 2.5 if hr > 1 else hr / 2.5, 2)
    cases.append(_planted(
        sentence(fact, s.direction, big, p, n), fact.spec,
        Stated(direction=s.direction, hr=big, p_value=p, n=n), "inflated_effect", fact.key))

    causal = sentence(fact, s.direction, hr, p, n).replace(" have ", " cause ", 1).replace(
        " are associated with ", " cause ", 1)
    cases.append(_planted(causal, fact.spec, s.model_copy(), "causal_overclaim", fact.key))

    if 1 / 1.2 < hr < 1.2:
        cases.append(_planted(
            f"{fact.exposed_label} are a major driver of {s.direction} overall survival (HR {hr:.2f}, p={p:.2g}).",
            fact.spec, s.model_copy(), "strong_words", fact.key))

    if 0.05 / 40 < p < 0.05:
        label = fact.exposed_label
        if not label[1:2].isupper():  # keep acronyms like "ER-negative" intact
            label = label[0].lower() + label[1:]
        cases.append(_planted(
            f"After screening 40 clinical and molecular features, {label} "
            f"stood out with {s.direction} overall survival (p={p:.2g}).",
            fact.spec, s.model_copy(update={"tests_run": 40}), "multiple_testing", fact.key))
    return cases


# Variables with no real survival effect in METABRIC are found by the data, not
# assumed: any candidate with p > 0.2 becomes a "null presented as signal" case.
NULL_CANDIDATES = [
    ("esr1_z_all", AnalysisSpec(variable="esr1_z"), "Higher ESR1 expression"),
    ("histology_lobular", AnalysisSpec(variable="histology", exposed="Lobular", reference="Ductal/NST"),
     "Lobular tumors"),
    ("chemo_in_basal", AnalysisSpec(variable="chemotherapy", exposed="YES", reference="NO",
                                    subgroup=[Filter(column="pam50", op="==", value="Basal")]),
     "Basal-like tumors treated with chemotherapy"),
]

UNDERPOWERED = [
    ("young_her2pos", AnalysisSpec(variable="her2_status", exposed="Positive", reference="Negative",
                                   subgroup=[Filter(column="age", op="<", value=40)]),
     "Among patients under 40, HER2-positive tumors"),
    ("young_basal", AnalysisSpec(variable="pam50", exposed="Basal", reference="LumA",
                                 subgroup=[Filter(column="age", op="<", value=40)]),
     "Among patients under 40, basal-like tumors"),
]


def nulls_and_small(metabric: pd.DataFrame) -> list[Case]:
    cases = []
    for key, spec, label in NULL_CANDIDATES:
        try:
            r = analyse(metabric, spec)
        except ValueError:
            continue
        if r.p_value <= 0.2:
            continue
        cases.append(_planted(
            f"{label} are associated with worse overall survival (HR 1.25, p=0.004, n={r.n}).",
            spec, Stated(direction="worse", hr=1.25, p_value=0.004, n=r.n), "null_as_signal", key))
    for key, spec, label in UNDERPOWERED:
        try:
            r = analyse(metabric, spec)
        except ValueError:
            continue
        smallest = min(r.events_exposed or 0, r.events_reference or 0)
        if smallest >= 20:
            continue
        cases.append(_planted(
            f"{label} have {r.direction} overall survival (HR {r.hr:.2f}, p={r.p_value:.2g}, n={r.n}).",
            spec, Stated(direction=r.direction, hr=round(r.hr, 2), p_value=r.p_value, n=r.n),
            "underpowered", key))
    return cases


def build_eval_set(metabric: pd.DataFrame, answer_key: list[tuple[Fact, Claim]]) -> list[Case]:
    cases = [Case(claim=c, is_true=True, origin=f.key,
                  acceptable={Label.VERIFIED, Label.ROBUST, Label.CONTESTED})
             for f, c in answer_key]
    for fact, claim in answer_key:
        cases.extend(corrupt(fact, claim))
    cases.extend(nulls_and_small(metabric))
    return cases
