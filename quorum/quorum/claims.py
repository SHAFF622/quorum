"""
The claim record: the one thing every part of Quorum passes around.

A claim is a sentence an AI scientist wrote ("Basal tumors have worse overall
survival than Luminal A, HR 1.31"), plus the structured details needed to
re-check it (`spec`, `stated`) and where it came from (`provenance`).

Claims and verdicts are append-only: nothing is edited, a correction is a new
record pointing at its parent. Each record carries a content hash, like a
receipt number, so you can prove later that nothing changed.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from quorum.survival import AnalysisSpec


class ClaimType(str, Enum):
    DATA = "data"  # a number computed from the dataset
    LITERATURE = "literature"  # a statement about what papers report
    INTERPRETATION = "interpretation"  # what the numbers mean


class Stated(BaseModel):
    """The numbers the claim itself states. Compared against the re-analysis."""

    direction: Optional[Literal["worse", "better"]] = None
    hr: Optional[float] = None
    p_value: Optional[float] = None
    n: Optional[int] = None
    events: Optional[int] = None
    tests_run: Optional[int] = Field(
        default=None, description="How many hypotheses were tested to find this one."
    )


class Provenance(BaseModel):
    """Where a claim came from: enough to reproduce or audit it."""

    source: str = "unknown"  # e.g. explorer, kosmos-community, answer-key, planted
    run_id: Optional[str] = None
    goal_wording: Optional[str] = None
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    code: Optional[str] = None
    code_sha256: Optional[str] = None
    data_sha256: Optional[str] = None


class Claim(BaseModel):
    claim_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: ClaimType = ClaimType.DATA
    text: str
    spec: Optional[AnalysisSpec] = None
    stated: Stated = Field(default_factory=Stated)
    citations: list[str] = Field(default_factory=list, description="PubMed IDs")
    provenance: Provenance = Field(default_factory=Provenance)
    parents: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def content_hash(self) -> str:
        """Hash of what the claim says (not when or by whom it was stored)."""
        body = self.model_dump(mode="json", exclude={"claim_id", "created_at", "provenance", "parents"})
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class Label(str, Enum):
    """The five possible verdicts, best to worst."""

    ROBUST = "robust"  # verified, and found in at least 2 independent runs
    VERIFIED = "verified"
    CONTESTED = "contested"
    UNVERIFIABLE = "unverifiable"
    REFUTED = "refuted"


class Evidence(BaseModel):
    """One referee's finding about one claim."""

    referee: str  # reanalyst, rules, skeptic, librarian, replicator, consensus
    outcome: Literal["supports", "contests", "refutes", "no_evidence"]
    detail: str = ""
    data: dict = Field(default_factory=dict)


class Verdict(BaseModel):
    claim_id: str
    label: Label
    reasons: list[str]
    evidence: list[Evidence]
    rule_version: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def triage(self) -> str:
        return {
            Label.ROBUST: "act on",
            Label.VERIFIED: "act on",
            Label.CONTESTED: "check first",
            Label.UNVERIFIABLE: "set aside",
            Label.REFUTED: "set aside",
        }[self.label]
