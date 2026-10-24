"""
The Librarian: does each cited paper actually support the sentence citing it?

Kosmos's own error log has an example: it mis-annotated a microRNA binding site
on SOD2 by trusting one automated database prediction. Citations look like
evidence even when they aren't. The Librarian fetches each cited paper's
abstract from Europe PMC (free, no key) and asks Claude, seeing only the
sentence and the abstracts, whether they support it.

- every citation supports the claim      -> supports
- a citation is unrelated or too weak     -> contests
- a citation says the opposite            -> refutes
- the cited ID doesn't exist              -> refutes (a made-up citation)
- no citations                            -> no evidence
"""

from __future__ import annotations

from typing import Literal, Optional

import httpx
from pydantic import BaseModel

from quorum.claims import Claim, Evidence
from quorum.llm import LLM, SONNET, Refused

PROMPT_VERSION = "librarian-v1"
SYSTEM = """You check citations. You get one sentence from an AI scientist's report and the abstracts
of the papers it cites. For each paper decide whether its abstract supports the sentence ("supports"),
is on a related topic but doesn't back the specific statement ("unrelated_or_weak"), or says the
opposite ("contradicts"). Judge only from the abstracts given. Be brief."""


class PaperVerdict(BaseModel):
    pmid: str
    judgement: Literal["supports", "unrelated_or_weak", "contradicts"]
    reason: str


class LibrarianOut(BaseModel):
    papers: list[PaperVerdict]


def fetch_abstract(pmid: str) -> Optional[dict]:
    """Title and abstract for a PubMed ID (shared cache with the literature agent); None if it doesn't exist."""
    from quorum.literature import fetch_papers
    found = fetch_papers([pmid])
    return {"pmid": pmid, "title": found[0]["title"], "abstract": found[0]["abstract"]} if found else None


def check_citations(claim: Claim, llm: LLM, fetch=fetch_abstract) -> Evidence:
    if not claim.citations:
        return Evidence(referee="librarian", outcome="no_evidence", detail="No citations to check.")
    papers, missing = [], []
    for pmid in claim.citations:
        try:
            p = fetch(pmid)
        except httpx.HTTPError as e:
            return Evidence(referee="librarian", outcome="no_evidence", detail=f"Europe PMC unreachable: {e}")
        (papers if p else missing).append(p or pmid)
    if missing:
        return Evidence(referee="librarian", outcome="refutes",
                        detail=f"Cited PubMed ID(s) not found: {', '.join(missing)}.", data={"missing": missing})
    user = f"Sentence: {claim.text}\n\n" + "\n\n".join(
        f"PMID {p['pmid']}: {p['title']}\n{p['abstract'][:3000]}" for p in papers)
    try:
        out = llm.json(LibrarianOut, model=SONNET, system=SYSTEM, user=user,
                       prompt_version=PROMPT_VERSION, effort="low")
    except Refused as e:
        return Evidence(referee="librarian", outcome="no_evidence", detail=str(e))
    judgements = {v.judgement for v in out.papers}
    detail = " ".join(f"PMID {v.pmid}: {v.judgement.replace('_', ' ')} ({v.reason})" for v in out.papers)
    data = {"papers": [v.model_dump() for v in out.papers]}
    if "contradicts" in judgements:
        return Evidence(referee="librarian", outcome="refutes", detail=detail, data=data)
    if "unrelated_or_weak" in judgements:
        return Evidence(referee="librarian", outcome="contests", detail=detail, data=data)
    return Evidence(referee="librarian", outcome="supports", detail=detail, data=data)
