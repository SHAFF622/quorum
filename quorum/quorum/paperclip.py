"""
Paperclip (paperclip.gxl.ai): a search engine built for AI agents over full-text
biomedical papers, trials and patents. Quorum uses two parts of its HTTP API:

- /search: better paper search for the literature panel (full-text index,
  keyword and meaning-based search fused), used instead of Europe PMC when a
  key is set.
- /claims/support: give it a sentence, get back passages from published papers
  that support or refute it, each with the paper's study design and whether
  it was retracted. This powers the "Published-evidence checker" referee.

Needs a personal key from https://paperclip.gxl.ai/keys, put in quorum/.env as
PAPERCLIP_API_KEY=... (the .env file is gitignored). Without a key, everything
here quietly reports "not connected" and Quorum works as before.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx

from quorum.claims import Claim, Evidence
from quorum.llm import load_env

API = "https://paperclip.gxl.ai/api/v1"
KEYS_URL = "https://paperclip.gxl.ai/keys"

# The referee only pushes a claim down when the published record clearly leans
# against it: several refuting passages, carrying at least twice the weight of
# the supporting ones. One stray contrary paper is not enough.
MIN_REFUTING = 3
WEIGHT_RATIO = 2.0


class PaperclipError(RuntimeError):
    pass


def api_key() -> Optional[str]:
    load_env()
    return os.environ.get("PAPERCLIP_API_KEY") or None


def connected() -> bool:
    return api_key() is not None


def _post(path: str, body: dict, timeout: float = 90) -> dict:
    key = api_key()
    if not key:
        raise PaperclipError(f"No Paperclip key. Create one at {KEYS_URL} and add PAPERCLIP_API_KEY to quorum/.env.")
    r = httpx.post(f"{API}{path}", json=body, headers={"X-API-Key": key}, timeout=timeout)
    if r.status_code >= 400:
        try:
            msg = r.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            msg = r.text[:200]
        raise PaperclipError(f"Paperclip said {r.status_code}: {msg}")
    return r.json()


def search(question: str, n: int = 8) -> list[dict]:
    """Papers for the literature panel, in the same shape as literature.py's."""
    out = _post("/search", {"query": f"{question} breast cancer", "num_results": n,
                            "contents": {"abstract": True, "highlights": True}})
    papers = []
    for h in out.get("papers") or out.get("results") or []:
        if h.get("object", "paper") != "paper":
            continue
        ids = h.get("identifiers") or {}
        pmid, pmcid = h.get("pmid") or ids.get("pmid"), ids.get("pmc")
        authors = h.get("authors") or []
        papers.append({
            "pmid": pmid or h["id"],
            "pmcid": pmcid,
            "title": h.get("title", "").rstrip("."),
            "authors": (authors[0] + " et al.") if len(authors) > 1 else (authors[0] if authors else ""),
            "journal": h.get("journal") or h.get("source") or "",
            "year": h.get("year"),
            "doi": h.get("doi"),
            "abstract": h.get("abstract") or h.get("snippet") or "",
            "open_access": bool(pmcid),
            "links": {"pubmed": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                      "pmc": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else None,
                      "pdf": None, "paperclip": h.get("url")},
            "via": "paperclip",
        })
    return papers


def _item(e: dict) -> dict:
    p = e.get("paper") or {}
    return {"evidence": (e.get("evidence") or "")[:400], "confidence": round(e.get("confidence") or 0, 2),
            "weight": e.get("weight") or 0, "title": p.get("title"), "pmid": p.get("pmid"),
            "year": p.get("year"), "design": p.get("design"), "retracted": bool(p.get("retracted"))}


def claim_evidence(claim: Claim) -> Evidence:
    """The Published-evidence checker: what does the literature say about this sentence?"""
    if not connected():
        return Evidence(referee="evidence", outcome="no_evidence", detail="Paperclip not connected.")
    try:
        out = _post("/claims/support", {"claim": claim.text, "limit": 5})
    except (PaperclipError, httpx.HTTPError) as e:
        return Evidence(referee="evidence", outcome="no_evidence", detail=f"Paperclip unavailable: {e}")
    # Retracted papers carry zero weight already; leave them out of the counts too.
    sup = [_item(e) for e in out.get("supports", []) if not (e.get("paper") or {}).get("retracted")]
    ref = [_item(e) for e in out.get("refutes", []) if not (e.get("paper") or {}).get("retracted")]
    counts = out.get("counts", {})
    n_sup = counts.get("supports", len(sup)) - counts.get("supports_retracted", 0)
    n_ref = counts.get("refutes", len(ref)) - counts.get("refutes_retracted", 0)
    w_sup, w_ref = sum(x["weight"] for x in sup), sum(x["weight"] for x in ref)
    data = {"supports": sup, "refutes": ref, "counts": {"supports": n_sup, "refutes": n_ref,
                                                         "retrieved": counts.get("retrieved")}}
    summary = f"{n_sup} published passages support it, {n_ref} refute it (of {counts.get('retrieved', '?')} read)."
    if n_ref >= MIN_REFUTING and w_ref > WEIGHT_RATIO * max(w_sup, 1e-9):
        top = ref[0]
        return Evidence(referee="evidence", outcome="contests",
                        detail=f"{summary} The published record leans against it, e.g. "
                               f"PMID {top['pmid'] or '?'} ({top['year'] or '?'}): \"{top['evidence'][:200]}\"",
                        data=data)
    if n_sup:
        # Agreement with papers is reassuring but never raises the label (verdict.py only trusts the data).
        return Evidence(referee="evidence", outcome="supports", detail=summary, data=data)
    return Evidence(referee="evidence", outcome="no_evidence", detail=summary, data=data)
