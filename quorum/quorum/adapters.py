"""
Turn other AI scientists' output into Quorum claims.

- Open-source Kosmos (jimmc414/Kosmos): each finding is saved as
  <artifacts>/cycle_N/task_M_finding.json with a summary, statistics, an
  interpretation and citations. One finding becomes up to two claims: the data
  claim (summary + its statistics) and the interpretation claim, because Kosmos's
  accuracy differs most between those two kinds (85.5% vs 57.9%).
- Plain JSON lines: {"text": ..., "type": "data"|"interpretation", "run_id": ...,
  "citations": [...]} per line, for any other system (ScienceClaw teams, a real
  Kosmos report typed up by hand).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from quorum.claims import Claim, ClaimType, Provenance


def _stats_text(stats: dict) -> str:
    keep = []
    for k, v in (stats or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            keep.append(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}")
    return f" ({', '.join(keep)})" if keep else ""


def _pmids(citations) -> list[str]:
    out = []
    for c in citations or []:
        if isinstance(c, dict):
            for key in ("pmid", "PMID", "pubmed_id"):
                if c.get(key):
                    out.append(str(c[key]))
        elif isinstance(c, str) and c.isdigit():
            out.append(c)
    return out


def from_kosmos(artifacts_dir: str | Path, run_id: str | None = None) -> list[Claim]:
    root = Path(artifacts_dir)
    run_id = run_id or root.name
    claims = []
    for path in sorted(root.glob("cycle_*/task_*_finding.json")):
        f = json.loads(path.read_text())
        prov = Provenance(
            source="kosmos-community", run_id=run_id,
            code=(f.get("code_provenance") or {}).get("code_snippet"),
            prompt_version=f.get("finding_id"),
        )
        citations = _pmids(f.get("citations"))
        if f.get("summary"):
            claims.append(Claim(type=ClaimType.DATA, text=f["summary"].strip() + _stats_text(f.get("statistics")),
                                citations=citations, provenance=prov))
        if f.get("interpretation"):
            claims.append(Claim(type=ClaimType.INTERPRETATION, text=f["interpretation"].strip(),
                                citations=citations, provenance=prov))
    return claims


def from_jsonl(path: str | Path) -> list[Claim]:
    claims = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        claims.append(Claim(
            type=ClaimType(row.get("type", "data")), text=row["text"],
            citations=[str(c) for c in row.get("citations", [])],
            provenance=Provenance(source=row.get("source", "jsonl"), run_id=row.get("run_id", "run1")),
        ))
    return claims


def load_claims(kosmos_dirs: Iterable[str] = (), jsonl: Iterable[str] = ()) -> list[Claim]:
    claims = []
    for d in kosmos_dirs:
        claims += from_kosmos(d)
    for j in jsonl:
        claims += from_jsonl(j)
    return claims
