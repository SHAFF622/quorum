"""
The claim board: a shared, append-only notebook of claims, evidence and verdicts.

Explorers post claims; referees read them and post evidence; plain code posts
verdicts. Nothing is updated in place, so the full history of how a claim was
judged is always there (the ScienceClaw "artifact graph" idea, in SQLite).

A human can also post a "request" (e.g. "check this in ER-positive patients
only") that referees pick up, which is how you steer a run mid-way.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from quorum.claims import Claim, Evidence, Verdict

SCHEMA = """
create table if not exists claims (
    claim_id text primary key,
    content_sha256 text not null,
    body text not null,
    created_at text not null
);
create table if not exists evidence (
    id integer primary key autoincrement,
    claim_id text not null references claims(claim_id),
    body text not null,
    created_at text not null
);
create table if not exists verdicts (
    id integer primary key autoincrement,
    claim_id text not null references claims(claim_id),
    body text not null,
    created_at text not null
);
create table if not exists requests (
    id integer primary key autoincrement,
    claim_id text,
    text text not null,
    author text not null,
    status text not null default 'open',
    created_at text not null
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ClaimBoard:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)

    # -- claims -------------------------------------------------------------

    def post_claim(self, claim: Claim) -> Claim:
        self.db.execute(
            "insert into claims values (?, ?, ?, ?)",
            (claim.claim_id, claim.content_hash(), claim.model_dump_json(), claim.created_at),
        )
        self.db.commit()
        return claim

    def claims(self) -> list[Claim]:
        rows = self.db.execute("select body from claims order by created_at").fetchall()
        return [Claim.model_validate_json(r[0]) for r in rows]

    def claim(self, claim_id: str) -> Claim:
        row = self.db.execute("select body from claims where claim_id = ?", (claim_id,)).fetchone()
        if row is None:
            raise KeyError(claim_id)
        return Claim.model_validate_json(row[0])

    # -- evidence and verdicts ---------------------------------------------

    def post_evidence(self, claim_id: str, evidence: Evidence) -> None:
        self.db.execute(
            "insert into evidence (claim_id, body, created_at) values (?, ?, ?)",
            (claim_id, evidence.model_dump_json(), _now()),
        )
        self.db.commit()

    def evidence(self, claim_id: str) -> list[Evidence]:
        rows = self.db.execute(
            "select body from evidence where claim_id = ? order by id", (claim_id,)
        ).fetchall()
        return [Evidence.model_validate_json(r[0]) for r in rows]

    def post_verdict(self, verdict: Verdict) -> None:
        self.db.execute(
            "insert into verdicts (claim_id, body, created_at) values (?, ?, ?)",
            (verdict.claim_id, verdict.model_dump_json(), verdict.created_at),
        )
        self.db.commit()

    def latest_verdict(self, claim_id: str) -> Optional[Verdict]:
        row = self.db.execute(
            "select body from verdicts where claim_id = ? order by id desc limit 1", (claim_id,)
        ).fetchone()
        return Verdict.model_validate_json(row[0]) if row else None

    def unjudged(self) -> Iterator[Claim]:
        """Claims with no verdict yet: what referees should pick up next."""
        rows = self.db.execute(
            "select c.body from claims c left join verdicts v on v.claim_id = c.claim_id "
            "where v.id is null order by c.created_at"
        ).fetchall()
        for r in rows:
            yield Claim.model_validate_json(r[0])

    # -- human steering -----------------------------------------------------

    def post_request(self, text: str, claim_id: Optional[str] = None, author: str = "human") -> int:
        cur = self.db.execute(
            "insert into requests (claim_id, text, author, created_at) values (?, ?, ?, ?)",
            (claim_id, text, author, _now()),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def open_requests(self) -> list[dict]:
        rows = self.db.execute(
            "select id, claim_id, text, author from requests where status = 'open' order by id"
        ).fetchall()
        return [dict(zip(("id", "claim_id", "text", "author"), r)) for r in rows]

    def close_request(self, request_id: int) -> None:
        self.db.execute("update requests set status = 'done' where id = ?", (request_id,))
        self.db.commit()

    def export(self) -> list[dict]:
        """Everything about every claim, for the report."""
        out = []
        for c in self.claims():
            v = self.latest_verdict(c.claim_id)
            out.append({
                "claim": json.loads(c.model_dump_json()),
                "evidence": [json.loads(e.model_dump_json()) for e in self.evidence(c.claim_id)],
                "verdict": json.loads(v.model_dump_json()) if v else None,
            })
        return out
