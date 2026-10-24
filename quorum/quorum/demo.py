"""
Offline demo: the full board with no Claude calls and no cost.

It referees the evaluation claims (8 settled facts plus 32 planted errors)
with the code-only referees (Re-analyst given the exact analysis, stats rules,
TCGA replicator) and renders the same HTML board a real run produces. Use it
to show the pipeline when there is no network or no API key, and say plainly
that these are test claims, not an AI scientist's output.

    python -m quorum.demo
"""

from __future__ import annotations

import json
from pathlib import Path

from quorum import answer_key, planted
from quorum.board import ClaimBoard
from quorum.data import data_card, load
from quorum.evaluate import code_panel
from quorum.html import render
from quorum.verdict import RULE_VERSION

OUT = Path(__file__).resolve().parent.parent / "runs" / "demo-offline"


def main() -> Path:
    cohorts = {"metabric": load("metabric"), "tcga": load("tcga")}
    cases = planted.build_eval_set(cohorts["metabric"], answer_key.build(cohorts["metabric"]))
    judge = code_panel(cohorts)

    OUT.mkdir(parents=True, exist_ok=True)
    db = OUT / "board.sqlite"
    if db.exists():
        db.unlink()
    board = ClaimBoard(db)
    labels: dict[str, int] = {}
    for case in cases:
        claim = case.claim.model_copy(update={"provenance": case.claim.provenance.model_copy(
            update={"run_id": "planted error" if not case.is_true else "settled fact"})})
        board.post_claim(claim)
        v = judge(claim)
        for e in v.evidence:
            board.post_evidence(claim.claim_id, e)
        board.post_verdict(v)
        labels[v.label.value] = labels.get(v.label.value, 0) + 1

    summary = {
        "rule_version": RULE_VERSION, "wordings": [], "claims": len(cases), "labels": labels,
        "agreement_between_runs": {}, "llm": {"spent_usd": 0.0},
        "data_cards": {k: data_card(df).to_dict() for k, df in cohorts.items()},
        "note": "Offline demo on labeled test claims; not an AI scientist run.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT / "claims.json").write_text(json.dumps(board.export(), indent=2, default=str))
    page = render(OUT)
    print(f"{labels}\n{page}")
    return page


if __name__ == "__main__":
    main()
