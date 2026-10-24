"""
Referee claims from any AI scientist: the tool other teams can call.

    # findings saved by the open-source Kosmos (one folder per run)
    python -m quorum.referee --kosmos ../kosmos-community/artifacts/run_a ../kosmos-community/artifacts/run_b

    # any system: one JSON object per line {"text", "type", "run_id", "citations"}
    python -m quorum.referee --jsonl claims.jsonl --budget 2

Writes runs/referee-<timestamp>/ with report.html, claims.json and summary.json,
the same layout as a full Quorum run.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from quorum.adapters import load_claims
from quorum.board import ClaimBoard
from quorum.data import load
from quorum.html import render
from quorum.llm import LLM, BudgetExceeded
from quorum.panel import agreement, apply_consensus, judge
from quorum.verdict import RULE_VERSION

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kosmos", nargs="*", default=[], help="open-source Kosmos artifacts folders")
    ap.add_argument("--jsonl", nargs="*", default=[], help="claims as JSON lines")
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--no-skeptic", action="store_true", help="cheaper: skip the Opus skeptic")
    args = ap.parse_args()

    claims = load_claims(args.kosmos, args.jsonl)
    if not claims:
        raise SystemExit("No claims found. Check the folder or file paths.")
    out = RUNS_DIR / f"referee-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    cohorts = {"metabric": load("metabric"), "tcga": load("tcga")}
    llm = LLM(budget_usd=args.budget, log_path=out / "llm_calls.jsonl")
    board = ClaimBoard(out / "board.sqlite")

    verdicts = {}
    for c in claims:
        board.post_claim(c)
        try:
            verdicts[c.claim_id] = judge(c, cohorts, llm, use_skeptic=not args.no_skeptic)
        except BudgetExceeded as e:
            print(f"[budget] stopped: {e}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"[judge] skipped one claim after {e.__class__.__name__}: {str(e)[:200]}")
            continue
        print(f"[judge] {verdicts[c.claim_id].label.value:12} {c.text[:90]}")
    claims = [c for c in claims if c.claim_id in verdicts]
    verdicts = apply_consensus(claims, verdicts)
    for c in claims:
        for e in verdicts[c.claim_id].evidence:
            board.post_evidence(c.claim_id, e)
        board.post_verdict(verdicts[c.claim_id])

    runs = sorted({c.provenance.run_id for c in claims})
    summary = {"rule_version": RULE_VERSION, "wordings": runs, "claims": len(claims),
               "labels": {}, "agreement_between_runs": agreement(claims, verdicts), "llm": llm.summary()}
    for v in verdicts.values():
        summary["labels"][v.label.value] = summary["labels"].get(v.label.value, 0) + 1
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "claims.json").write_text(json.dumps(board.export(), indent=2, default=str))
    print(f"{summary['labels']}  spent ${llm.spent_usd:.2f}\n{render(out)}")


if __name__ == "__main__":
    main()
