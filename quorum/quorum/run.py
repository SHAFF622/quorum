"""
Run the whole AI scientist: explore with several wordings, referee, report.

    python -m quorum.run --budget 5            # 3 wordings, METABRIC + TCGA
    python -m quorum.run --wordings 1 --budget 1   # smallest smoke run

Writes everything to runs/<timestamp>/: the claim board (SQLite), every claim
with its evidence and verdict (JSON), a plain-English report (Markdown), and
a log of every Claude call with its cost.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from quorum.agents import explore
from quorum.board import ClaimBoard
from quorum.data import data_card, load
from quorum.html import render
from quorum.llm import LLM, BudgetExceeded, Refused
from quorum.panel import agreement, apply_consensus, judge
from quorum.verdict import RULE_VERSION

FOCUS = (" Pay particular attention to estrogen receptor expression as a continuous score (ESR1), "
         "not only positive/negative status.")
WORDINGS = [
    "Which clinical and molecular features best explain differences in overall survival in breast "
    "cancer, and in which patient subgroups?" + FOCUS,
    "In breast cancer, what patient and tumor characteristics are most strongly linked to how long "
    "patients live, and does that differ between groups of patients?" + FOCUS,
    "Identify the clinical and molecular factors that drive variation in breast cancer overall "
    "survival, and the subgroups where those factors matter most." + FOCUS,
]

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wordings", type=int, default=3)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--max-analyses", type=int, default=15)
    ap.add_argument("--cohorts", default="metabric,tcga")
    args = ap.parse_args()

    out = RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    cohorts = {c: load(c) for c in args.cohorts.split(",")}
    llm = LLM(budget_usd=args.budget, log_path=out / "llm_calls.jsonl")
    board = ClaimBoard(out / "board.sqlite")

    # 1. Explore, once per wording.
    for i, goal in enumerate(WORDINGS[: args.wordings], 1):
        print(f"[explore] run {i}: {goal[:70]}...")
        try:
            for c in explore(goal, cohorts, llm, run_id=f"run{i}", max_analyses=args.max_analyses):
                board.post_claim(c)
        except BudgetExceeded as e:
            print(f"[budget] stopped exploring: {e}")
            break
        except (Refused, RuntimeError) as e:
            print(f"[explore] run {i} failed, continuing with the others: {e}")
            continue
        print(f"          {llm.summary()}")

    # 2. Referee every claim.
    claims = board.claims()
    verdicts = {}
    for c in claims:
        try:
            verdicts[c.claim_id] = judge(c, cohorts, llm)
        except BudgetExceeded as e:
            print(f"[budget] stopped refereeing: {e}")
            break
        except Exception as e:  # noqa: BLE001 - one bad claim must not lose the run
            print(f"[judge] skipped one claim after {e.__class__.__name__}: {str(e)[:200]}")
            continue
        print(f"[judge] {verdicts[c.claim_id].label.value:12} {c.text[:90]}")
    claims = [c for c in claims if c.claim_id in verdicts]

    # 3. Consensus across runs, then save.
    verdicts = apply_consensus(claims, verdicts)
    for c in claims:
        for e in verdicts[c.claim_id].evidence:
            board.post_evidence(c.claim_id, e)
        board.post_verdict(verdicts[c.claim_id])

    summary = {
        "rule_version": RULE_VERSION,
        "wordings": WORDINGS[: args.wordings],
        "claims": len(claims),
        "labels": _count(v.label.value for v in verdicts.values()),
        "agreement_between_runs": agreement(claims, verdicts),
        "llm": llm.summary(),
        "data_cards": {k: data_card(df).to_dict() for k, df in cohorts.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "claims.json").write_text(json.dumps(board.export(), indent=2, default=str))
    (out / "report.md").write_text(report(claims, verdicts, summary))
    page = render(out)
    print(f"\nDone. {summary['labels']}  spent ${llm.spent_usd:.2f}\nReport: {page}")


def _count(items) -> dict:
    d: dict = defaultdict(int)
    for i in items:
        d[i] += 1
    return dict(d)


def report(claims, verdicts, summary) -> str:
    groups = defaultdict(list)
    for c in claims:
        groups[verdicts[c.claim_id].triage].append(c)
    lines = ["# Quorum report", "",
             f"{summary['claims']} claims from {len(summary['wordings'])} independent runs; "
             f"spent ${summary['llm']['spent_usd']:.2f}. Verdict rule {summary['rule_version']}.", "",
             f"Agreement between runs (Jaccard): all checkable findings "
             f"{_fmt(summary['agreement_between_runs']['checkable_findings'])}, verified findings "
             f"{_fmt(summary['agreement_between_runs']['verified_findings'])}.", ""]
    for bucket in ("act on", "check first", "set aside"):
        lines += [f"## {bucket.capitalize()} ({len(groups[bucket])})", ""]
        for c in groups[bucket]:
            v = verdicts[c.claim_id]
            lines.append(f"- **{v.label.value}** ({c.provenance.run_id}) {c.text}")
            for r in v.reasons:
                lines.append(f"    - {r}")
        lines.append("")
    return "\n".join(lines)


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


if __name__ == "__main__":
    main()
