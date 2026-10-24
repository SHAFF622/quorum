"""
Score a referee setup on the planted-error set.

Two numbers matter (the same ones you'd ask of a medical test):
- catch rate: of the false claims, how many did it reject?
- false-reject rate: of the true claims, how many did it wrongly reject?

Every setup is scored on the same frozen set, with the verdict rule version
recorded, so comparisons are fair (the frozen-metric rule).

Run: python -m quorum.evaluate
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Callable

import pandas as pd

from quorum import answer_key, planted, reanalyst, rules
from quorum.claims import Claim, Label, Verdict
from quorum.data import load
from quorum.planted import Case
from quorum.replicator import replicate
from quorum.verdict import RULE_VERSION, decide

Judge = Callable[[Claim], Verdict]


def code_panel(cohorts: dict[str, pd.DataFrame]) -> Judge:
    """Referees that need no AI: Re-analyst (structured spec) + stats rules."""
    def judge(claim: Claim) -> Verdict:
        ev, ref = reanalyst.reanalyse(claim, cohorts)
        evidence = [ev, *rules.check(claim, ref)]
        if ref is not None and ev.outcome == "supports":
            evidence.append(replicate(claim.spec, ref, cohorts))
        return decide(claim, evidence)
    return judge


def accept_all(claim: Claim) -> Verdict:
    """Baseline A: no checking at all."""
    return Verdict(claim_id=claim.claim_id, label=Label.VERIFIED, reasons=["no checking"],
                   evidence=[], rule_version="none")


def score(cases: list[Case], judge: Judge) -> dict:
    """Score every case; a budget stop ends early, an error skips one case. Both are reported."""
    from quorum.llm import BudgetExceeded

    caught, false_rejects = 0, 0
    by_defect: dict[str, list[bool]] = defaultdict(list)
    rows, scored, errors, stopped = [], [], [], None
    for case in cases:
        try:
            v = judge(case.claim)
        except BudgetExceeded as e:
            stopped = str(e)
            break
        except Exception as e:  # noqa: BLE001 - one bad case must not lose the rest
            errors.append(f"{case.origin}/{case.defect or 'true'}: {e.__class__.__name__}: {str(e)[:200]}")
            continue
        scored.append(case)
        ok = v.label in case.acceptable
        if case.is_true:
            false_rejects += v.label == Label.REFUTED
        else:
            caught += ok
            by_defect[case.defect].append(ok)
        rows.append({"origin": case.origin, "defect": case.defect or "true", "label": v.label.value,
                     "ok": ok, "text": case.claim.text, "reasons": v.reasons})
    n_false = sum(not c.is_true for c in scored)
    n_true = sum(c.is_true for c in scored)
    return {
        "rule_version": RULE_VERSION,
        "scored": len(scored),
        "of": len(cases),
        "stopped_by_budget": stopped,
        "errors": errors,
        "n_true": n_true,
        "n_false": n_false,
        "catch_rate": caught / n_false if n_false else None,
        "false_reject_rate": false_rejects / n_true if n_true else None,
        "by_defect": {d: f"{sum(v)}/{len(v)}" for d, v in sorted(by_defect.items())},
        "labels": dict(Counter(r["label"] for r in rows)),
        "rows": rows,
    }


def main() -> None:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also score the Claude panel and the single-agent baseline")
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N cases (cheap trial)")
    args = ap.parse_args()

    metabric = load("metabric")
    key = answer_key.build(metabric)
    cases = planted.build_eval_set(metabric, key)
    # A fixed shuffle, so a --limit trial mixes true facts and planted errors
    # the same way every time.
    import random
    random.Random(0).shuffle(cases)
    if args.limit:
        cases = cases[: args.limit]
    cohorts = {"metabric": metabric, "tcga": load("tcga")}

    setups: list[tuple[str, Judge]] = [("no checking", accept_all),
                                       ("code-only panel (given the exact analysis)", code_panel(cohorts))]
    llm = None
    if args.llm:
        from quorum.agents import single_agent
        from quorum.llm import LLM
        from quorum.panel import judge as panel_judge

        out = Path(__file__).resolve().parent.parent / "runs" / "eval"
        # Same budget for each setup, so neither can starve the other.
        llm_single = LLM(budget_usd=args.budget / 2, log_path=out / "llm_calls_single_agent.jsonl")
        llm_panel = LLM(budget_usd=args.budget / 2, log_path=out / "llm_calls_panel.jsonl")
        llm = {"single agent (claim text only)": llm_single, "Quorum panel (claim text only)": llm_panel}
        setups += [("single agent (claim text only)", lambda c: single_agent(c, cohorts, llm_single)),
                   ("Quorum panel (claim text only)", lambda c: panel_judge(c, cohorts, llm_panel))]

    results = {}
    for name, judge in setups:
        result = score(cases, judge)
        results[name] = result
        print(f"\n== {name}")
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
        for r in result["rows"]:
            if not r["ok"]:
                print(f"  MISSED [{r['defect']}] {r['label']}: {r['text'][:110]}")
        if llm and name in llm:
            result["llm"] = llm[name].summary()
            print(f"  spent: {result['llm']}")
    if llm:
        (out / "results.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
