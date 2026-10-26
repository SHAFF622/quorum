"""
The whole AI scientist as one function that reports every step as it happens.

The website calls `run_pipeline(..., emit=...)`; each step sends small JSON
events (data checked, analysis run, claim written, referee result, verdict),
so a viewer can watch the work instead of waiting for a final report.

Two modes:
- "ai": Claude explores (three wordings of the question) and the AI referees
  (blind re-checker, skeptic, citation checker) join the code referees.
- "simple": no API key needed. A simple automatic analyst explores, and only
  the code referees judge. Every event says which mode produced it.
"""

from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from quorum.board import ClaimBoard
from quorum.codegen import code_bundle
from quorum.data import data_card, load
from quorum.html import render
from quorum.llm import HAIKU, LLM, BudgetExceeded, Refused
from quorum.panel import agreement, apply_consensus, judge
from quorum.simple_explorer import describe, explore_simple
from quorum.verdict import RULE_VERSION

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
Emit = Callable[[dict], None]

REFEREE_NAMES = {
    "reanalyst": "Re-checker",
    "rules": "Rule-keeper",
    "adjuster": "Fair-comparison check",
    "replicator": "Second opinion (other patients)",
    "librarian": "Citation checker",
    "evidence": "Published-evidence checker",
    "skeptic": "Skeptic",
    "consensus": "Agreement across runs",
}
REFEREE_JOBS = {
    "reanalyst": "Recomputes the claim from the raw data.",
    "rules": "Looks for classic statistics mistakes.",
    "adjuster": "Re-runs it on comparable patients, and counting only cancer deaths.",
    "replicator": "Re-runs it on a separate group of patients (TCGA).",
    "librarian": "Checks that cited papers support the sentence.",
    "evidence": "Searches published papers (via Paperclip) for passages that support or refute the finding.",
    "skeptic": "Asks whether the conclusion goes beyond the numbers.",
    "consensus": "Checks whether independent runs found the same thing.",
}


class Wordings(BaseModel):
    wordings: list[str]


def parse_upload(text: str) -> pd.DataFrame:
    """A CSV with one row per patient; needs os_months and os_event (1 died, 0 alive)."""
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip() for c in df.columns]
    missing = {"os_months", "os_event"} - set(df.columns)
    if missing:
        raise ValueError(f"The file needs columns {sorted(missing)}: survival time in months, and 1 if the patient died.")
    for c in ("os_months", "os_event", "dss_months", "dss_event"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "dss_event" in df.columns and "dss_months" not in df.columns:
        df["dss_months"] = df["os_months"]
    df.insert(0, "cohort", "uploaded")
    if len(df) > 50_000:
        raise ValueError("Files up to 50,000 patients, please.")
    return df


def _card(df: pd.DataFrame) -> dict:
    card = data_card(df).to_dict()
    cols = []
    for c in df.columns:
        if c == "cohort":
            continue
        s = df[c].dropna()
        kind = "number" if pd.api.types.is_numeric_dtype(s) and s.nunique() > 3 else "category"
        sample = (f"{s.min():.3g} to {s.max():.3g}" if kind == "number" and len(s) else
                  ", ".join(map(str, s.astype(str).value_counts().index[:4])))
        cols.append({"name": c, "kind": kind, "filled": round(float(df[c].notna().mean()), 3), "values": sample})
    card["columns"] = cols
    return card


def paraphrase(question: str, llm: LLM) -> list[str]:
    try:
        out = llm.json(Wordings, model=HAIKU, prompt_version="paraphrase-v1", effort="low",
                       system="Rewrite the research question two different ways with the same meaning. "
                              "Return exactly two rewordings.",
                       user=question)
        return [question] + [w for w in out.wordings if w.strip()][:2]
    except Exception:  # noqa: BLE001 - rewording is optional; one wording still works
        return [question]


def run_pipeline(question: str, dataset: str, mode: str, emit: Emit, upload: Optional[str] = None,
                 budget: float = 2.0, wordings: int = 3) -> dict:
    out_dir = RUNS_DIR / f"web-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data
    emit({"stage": "data", "status": "start", "dataset": dataset})
    if dataset == "upload":
        primary_name, primary = "uploaded", parse_upload(upload or "")
        cohorts = {"uploaded": primary}
    elif dataset == "tcga":
        primary_name, primary = "tcga", load("tcga")
        cohorts = {"tcga": primary}
    else:
        primary_name, primary = "metabric", load("metabric")
        cohorts = {"metabric": primary, "tcga": load("tcga")}
    emit({"stage": "data", "status": "done", "card": _card(primary),
          "second_cohort": "tcga" if "tcga" in cohorts and primary_name != "tcga" else None})

    # 1b. Literature: the papers behind the question, ranked by how closely they match it
    emit({"stage": "literature", "status": "start"})
    literature = ""
    try:
        from quorum.literature import context_for, library
        lib = library(question)
        literature = context_for(lib)
        emit({"stage": "literature", "status": "done", "papers": len(lib["papers"]),
              "full_text": sum(1 for p in lib["papers"] if p.get("full_text")),
              "passages": len(lib["passages"]),
              "top": [{"pmid": p["pmid"], "title": p["title"]} for p in lib["papers"][:3]]})
    except Exception as e:  # noqa: BLE001 - no internet must not stop the data analysis
        emit({"stage": "notice", "message": f"Literature search unavailable ({e.__class__.__name__}); continuing with the data only."})

    # 2. Question
    if mode == "ai":
        llm = LLM(budget_usd=budget, log_path=out_dir / "llm_calls.jsonl")
    elif mode == "kiro":
        from quorum.kiro_llm import KiroLLM
        llm = KiroLLM(max_calls=130, log_path=out_dir / "llm_calls.jsonl")
    else:
        llm = None
    goals = paraphrase(question, llm)[:wordings] if llm else [question]
    emit({"stage": "question", "mode": mode, "wordings": goals})
    who = {"ai": "Claude (AI scientist)", "kiro": "Kiro (AI scientist)"}.get(mode, "Simple analyst (no AI)")

    # 3. Explore (several wordings at once when an AI is exploring)
    board = ClaimBoard(out_dir / "board.sqlite")

    def explore_one(i: int, goal: str):
        run_id = f"run{i}"
        emit({"stage": "explore", "status": "start", "run": run_id, "wording": goal, "who": who})
        count = {"n": 0}

        def on_analysis(spec, result):
            count["n"] += 1
            emit({"stage": "analysis", "run": run_id, "n": count["n"],
                  "what": describe(spec) if spec is not None else "analysis", "result": result,
                  "spec": spec.model_dump(mode="json") if spec is not None else None})

        try:
            if mode == "ai":
                from quorum.agents import explore
                claims = explore(goal, {primary_name: primary}, llm, run_id=run_id,
                                 max_analyses=12, on_analysis=on_analysis, literature=literature)
            elif mode == "kiro":
                from quorum.agents import explore_planned
                claims = explore_planned(goal, {primary_name: primary}, llm, run_id=run_id,
                                         max_analyses=10, max_claims=6, on_analysis=on_analysis,
                                         literature=literature)
            else:
                claims = explore_simple(goal, primary, primary_name, run_id=run_id, on_analysis=on_analysis)
        except BudgetExceeded as e:
            emit({"stage": "notice", "message": f"Budget reached while exploring: {e}"})
            return run_id, []
        except Exception as e:  # noqa: BLE001 - one run failing must not stop the others
            emit({"stage": "notice", "message": f"Run {i} could not finish: {str(e)[:200]}"})
            return run_id, []
        emit({"stage": "claims", "run": run_id,
              "claims": [{"id": c.claim_id, "text": c.text, "type": c.type.value, "citations": c.citations}
                         for c in claims]})
        return run_id, claims

    workers = len(goals) if llm else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        explored = list(pool.map(lambda ig: explore_one(*ig), enumerate(goals, 1)))
    all_claims = []
    for _, claims in explored:
        for c in claims:
            board.post_claim(c)
        all_claims += claims

    # 4. Referees (several claims at once when AI referees are slow)
    # Only a question that names "overall survival" fixes the endpoint; otherwise
    # counting breast cancer deaths is a fair way to answer it.
    declared = "OS" if "overall survival" in question.lower() else None
    contexts: dict[str, list[str]] = {}

    def judge_one(c):
        emit({"stage": "judging", "claim_id": c.claim_id})

        def on_evidence(e):
            emit({"stage": "referee", "claim_id": c.claim_id, "referee": e.referee,
                  "name": REFEREE_NAMES.get(e.referee, e.referee), "job": REFEREE_JOBS.get(e.referee, ""),
                  "outcome": e.outcome, "detail": e.detail,
                  "data": {k: v for k, v in e.data.items() if k in ("hr", "ci", "p_value", "n", "events", "papers", "missing", "supports", "refutes", "counts")}})

        try:
            v = judge(c, cohorts, llm, on_evidence=on_evidence, declared_endpoint=declared,
                      context=contexts.get(c.provenance.run_id))
        except BudgetExceeded as e:
            emit({"stage": "notice", "message": f"Budget reached while refereeing: {e}"})
            return c.claim_id, None
        except Exception as e:  # noqa: BLE001 - one claim's failure must not stop the run
            emit({"stage": "notice", "message": f"One claim could not be refereed: {e.__class__.__name__}: {str(e)[:160]}"})
            return c.claim_id, None
        emit({"stage": "verdict", "claim_id": c.claim_id, "label": v.label.value, "triage": v.triage,
              "reasons": v.reasons})
        return c.claim_id, v

    data_claims = [c for c in all_claims if c.type.value != "interpretation"]
    interpretations = [c for c in all_claims if c.type.value == "interpretation"]
    with ThreadPoolExecutor(max_workers=8 if llm else 1) as pool:
        verdicts = {cid: v for cid, v in pool.map(judge_one, data_claims) if v is not None}
        for c in data_claims:
            if c.claim_id in verdicts:
                contexts.setdefault(c.provenance.run_id, []).append(
                    f"{c.text} [referees: {verdicts[c.claim_id].label.value}]")
        verdicts.update({cid: v for cid, v in pool.map(judge_one, interpretations) if v is not None})

    # 5. Consensus across runs, final verdicts, report
    judged = [c for c in all_claims if c.claim_id in verdicts]
    verdicts = apply_consensus(judged, verdicts)
    for c in judged:
        for e in verdicts[c.claim_id].evidence:
            board.post_evidence(c.claim_id, e)
        board.post_verdict(verdicts[c.claim_id])
        try:
            code = code_bundle(c.spec, verdicts[c.claim_id].evidence)
        except Exception:  # noqa: BLE001 - code display is a convenience, never a blocker
            code = []
        emit({"stage": "final_verdict", "claim_id": c.claim_id, "label": verdicts[c.claim_id].label.value,
              "triage": verdicts[c.claim_id].triage, "reasons": verdicts[c.claim_id].reasons, "code": code})

    labels: dict[str, int] = {}
    for v in verdicts.values():
        labels[v.label.value] = labels.get(v.label.value, 0) + 1
    summary = {
        "rule_version": RULE_VERSION, "mode": mode, "question": question, "wordings": goals,
        "dataset": dataset, "claims": len(judged), "labels": labels,
        "agreement_between_runs": agreement(judged, verdicts) if len(goals) > 1 else {},
        "llm": llm.summary() if llm else {"spent_usd": 0.0, "calls": 0},
        "provider": mode,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "claims.json").write_text(json.dumps(board.export(), indent=2, default=str))
    render(out_dir)
    emit({"stage": "done", "summary": summary, "report": f"/runs/{out_dir.name}/report.html"})
    return summary


def json_safe(obj):
    """Numbers from numpy/pandas -> plain JSON."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj
