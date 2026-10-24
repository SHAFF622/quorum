"""
The AI parts: an explorer that makes claims, and referees that check them.

- Explorer (the "AI scientist"): given a research goal, it runs survival
  analyses through our trusted tool and writes up claims. It never runs its own
  code, so its mistakes show up where Kosmos's do: in what it says the numbers
  mean.
- Blind Re-analyst: reads only the claim's text (never the explorer's analysis),
  decides what analysis would check it and which numbers it states. Plain code
  then re-runs that analysis and compares (quorum.reanalyst).
- Skeptic: reads the claim with the re-analysis numbers and judges the
  interpretation: overclaiming, confounding, causal talk, scope. It can only
  lower trust.
- Single agent (the baseline): the same model with the same tool, asked "is
  this claim correct?". The panel has to beat this.
"""

from __future__ import annotations

import json
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from quorum.claims import Claim, ClaimType, Evidence, Label, Provenance, Stated, Verdict
from quorum.data import dictionary
from quorum.llm import LLM, OPUS, SONNET, Refused
from quorum.reanalyst import reanalyse
from quorum.survival import AnalysisSpec, Filter, analyse

# ---------------------------------------------------------------------------
# Shapes Claude fills in (kept simple so structured outputs accept them)
# ---------------------------------------------------------------------------


class FilterOut(BaseModel):
    column: str
    op: Literal["==", "!=", "<", "<=", ">", ">="]
    value: str


class SpecOut(BaseModel):
    cohort: str = Field(description="Which cohort to analyse: one of the cohort names given in the data description.")
    variable: str
    exposed: str = Field(description="Exposed level for a categorical comparison, or empty for continuous.")
    reference: str = Field(description="Reference level for a categorical comparison, or empty for continuous.")
    subgroup: list[FilterOut]
    adjust_for: list[str] = Field(description="Numeric columns to adjust for; empty if none.")
    endpoint: Literal["OS", "DSS"] = Field(
        default="OS", description="OS: death from any cause. DSS: death from breast cancer only.")


def to_spec(s: SpecOut) -> AnalysisSpec:
    def val(v: str):
        try:
            return float(v)
        except ValueError:
            return v
    return AnalysisSpec(
        cohort=s.cohort, variable=s.variable,
        exposed=s.exposed or None, reference=s.reference or None,
        subgroup=[Filter(column=f.column, op=f.op, value=val(f.value)) for f in s.subgroup],
        adjust_for=s.adjust_for, endpoint=s.endpoint,
    )


def _tool_schema(model: type[BaseModel]) -> dict:
    from quorum.llm import _strict
    return _strict(model.model_json_schema())


def run_tool(spec_json: dict, cohorts: dict[str, pd.DataFrame]) -> tuple[dict, Optional[AnalysisSpec]]:
    """The survival tool both the explorer and the baseline may call."""
    try:
        spec = to_spec(SpecOut.model_validate(spec_json))
        r = analyse(cohorts[spec.cohort], spec)
    except Exception as e:  # noqa: BLE001 - report any failure back to the model
        return {"error": str(e)}, None
    return {
        "hr": round(r.hr, 3), "ci_95": [round(r.ci_low, 3), round(r.ci_high, 3)],
        "p_value": float(f"{r.p_value:.3g}"), "n": r.n, "deaths": r.events,
        "deaths_exposed": r.events_exposed, "deaths_reference": r.events_reference,
        "notes": r.notes,
    }, spec


DATA_NOTES = (
    "Notes on the data: overall survival (OS) counts deaths from any cause, so age matters a lot; "
    "in METABRIC 497 of 1,144 deaths were from other causes. Disease-specific survival (DSS) counts "
    "only breast cancer deaths. "
    "METABRIC patients were diagnosed between 1977 and 2005, mostly before trastuzumab for HER2. "
    "TCGA has short follow-up (median about 27 months) and few deaths. "
    "Hazard ratio (HR) above 1 means the exposed group, or higher values, die sooner."
)

# ---------------------------------------------------------------------------
# Explorer
# ---------------------------------------------------------------------------

EXPLORER_PROMPT_VERSION = "explorer-v1"
EXPLORER_SYSTEM = """You are an AI scientist studying breast cancer survival in public cohort data.

You can run Cox survival analyses with the `run_survival_analysis` tool. Investigate the research goal,
then call `submit_claims` once with your findings: data claims (a number you computed, citing the
index of the analysis it came from) and interpretation claims (what the findings mean).
Write each claim as one self-contained sentence a scientist could check, stating the numbers.

{dictionary}

{notes}"""


class ClaimOut(BaseModel):
    text: str
    type: Literal["data", "interpretation"]
    analysis_index: int = Field(description="Index of the analysis this claim reports; -1 for interpretation claims.")
    citations: list[str] = Field(default_factory=list,
                                 description="PubMed IDs (digits only) of papers from the literature that back this "
                                             "sentence; empty if none do.")


class ClaimsOut(BaseModel):
    claims: list[ClaimOut]


def _literature_block(literature: str) -> str:
    """What the literature agent found, for the scientist's prompt. It may cite these papers, and only these."""
    if not literature:
        return ""
    return ("\n\nWHAT PUBLISHED PAPERS SAY (from the literature agent). Use these to plan and to interpret. "
            "When a passage below backs a sentence you write, put its PubMed ID in that claim's citations. "
            "Cite only IDs listed here; the Citation checker will read each cited paper.\n" + literature)


def explore(goal: str, cohorts: dict[str, pd.DataFrame], llm: LLM, run_id: str,
            max_analyses: int = 20, model: str = SONNET, on_analysis=None, literature: str = "") -> list[Claim]:
    """Run one exploration and return its claims (not yet judged)."""
    system = EXPLORER_SYSTEM.format(
        dictionary="\n\n".join(dictionary(df) for df in cohorts.values()), notes=DATA_NOTES) + _literature_block(literature)
    tools = [
        {"name": "run_survival_analysis", "strict": True,
         "description": "Run a Cox survival model (endpoint OS or DSS). For a categorical variable give "
                        "exposed and reference levels; for a continuous variable leave both empty (HR is per SD).",
         "input_schema": _tool_schema(SpecOut)},
        {"name": "submit_claims", "strict": True,
         "description": "Submit your final claims. Call this exactly once, when done.",
         "input_schema": _tool_schema(ClaimsOut)},
    ]
    messages: list = [{"role": "user", "content": f"Research goal: {goal}"}]
    analyses: list[tuple[dict, Optional[AnalysisSpec]]] = []

    for _ in range(max_analyses + 5):
        response = llm.create(model=model, system=system, messages=messages, tools=tools,
                              prompt_version=EXPLORER_PROMPT_VERSION, effort="medium")
        messages.append({"role": "assistant", "content": response.content})
        uses = [b for b in response.content if b.type == "tool_use"]
        if not uses:
            messages.append({"role": "user", "content": "Please call submit_claims with your findings."})
            continue
        results = []
        for use in uses:
            if use.name == "submit_claims":
                submitted = ClaimsOut.model_validate(use.input)
                return _to_claims(submitted, analyses, goal, run_id, model)
            if len(analyses) >= max_analyses:
                out = {"error": f"Analysis limit of {max_analyses} reached; submit your claims now."}
            else:
                out, spec = run_tool(use.input, cohorts)
                analyses.append((out, spec))
                out = {"analysis_index": len(analyses) - 1, **out}
                if on_analysis:
                    on_analysis(spec, out)
            results.append({"type": "tool_result", "tool_use_id": use.id, "content": json.dumps(out)})
        messages.append({"role": "user", "content": results})
    raise RuntimeError("Explorer did not submit claims.")


def _to_claims(submitted: ClaimsOut, analyses, goal: str, run_id: str, model: str) -> list[Claim]:
    tests_run = sum(1 for _, s in analyses if s is not None)
    claims = []
    for c in submitted.claims:
        spec = None
        if c.type == "data" and 0 <= c.analysis_index < len(analyses):
            spec = analyses[c.analysis_index][1]
        claims.append(Claim(
            type=ClaimType(c.type), text=c.text,
            citations=[x for x in (str(x).strip().removeprefix("PMID").strip() for x in c.citations) if x.isdigit()],
            stated=Stated(tests_run=tests_run),
            provenance=Provenance(
                source="explorer", run_id=run_id, goal_wording=goal, model=model,
                prompt_version=EXPLORER_PROMPT_VERSION,
                code=spec.model_dump_json() if spec else None,
            ),
        ))
    return claims


# ---------------------------------------------------------------------------
# Blind Re-analyst
# ---------------------------------------------------------------------------

REANALYST_PROMPT_VERSION = "reanalyst-v1"
REANALYST_SYSTEM = """You are a blind re-analyst. You receive one claim about breast cancer survival,
written by someone else. You never see their analysis. Work out:
1. Whether it is a data claim that a single Cox survival analysis on these columns can check.
2. If so, exactly which analysis would check it (cohort, variable, levels, subgroup, adjustments, endpoint).
   Include every restriction the sentence states: "ER-positive" means subgroup er_status == Positive; a
   subtype such as "Luminal B" means pam50 == LumB; "who received hormone therapy" means hormone_therapy == YES;
   "adjusted for age, grade, ..." means adjust_for those columns; "breast cancer death", "cancer-specific" or
   "disease-specific" means endpoint DSS; "all-cause", "overall survival" or "death from any cause" means OS.
   If the claim says there is NO significant association, set direction to "".
3. The numbers the claim itself states: direction for the exposed group or for higher values
   ("worse" or "better"), hazard ratio, p-value, n. Use -1 for a number not stated, "" for no direction.

{dictionary}

{notes}"""


class StatedOut(BaseModel):
    direction: Literal["worse", "better", ""]
    hr: float
    p_value: float
    n: int


class ReanalysisPlan(BaseModel):
    checkable: bool
    reason: str
    spec: SpecOut
    stated: StatedOut


def blind_reanalysis(claim: Claim, cohorts: dict[str, pd.DataFrame], llm: LLM) -> tuple[Evidence, Optional[object]]:
    """Plan the check from the text alone, then let plain code run and compare it."""
    system = REANALYST_SYSTEM.format(
        dictionary="\n\n".join(dictionary(df) for df in cohorts.values()), notes=DATA_NOTES)
    try:
        plan = llm.json(ReanalysisPlan, model=SONNET, system=system, user=f"Claim: {claim.text}",
                        prompt_version=REANALYST_PROMPT_VERSION, effort="low")
    except Refused as e:
        return Evidence(referee="reanalyst", outcome="no_evidence", detail=str(e)), None
    if not plan.checkable:
        return Evidence(referee="reanalyst", outcome="no_evidence",
                        detail=f"Not checkable against the data: {plan.reason}"), None
    st = plan.stated
    planned = claim.model_copy(update={
        "type": ClaimType.DATA,
        "spec": to_spec(plan.spec),
        "stated": Stated(
            direction=st.direction or None,
            hr=st.hr if st.hr > 0 else None,
            p_value=st.p_value if st.p_value >= 0 else None,
            n=st.n if st.n > 0 else None,
            tests_run=claim.stated.tests_run,
        ),
    })
    evidence, ref = reanalyse(planned, cohorts)
    evidence.data["planned_spec"] = json.loads(planned.spec.model_dump_json())
    return evidence, (planned, ref)


# ---------------------------------------------------------------------------
# Skeptic
# ---------------------------------------------------------------------------

SKEPTIC_PROMPT_VERSION = "skeptic-v1"
SKEPTIC_SYSTEM = """You are a skeptical senior biostatistician reviewing one claim from an AI scientist
about breast cancer survival. Your job is to catch interpretation errors: calling a significant result
important when the effect is small, causal language on observational data, ignoring confounding (for
example by age, since overall survival counts all deaths), generalising beyond the subgroup or era,
claims resting on too few deaths, or numbers that don't match the re-analysis.

Answer "ok" if the claim is a fair reading of the evidence, "contests" if a scientist should check it
before relying on it, "refutes" if it is wrong or misleading as written. Be specific and brief.

{notes}"""


class SkepticOut(BaseModel):
    outcome: Literal["ok", "contests", "refutes"]
    problem: Literal["none", "overclaim", "confounding", "causal_language", "scope", "underpowered",
                     "numbers_mismatch", "other"]
    reason: str


def skeptic(claim: Claim, reanalysis: Optional[Evidence], rule_flags: list[Evidence], llm: LLM,
            context: Optional[list[str]] = None) -> Evidence:
    parts = [f"Claim ({claim.type.value}): {claim.text}"]
    if context:
        parts.append("This is an interpretation. Judge it only against the same report's data findings, "
                     "already checked by the referees:\n" + "\n".join(f"- {c}" for c in context))
    if reanalysis is not None:
        parts.append(f"Independent re-analysis: {reanalysis.outcome}. {reanalysis.detail}")
    if rule_flags:
        parts.append("Automatic rule flags: " + "; ".join(e.detail for e in rule_flags))
    if claim.stated.tests_run:
        parts.append(f"The AI scientist ran {claim.stated.tests_run} analyses in this session.")
    try:
        out = llm.json(SkepticOut, model=OPUS, system=SKEPTIC_SYSTEM.format(notes=DATA_NOTES),
                       user="\n".join(parts), prompt_version=SKEPTIC_PROMPT_VERSION, effort="medium")
    except Refused as e:
        return Evidence(referee="skeptic", outcome="no_evidence", detail=str(e))
    outcome = {"ok": "supports", "contests": "contests", "refutes": "refutes"}[out.outcome]
    return Evidence(referee="skeptic", outcome=outcome, detail=out.reason, data={"problem": out.problem})


# ---------------------------------------------------------------------------
# Single-agent baseline
# ---------------------------------------------------------------------------

BASELINE_PROMPT_VERSION = "single-agent-v1"
BASELINE_SYSTEM = """You check claims about breast cancer survival. You can run Cox survival analyses
with `run_survival_analysis`. Decide whether the claim is correct, then call `submit_verdict` once.

{dictionary}

{notes}"""


class BaselineOut(BaseModel):
    verdict: Literal["correct", "incorrect", "unsure"]
    reason: str


def single_agent(claim: Claim, cohorts: dict[str, pd.DataFrame], llm: LLM, max_analyses: int = 4) -> Verdict:
    system = BASELINE_SYSTEM.format(
        dictionary="\n\n".join(dictionary(df) for df in cohorts.values()), notes=DATA_NOTES)
    tools = [
        {"name": "run_survival_analysis", "strict": True,
         "description": "Run a Cox model of overall survival (categorical: exposed vs reference; continuous: per SD).",
         "input_schema": _tool_schema(SpecOut)},
        {"name": "submit_verdict", "strict": True, "description": "Submit your verdict. Call once.",
         "input_schema": _tool_schema(BaselineOut)},
    ]
    messages: list = [{"role": "user", "content": f"Claim: {claim.text}"}]
    used = 0
    try:
        for _ in range(max_analyses + 3):
            response = llm.create(model=SONNET, system=system, messages=messages, tools=tools,
                                  prompt_version=BASELINE_PROMPT_VERSION, effort="medium")
            messages.append({"role": "assistant", "content": response.content})
            uses = [b for b in response.content if b.type == "tool_use"]
            if not uses:
                messages.append({"role": "user", "content": "Please call submit_verdict."})
                continue
            results = []
            for use in uses:
                if use.name == "submit_verdict":
                    out = BaselineOut.model_validate(use.input)
                    label = {"correct": Label.VERIFIED, "incorrect": Label.REFUTED,
                             "unsure": Label.CONTESTED}[out.verdict]
                    return Verdict(claim_id=claim.claim_id, label=label, reasons=[out.reason],
                                   evidence=[], rule_version=BASELINE_PROMPT_VERSION)
                used += 1
                out = run_tool(use.input, cohorts)[0] if used <= max_analyses else {"error": "limit reached"}
                results.append({"type": "tool_result", "tool_use_id": use.id, "content": json.dumps(out)})
            messages.append({"role": "user", "content": results})
    except Refused as e:
        return Verdict(claim_id=claim.claim_id, label=Label.UNVERIFIABLE, reasons=[str(e)],
                       evidence=[], rule_version=BASELINE_PROMPT_VERSION)
    return Verdict(claim_id=claim.claim_id, label=Label.UNVERIFIABLE, reasons=["no verdict"],
                   evidence=[], rule_version=BASELINE_PROMPT_VERSION)


# ---------------------------------------------------------------------------
# Explorer for prompt-only AI (Kiro): plan, run with trusted code, write up
# ---------------------------------------------------------------------------

PLANNED_PROMPT_VERSION = "explorer-planned-v1"


class PlanOut(BaseModel):
    analyses: list[SpecOut] = Field(description="Survival analyses to run, most informative first.")


def explore_planned(goal: str, cohorts: dict[str, pd.DataFrame], llm, run_id: str,
                    max_analyses: int = 12, max_claims: int = 8, model: str = SONNET,
                    on_analysis=None, literature: str = "") -> list[Claim]:
    """
    The same AI scientist for an AI that can't call tools directly.
    1. The AI plans a list of survival analyses for the goal.
    2. Quorum's trusted code runs them (the AI never runs code).
    3. The AI reads the results and writes its findings.
    """
    data = "\n\n".join(dictionary(df) for df in cohorts.values())
    plan = llm.json(PlanOut, model=model, prompt_version=PLANNED_PROMPT_VERSION + "-plan",
                    system=("You are an AI scientist studying breast cancer survival in public cohort data. "
                            f"Plan up to {max_analyses} Cox survival analyses that best answer the research goal. "
                            "For a categorical variable give exposed and reference levels exactly as listed; for a "
                            "continuous variable leave both empty. Use only the columns listed.\n\n"
                            f"{data}\n\n{DATA_NOTES}{_literature_block(literature)}"),
                    user=f"Research goal: {goal}")

    analyses: list[tuple[dict, Optional[AnalysisSpec]]] = []
    for s in plan.analyses[:max_analyses]:
        out, spec = run_tool(s.model_dump(), cohorts)
        analyses.append((out, spec))
        if on_analysis:
            on_analysis(spec, {"analysis_index": len(analyses) - 1, **out})

    results = [{"analysis_index": i, "analysis": (s.model_dump(mode="json") if s else None), "result": out}
               for i, (out, s) in enumerate(analyses)]
    written = llm.json(ClaimsOut, model=model, prompt_version=PLANNED_PROMPT_VERSION + "-write",
                       system=("You are an AI scientist. Here are the results of the survival analyses you planned. "
                               f"Write up to {max_claims} findings: data claims (one number you computed, citing its "
                               "analysis_index) and interpretation claims (what the findings mean, analysis_index -1). "
                               "Each claim is one self-contained sentence a scientist could check, stating the numbers, the patient "
                               "group, any adjustments, and whether deaths are from any cause or from breast cancer. Never mention "
                               "analysis indexes or internal labels in the sentence.\n\n"
                               f"{DATA_NOTES}{_literature_block(literature)}"),
                       user=f"Research goal: {goal}\n\nResults:\n{json.dumps(results, default=str)}")
    written.claims = written.claims[:max_claims]
    return _to_claims(written, analyses, goal, run_id, model)
