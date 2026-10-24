"""
The Re-analyst's judging step: re-run a claim's analysis and compare.

This is the part that needs no AI. Given a claim's structured spec, it runs
the trusted reference analysis (quorum.survival) on the raw data, never the
explorer's own code, and compares the result with what the claim states.
That mirrors how Kosmos's human experts graded data claims: re-derive the
number yourself, then decide if the claim holds.

(Milestone 2 adds the AI half: reading a free-text claim and writing the spec
without seeing the explorer's code.)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from quorum.claims import Claim, ClaimType, Evidence
from quorum.survival import SurvivalResult, analyse

# How far a stated hazard ratio may sit outside the re-analysis confidence
# interval before we call it misstated (10% slack for rounding and method).
HR_TOLERANCE = 1.10
ALPHA = 0.05


def reanalyse(claim: Claim, cohorts: dict[str, pd.DataFrame]) -> tuple[Evidence, Optional[SurvivalResult]]:
    if claim.type != ClaimType.DATA or claim.spec is None:
        return Evidence(referee="reanalyst", outcome="no_evidence",
                        detail="Not a data claim with an analysis spec."), None

    df = cohorts.get(claim.spec.cohort)
    if df is None:
        return Evidence(referee="reanalyst", outcome="no_evidence",
                        detail=f"Cohort {claim.spec.cohort!r} not loaded."), None
    try:
        ref = analyse(df, claim.spec)
    except (ValueError, KeyError) as e:
        return Evidence(referee="reanalyst", outcome="no_evidence",
                        detail=f"Could not re-run the analysis: {e}"), None

    s = claim.stated
    data = {
        "hr": round(ref.hr, 3), "ci": [round(ref.ci_low, 3), round(ref.ci_high, 3)],
        "p_value": ref.p_value, "n": ref.n, "events": ref.events,
        "events_exposed": ref.events_exposed, "events_reference": ref.events_reference,
    }
    found = f"HR {ref.hr:.2f} (95% CI {ref.ci_low:.2f} to {ref.ci_high:.2f}), p={ref.p_value:.2g}, n={ref.n}, deaths={ref.events}"
    significant = ref.p_value < ALPHA

    problems = []
    if s.direction and significant and s.direction != ref.direction:
        problems.append(f"Direction is wrong: the data say {ref.direction}, the claim says {s.direction}.")
    if s.hr is not None and not (ref.ci_low / HR_TOLERANCE <= s.hr <= ref.ci_high * HR_TOLERANCE):
        problems.append(f"Stated HR {s.hr:.2f} is outside the re-analysed range.")
    if s.p_value is not None and s.p_value < ALPHA and not significant:
        problems.append(f"Claim says significant (p={s.p_value:.2g}); re-analysis is not (p={ref.p_value:.2g}).")
    claims_nonsignificant = s.p_value is not None and s.p_value >= ALPHA
    if s.direction and not significant and not claims_nonsignificant:
        problems.append("The data show no clear association (95% CI includes no effect).")

    if not problems and s.p_value is not None and s.p_value >= ALPHA and not significant:
        return Evidence(referee="reanalyst", outcome="supports",
                        detail=f"Reproduced: no significant effect, as stated. {found}.", data=data), ref

    if problems:
        return Evidence(referee="reanalyst", outcome="refutes",
                        detail=" ".join(problems) + f" Re-analysis: {found}.", data=data), ref
    return Evidence(referee="reanalyst", outcome="supports",
                    detail=f"Reproduced. {found}.", data=data), ref
