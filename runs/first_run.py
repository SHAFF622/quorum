"""
First real Kosmos run: the README example, made small and safe.

- 1 cycle x 2 tasks instead of 5 x 10, so it costs cents, not dollars.
- Passes a real Claude client. Without one, Kosmos silently falls back to
  mock plans and mock scores and every task fails ("No 'data_analyst' agent").
- Spending is capped by RESEARCH_BUDGET_USD in kosmos-community/.env.
- Prints what the run actually spent at the end.

Run from ~/ai-scientist with the Kosmos environment's Python:
    kosmos-community/.venv/bin/python runs/first_run.py
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

KOSMOS_DIR = Path(__file__).resolve().parent.parent / "kosmos-community"
load_dotenv(KOSMOS_DIR / ".env")

key = os.environ.get("ANTHROPIC_API_KEY", "")
if not key or key.replace("9", "") == "":
    sys.exit("Set your personal ANTHROPIC_API_KEY in kosmos-community/.env first.")

from anthropic import Anthropic  # noqa: E402

from kosmos.core.metrics import get_metrics  # noqa: E402
from kosmos.workflow.research_loop import ResearchWorkflow  # noqa: E402

OBJECTIVE = (
    "Which clinical and molecular features best explain differences in overall "
    "survival in breast cancer, and in which patient subgroups?"
)


async def run():
    workflow = ResearchWorkflow(
        research_objective=OBJECTIVE,
        anthropic_client=Anthropic(api_key=key),
        artifacts_dir=str(Path(__file__).resolve().parent / "artifacts" / "first_run"),
    )
    result = await workflow.run(num_cycles=1, tasks_per_cycle=2)
    report = await workflow.generate_report()

    print(report)
    print("\n--- run summary ---")
    print(f"cycles completed: {result.get('cycles_completed')}")
    print(f"findings validated: {result.get('validated_findings')}")
    print(f"spent: ${get_metrics().spend_usd:.4f}")
    print(f"stopped by budget: {getattr(workflow, 'budget_exceeded', False)}")


asyncio.run(run())
