"""
One small door to Claude, used by every AI part of Quorum.

What it adds on top of the Anthropic SDK:
- A hard dollar budget. Each call is priced from the tokens the API reports;
  once the budget is spent, the next call is refused before it is sent.
- JSON answers checked against a Pydantic schema (structured outputs), so the
  rest of the code never parses free text.
- Prompt caching on the system prompt, which every referee call repeats.
- Refusals handled explicitly. A safety classifier may decline a biology
  request; we ask the API to retry on its recommended fallback model
  (`fallbacks: "default"`) and, if the whole chain still refuses, raise
  `Refused` so the caller records "no evidence" instead of crashing.
- A call log (JSON lines): model, prompt version, tokens, cost, for every call.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypeVar

import anthropic
from pydantic import BaseModel

SONNET = "claude-sonnet-5-5"
OPUS = "claude-opus-5-5"
HAIKU = "claude-haiku-5-5"

# Dollars per million tokens: (input, output, cache read). Cached 2026-10-06.
PRICES = {
    SONNET: (2.00, 10.00, 0.20),
    OPUS: (4.00, 20.00, 0.20),
    HAIKU: (0.10, 0.50, 0.01),
}
CACHE_WRITE_MULTIPLIER = 1.25
FALLBACK_BETA = "server-side-fallback-2026-07-01"

T = TypeVar("T", bound=BaseModel)


class BudgetExceeded(RuntimeError):
    pass


class Refused(RuntimeError):
    def __init__(self, category: Optional[str], explanation: str = ""):
        self.category = category
        super().__init__(f"Claude declined ({category or 'unspecified'}): {explanation}")


ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path = ENV_FILE) -> None:
    """Read KEY=value lines from quorum/.env (gitignored) into the environment."""
    import os
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _strict(schema: dict) -> dict:
    """
    Make a Pydantic JSON schema as plain as structured outputs like it:
    references inlined (no $defs), every object closed to unknown keys, every
    property required, and no default values. Pydantic still validates the answer.
    """
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(dict(defs[node["$ref"].split("/")[-1]]))
        out = {k: walk(v) for k, v in node.items() if k not in ("$defs", "default", "title")}
        if out.get("type") == "object" or "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out.get("properties", {}))
        return out

    return walk(schema)


def cost_of(model: str, usage: Any) -> float:
    price_in, price_out, price_cache = PRICES.get(model, PRICES[OPUS])
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return (
        (usage.input_tokens or 0) * price_in
        + cache_write * price_in * CACHE_WRITE_MULTIPLIER
        + cache_read * price_cache
        + (usage.output_tokens or 0) * price_out
    ) / 1_000_000


@dataclass
class LLM:
    budget_usd: float = 5.0
    log_path: Optional[Path] = None
    use_fallbacks: bool = True
    client: Any = None
    spent_usd: float = 0.0
    calls: int = 0
    _log: list = field(default_factory=list)

    def __post_init__(self):
        if self.client is None:
            load_env()
            self.client = anthropic.Anthropic()

    # -- budget -------------------------------------------------------------

    def _check_budget(self) -> None:
        if self.spent_usd >= self.budget_usd:
            raise BudgetExceeded(f"Spent ${self.spent_usd:.2f} of ${self.budget_usd:.2f}.")

    def _record(self, model: str, prompt_version: str, response: Any, seconds: float) -> None:
        cost = cost_of(response.model or model, response.usage)
        self.spent_usd += cost
        self.calls += 1
        row = {
            "model": response.model or model, "prompt_version": prompt_version,
            "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens,
            "cache_read": getattr(response.usage, "cache_read_input_tokens", 0),
            "cost_usd": round(cost, 5), "seconds": round(seconds, 2),
            "stop_reason": response.stop_reason,
        }
        self._log.append(row)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

    # -- calls --------------------------------------------------------------

    def create(self, *, model: str, system: str, messages: list, prompt_version: str,
               effort: str = "medium", max_tokens: int = 16000, **kwargs) -> Any:
        """A Messages API call with budget, caching, fallbacks and logging."""
        self._check_budget()
        extra_headers, extra_body = {}, {}
        if self.use_fallbacks and model in (SONNET, OPUS):
            extra_headers["anthropic-beta"] = FALLBACK_BETA
            extra_body["fallbacks"] = "default"
        output_config = kwargs.pop("output_config", {})
        output_config.setdefault("effort", effort)
        request = dict(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            output_config=output_config,
            **kwargs,
        )
        started = time.monotonic()
        try:
            response = self.client.messages.create(
                **request, extra_headers=extra_headers or None, extra_body=extra_body or None)
        except anthropic.BadRequestError as e:
            if not extra_body:
                raise
            # The fallback option may not combine with every feature on every
            # model. Note it once and carry on without it rather than fail the run.
            print(f"[llm] request rejected with fallbacks on ({e.message[:160]}); retrying without them")
            self.use_fallbacks = False
            response = self.client.messages.create(**request)
        self._record(model, prompt_version, response, time.monotonic() - started)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise Refused(getattr(details, "category", None), getattr(details, "explanation", "") or "")
        return response

    def json(self, schema: type[T], *, model: str, system: str, user: str,
             prompt_version: str, effort: str = "medium", max_tokens: int = 16000) -> T:
        """Ask for one JSON object matching `schema` and return it validated."""
        response = self.create(
            model=model, system=system, prompt_version=prompt_version, effort=effort,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema",
                                      "schema": _strict(schema.model_json_schema())}},
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return schema.model_validate_json(text)

    def summary(self) -> dict:
        return {"calls": self.calls, "spent_usd": round(self.spent_usd, 4), "budget_usd": self.budget_usd}
