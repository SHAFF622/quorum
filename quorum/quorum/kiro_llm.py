"""
Kiro as Quorum's AI: the same job as quorum/llm.py, powered by the Kiro CLI.

Quorum's AI referees only ever need "send a prompt, get back JSON that fits a
schema". The Kiro CLI can do that in its non-interactive mode, so the
hackathon's Kiro credits can power the referees and the explorer.

Safety choices:
- Every call runs in an empty, throwaway folder with no tools trusted, so Kiro
  can't read the project (the referees must stay blind) or run commands.
- A hard cap on the number of calls stands in for a dollar budget (Kiro bills
  in credits, not dollars).
- The answer must parse as JSON matching the schema; one retry with the
  validation error, then a clear failure.
- The explorer never runs Kiro's code: Kiro plans analyses, Quorum's trusted
  survival code runs them.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TypeVar

from pydantic import BaseModel, ValidationError

from quorum.llm import HAIKU, OPUS, SONNET, BudgetExceeded, _strict

T = TypeVar("T", bound=BaseModel)
KIRO = shutil.which("kiro-cli") or str(Path.home() / ".local/bin/kiro-cli")

# Quorum's model roles -> models Kiro offers.
MODELS = {
    SONNET: "claude-sonnet-4.5",
    OPUS: "claude-sonnet-4.5",
    HAIKU: "claude-haiku-4.5",
}


def kiro_status() -> dict:
    """Is the Kiro CLI installed and signed in?"""
    try:
        out = subprocess.run([KIRO, "whoami"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return {"installed": False, "logged_in": False}
    text = (out.stdout + out.stderr).strip()
    return {"installed": True, "logged_in": out.returncode == 0 and "Logged in" in text}


class KiroError(RuntimeError):
    pass


def _extract_json(text: str) -> str:
    """Kiro may wrap JSON in a code fence or a sentence; take the outermost object."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise KiroError(f"No JSON object in Kiro's reply: {text[:200]!r}")
    return text[start:end + 1]


@dataclass
class KiroLLM:
    max_calls: int = 60
    timeout_s: int = 180
    log_path: Optional[Path] = None
    model_map: dict = field(default_factory=lambda: dict(MODELS))
    calls: int = 0
    seconds: float = 0.0
    provider: str = "kiro"
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _reserve(self) -> None:
        with self._lock:
            if self.calls >= self.max_calls:
                raise BudgetExceeded(f"Used all {self.max_calls} Kiro calls allowed for this run.")
            self.calls += 1

    def ask(self, prompt: str, model: str, prompt_version: str) -> str:
        """One non-interactive Kiro call from an empty folder; returns the reply text."""
        self._reserve()
        kiro_model = self.model_map.get(model, model)
        started = time.monotonic()
        error = None
        for attempt in range(2):  # one retry: a busy Kiro sometimes drops a call when many run at once
            with tempfile.TemporaryDirectory(prefix="quorum-kiro-") as blank:
                try:
                    out = subprocess.run(
                        [KIRO, "chat", "--no-interactive", "--trust-tools=", "--model", kiro_model, prompt],
                        capture_output=True, text=True, timeout=self.timeout_s, cwd=blank,
                    )
                    error = None if out.returncode == 0 else \
                        f"Kiro CLI failed ({out.returncode}): {(out.stderr or out.stdout)[-300:]}"
                except subprocess.TimeoutExpired:
                    error = f"Kiro took longer than {self.timeout_s}s."
            if error is None:
                break
            time.sleep(2 + 3 * attempt)
        took = time.monotonic() - started
        with self._lock:
            self.seconds += took
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps({"provider": "kiro", "model": kiro_model, "prompt_version": prompt_version,
                                    "seconds": round(took, 1), "chars_in": len(prompt),
                                    "chars_out": 0 if error else len(out.stdout), "error": error}) + "\n")
        if error:
            raise KiroError(error)
        return out.stdout

    def json(self, schema: type[T], *, model: str, system: str, user: str, prompt_version: str,
             effort: str = "medium", max_tokens: int = 16000) -> T:
        shape = json.dumps(_strict(schema.model_json_schema()))
        prompt = (f"{system}\n\n{user}\n\nReply with only one JSON object that matches this JSON schema. "
                  f"No prose, no code fences, no tools.\nSchema: {shape}")
        reply = self.ask(prompt, model, prompt_version)
        try:
            return schema.model_validate_json(_extract_json(reply))
        except (ValidationError, KiroError, ValueError) as first:
            retry = (f"{prompt}\n\nYour previous reply could not be used ({str(first)[:300]}). "
                     f"Reply again with only the JSON object.")
            reply = self.ask(retry, model, prompt_version + "-retry")
            return schema.model_validate_json(_extract_json(reply))

    def summary(self) -> dict:
        return {"provider": "kiro", "calls": self.calls, "max_calls": self.max_calls,
                "seconds": round(self.seconds, 1), "spent_usd": 0.0}
