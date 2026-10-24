"""
Quorum's website: type a research question, pick or upload data, watch it work.

    cd ~/ai-scientist/quorum && .venv/bin/python -m quorum.web
    then open http://localhost:8790

Built on Python's standard library only (no web framework to install). The
browser starts a run, then listens to a live stream of events (server-sent
events) and draws each step as it happens. One run at a time, so a stray
double-click can't double the spend.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from quorum.llm import load_env
from quorum.pipeline import RUNS_DIR, json_safe, run_pipeline

STATIC = Path(__file__).resolve().parent / "webapp"
PORT = int(os.environ.get("QUORUM_PORT", "8790"))
MAX_BODY = 12 * 1024 * 1024  # 12 MB of CSV
MAX_BUDGET = 10.0


class Run:
    def __init__(self):
        self.events: list[dict] = []
        self.done = False
        self.cond = threading.Condition()

    def emit(self, event: dict) -> None:
        with self.cond:
            self.events.append(json_safe(event))
            if event.get("stage") in ("done", "error"):
                self.done = True
            self.cond.notify_all()


RUNS: dict[str, Run] = {}
CHECKS: list[dict] = []  # recent single-finding checks, newest first (this session)


def history() -> dict:
    runs = []
    for d in sorted(RUNS_DIR.glob("web-*/summary.json"), reverse=True)[:10]:
        try:
            s = json.loads(d.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        runs.append({"report": f"/runs/{d.parent.name}/report.html", "question": s.get("question", ""),
                     "labels": s.get("labels", {}), "mode": s.get("mode"), "claims": s.get("claims")})
    return {"checks": CHECKS, "runs": runs}
BUSY = threading.Lock()
LIBRARY: dict[str, dict] = {}  # literature panels by question, so a second look is instant


def library_for(question: str) -> dict:
    from quorum.literature import library
    key = question.strip().lower()
    if key not in LIBRARY:
        LIBRARY[key] = library(question)
    return LIBRARY[key]


def literature_llm():
    """Kiro if signed in, else Claude if there's a key, else none (the answer is then the top passages)."""
    from quorum.kiro_llm import KiroLLM, kiro_status
    if kiro_status()["logged_in"]:
        return KiroLLM(max_calls=3)
    if has_key():
        from quorum.llm import LLM
        return LLM(budget_usd=0.25)
    return None


def has_key() -> bool:
    load_env()
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return key.startswith("sk-ant-")


def friendly(e: Exception) -> str:
    """Errors a person can act on, instead of a stack trace."""
    text = str(e)
    if "credit balance is too low" in text:
        return ("Your Anthropic account has no credits, so Claude can't run. Add credits at "
                "console.anthropic.com (Settings, Billing), or choose the free Simple analyst.")
    if "authentication" in text.lower() or "invalid x-api-key" in text.lower():
        return "Anthropic didn't accept the API key in quorum/.env. Check it was copied in full."
    if "rate_limit" in text or "429" in text:
        return "Anthropic is rate-limiting this key. Wait a minute and run again."
    if "os_months" in text or "os_event" in text:
        return text
    return f"{e.__class__.__name__}: {text[:300]}"


def start(params: dict) -> str:
    if not BUSY.acquire(blocking=False):
        raise RuntimeError("A run is already in progress. Wait for it to finish.")
    run_id = uuid.uuid4().hex[:10]
    run = RUNS[run_id] = Run()
    mode = params.get("mode", "simple")
    if mode == "ai" and not has_key():
        BUSY.release()
        raise RuntimeError("Claude mode needs your key in ~/ai-scientist/quorum/.env.")
    if mode == "kiro":
        from quorum.kiro_llm import kiro_status
        if not kiro_status()["logged_in"]:
            BUSY.release()
            raise RuntimeError("Kiro mode needs the Kiro CLI signed in: run kiro-cli login.")

    def work():
        try:
            run_pipeline(
                question=(params.get("question") or "").strip()[:2000],
                dataset=params.get("dataset", "metabric"),
                mode=mode,
                upload=params.get("csv"),
                budget=min(float(params.get("budget", 2.0)), MAX_BUDGET),
                wordings=max(1, min(int(params.get("wordings", 3)), 3)),
                emit=run.emit,
            )
        except Exception as e:  # noqa: BLE001 - report every failure to the page
            traceback.print_exc()
            run.emit({"stage": "error", "message": friendly(e)})
        finally:
            BUSY.release()

    threading.Thread(target=work, daemon=True).start()
    return run_id


class Handler(BaseHTTPRequestHandler):
    server_version = "Quorum/0.1"

    def log_message(self, fmt, *args):  # quieter console
        if "/events" not in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/options":
            from urllib.parse import parse_qs
            from quorum.checker import options
            ds = parse_qs(urlparse(self.path).query).get("dataset", ["metabric"])[0]
            return self._json(json_safe(options(ds)))
        if path == "/api/overview":
            from urllib.parse import parse_qs
            from quorum.checker import cohort
            from quorum.viz import overview
            ds = parse_qs(urlparse(self.path).query).get("dataset", ["metabric"])[0]
            return self._json(json_safe(overview(cohort("tcga" if ds == "tcga" else "metabric"))))
        if path == "/api/papers" or path.startswith("/api/papers/"):
            from urllib.parse import parse_qs
            q = parse_qs(urlparse(self.path).query).get("q", [""])[0].strip() or "breast cancer survival"
            try:
                lib = library_for(q)
            except Exception as e:  # noqa: BLE001 - usually no internet
                return self._json({"error": f"Could not reach the paper databases: {e.__class__.__name__}"}, 502)
            if path == "/api/papers":
                return self._json({"question": q, "sources": lib["sources"], "papers": lib["papers"],
                                   "passages": lib["passages"]})
            return self._json(self._read_paper(path.split("/")[3], q, lib))
        if path == "/api/history":
            return self._json(history())
        if path == "/api/status":
            from quorum.kiro_llm import kiro_status
            return self._json({"has_key": has_key(), "kiro": kiro_status()["logged_in"], "busy": BUSY.locked()})
        if path == "/sample.csv":
            from quorum.data import load
            df = load("metabric").dropna(subset=["os_months"]).head(400)
            cols = ["os_months", "os_event", "dss_months", "dss_event", "age", "grade", "tumor_size",
                    "nodes_positive", "er_status", "pam50", "esr1_z"]
            return self._send(200, df[cols].to_csv(index=False).encode(), "text/csv")
        if path.startswith("/api/runs/") and path.endswith("/events"):
            return self._stream(path.split("/")[3])
        if path.startswith("/runs/"):
            target = (RUNS_DIR / path[len("/runs/"):]).resolve()
            if RUNS_DIR.resolve() in target.parents and target.is_file() and target.suffix == ".html":
                return self._send(200, target.read_bytes(), "text/html; charset=utf-8")
        self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/check":
            from quorum.checker import check
            try:
                length = int(self.headers.get("Content-Length", 0))
                params = json.loads(self.rfile.read(min(length, 100_000)) or b"{}")
                result = json_safe(check(params))
                CHECKS.insert(0, {"text": result["text"], "label": result["label"], "params": params})
                del CHECKS[30:]
                return self._json(result)
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                return self._json({"error": friendly(e)}, 500)
        if path == "/api/papers/ask":
            from quorum.literature import answer
            params = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 100_000)) or b"{}")
            q = (params.get("question") or "").strip()
            if not q:
                return self._json({"error": "Type a question for the papers."}, 400)
            try:
                return self._json(answer(q, library_for(q), literature_llm()))
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                return self._json({"error": friendly(e)}, 500)
        if path == "/api/pdf":
            import base64
            from quorum.literature import pdf_sections, rank
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY * 2:
                return self._json({"error": "That PDF is too large (about 18 MB limit)."}, 413)
            params = json.loads(self.rfile.read(length) or b"{}")
            try:
                sections = pdf_sections(base64.b64decode(params.get("data", "")))
            except RuntimeError as e:
                return self._json({"error": str(e)}, 501)
            except Exception as e:  # noqa: BLE001 - a broken or scanned PDF
                return self._json({"error": f"Could not read that PDF ({e.__class__.__name__})."}, 400)
            name = params.get("name") or "your PDF"
            items = [{"pmid": "upload", "title": name, **s} for s in sections]
            top = [r for r in rank(params.get("question") or "breast cancer survival", items)[:8] if r["score"] > 0]
            return self._json({"name": name, "paragraphs": len(sections), "pages": len({s["section"] for s in sections}),
                               "passages": top})
        if path != "/api/run":
            return self._send(404, b"Not found", "text/plain")
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            return self._json({"error": "That file is too large (12 MB limit)."}, 413)
        try:
            params = json.loads(self.rfile.read(length) or b"{}")
            if not (params.get("question") or "").strip():
                return self._json({"error": "Type a research question first."}, 400)
            if params.get("dataset") == "upload" and not params.get("csv"):
                return self._json({"error": "Choose a CSV file to upload."}, 400)
            return self._json({"run_id": start(params)})
        except RuntimeError as e:
            return self._json({"error": str(e)}, HTTPStatus.CONFLICT)
        except (ValueError, json.JSONDecodeError) as e:
            return self._json({"error": f"Bad request: {e}"}, 400)

    @staticmethod
    def _read_paper(pmid: str, q: str, lib: dict) -> dict:
        """One paper, read in full when open access: its sections and the passages that best match the question."""
        from quorum.literature import fetch_papers, full_text, matched_terms, passages, rank
        paper = next((p for p in lib["papers"] if p["pmid"] == pmid), None)
        if paper is None:
            found = fetch_papers([pmid])
            if not found:
                return {"error": f"PubMed ID {pmid} was not found."}
            paper = found[0]
        secs = full_text(paper["pmcid"]) if paper.get("pmcid") else []
        ranked = [r for r in rank(q, passages(paper, secs)) if r["score"] > 0][:6]
        for r in ranked:
            r["matched"] = matched_terms(q, r["text"])
        sections = []
        for s in secs:
            if not sections or sections[-1]["section"] != s["section"]:
                sections.append({"section": s["section"], "paragraphs": 0})
            sections[-1]["paragraphs"] += 1
        return {"paper": paper, "full_text": bool(secs), "sections": sections, "passages": ranked}

    def _stream(self, run_id: str):
        run = RUNS.get(run_id)
        if run is None:
            return self._send(404, b"Unknown run", "text/plain")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.close_connection = True  # end the response when the run ends
        sent = 0
        try:
            while True:
                with run.cond:
                    while sent >= len(run.events) and not run.done:
                        run.cond.wait(timeout=15)
                        if sent >= len(run.events) and not run.done:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                    batch = run.events[sent:]
                    finished = run.done
                for ev in batch:
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                sent += len(batch)
                self.wfile.flush()
                if finished and sent >= len(run.events):
                    return
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Quorum is running at http://localhost:{PORT}  (key found: {has_key()})  Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
