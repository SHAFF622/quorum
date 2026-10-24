"""
A self-contained HTML board for one run: the page a judge or scientist reads.

Three columns (act on / check first / set aside). Each claim shows its verdict
and, one click away, every referee's evidence. No server, no extra packages:
open runs/<timestamp>/report.html in a browser.

    python -m quorum.html runs/<timestamp>
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

BADGE = {"robust": "good", "verified": "good", "contested": "warn", "unverifiable": "muted", "refuted": "bad"}
OUTCOME = {"supports": "good", "contests": "warn", "refutes": "bad", "no_evidence": "muted"}
TRIAGE = {"robust": "act on", "verified": "act on", "contested": "check first",
          "unverifiable": "set aside", "refuted": "set aside"}

CSS = """
:root{--bg:#fafaf8;--card:#fff;--ink:#1d1d1b;--quiet:#6b6b66;--line:#e4e3de;
--good:#2e7d4f;--warn:#a36a00;--bad:#b3261e;--muted:#8a8a84;--accent:#2f6fd0}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecebe6;--quiet:#a3a29c;
--line:#33332f;--good:#5cbf86;--warn:#e0a640;--bad:#ef7b72;--muted:#8f8e88;--accent:#7aa7ec}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1240px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:24px;margin:0 0 4px}.sub{color:var(--quiet);margin:0 0 20px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:24px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.stat b{display:block;font-size:22px}.stat span{color:var(--quiet);font-size:13px}
.cols{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
@media (max-width:860px){.cols{grid-template-columns:1fr}}
.col h2{font-size:15px;text-transform:uppercase;letter-spacing:.04em;color:var(--quiet);margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:10px}
.card p{margin:6px 0}.meta{color:var(--quiet);font-size:12px}
.badge{display:inline-block;font-size:12px;font-weight:600;border-radius:999px;padding:1px 9px;
border:1px solid currentColor}
.good{color:var(--good)}.warn{color:var(--warn)}.bad{color:var(--bad)}.muted{color:var(--muted)}
details{margin-top:6px}summary{cursor:pointer;color:var(--accent);font-size:13px}
ul.ev{list-style:none;padding:0;margin:8px 0 0}ul.ev li{font-size:13px;border-top:1px solid var(--line);padding:6px 0}
ul.ev b{text-transform:capitalize}.reasons{font-size:13px;color:var(--quiet);margin:4px 0 0;padding-left:18px}
"""


def _e(x) -> str:
    return html.escape(str(x))


def card(item: dict) -> str:
    c, v = item["claim"], item["verdict"] or {}
    label = v.get("label", "unverifiable")
    reasons = "".join(f"<li>{_e(r)}</li>" for r in v.get("reasons", []))
    evidence = "".join(
        f'<li><b class="{OUTCOME.get(e["outcome"], "muted")}">{_e(e["referee"])}</b> '
        f'<span class="{OUTCOME.get(e["outcome"], "muted")}">{_e(e["outcome"].replace("_", " "))}</span>'
        f' {_e(e["detail"])}</li>'
        for e in item.get("evidence", []))
    run = (c.get("provenance") or {}).get("run_id") or ""
    return (f'<div class="card"><span class="badge {BADGE.get(label, "muted")}">{_e(label)}</span> '
            f'<span class="meta">{_e(c["type"])} claim {_e(run)}</span>'
            f'<p>{_e(c["text"])}</p><ul class="reasons">{reasons}</ul>'
            f'<details><summary>Evidence from {len(item.get("evidence", []))} referee checks</summary>'
            f'<ul class="ev">{evidence}</ul></details></div>')


def render(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    items = json.loads((run_dir / "claims.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    groups = {"act on": [], "check first": [], "set aside": []}
    for it in items:
        label = (it["verdict"] or {}).get("label", "unverifiable")
        groups[TRIAGE[label]].append(it)

    agree = summary.get("agreement_between_runs") or {}

    def fmt(x):
        return "n/a" if x is None else f"{x:.2f}"

    stats = [
        (summary.get("claims", len(items)), "claims refereed"),
        (len(groups["act on"]), "act on"),
        (len(groups["check first"]), "check first"),
        (len(groups["set aside"]), "set aside"),
        (fmt(agree.get("checkable_findings")), "run agreement, all findings"),
        (fmt(agree.get("verified_findings")), "run agreement, verified"),
        (f'${(summary.get("llm") or {}).get("spent_usd", 0):.2f}', "Claude spend"),
    ]
    cols = "".join(
        f'<section class="col"><h2>{_e(name)} ({len(g)})</h2>{"".join(card(i) for i in g)}</section>'
        for name, g in groups.items())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Quorum Report</title>
<style>{CSS}</style></head><body><main>
<h1>Quorum report</h1>
<p class="sub">{len(summary.get("wordings", []))} independent runs of the same research question, every claim
re-checked by referees that each see only their own inputs. Verdict rule {_e(summary.get("rule_version", ""))}.</p>
<div class="stats">{"".join(f'<div class="stat"><b>{_e(a)}</b><span>{_e(b)}</span></div>' for a, b in stats)}</div>
<div class="cols">{cols}</div></main></body></html>"""
    out = run_dir / "report.html"
    out.write_text(page)
    return out


if __name__ == "__main__":
    print(render(Path(sys.argv[1])))
