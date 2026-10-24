# Quorum

**A referee panel for AI scientists.** AI scientists like Kosmos write reports with dozens of
claims, and the Kosmos paper itself says "there exists no automated method to reliably evaluate if a
claim is accurate, novel, and significant." Quorum is that method, for open-ended biology data:
independent referees re-check every claim, and plain code (never a model) decides the verdict.

Every claim ends up in one of three piles: **act on**, **check first**, or **set aside**, with the
evidence one click away.

## How it works

1. **Explorer** (the AI scientist) studies a research question on public breast cancer data
   (METABRIC, with TCGA for replication). It runs survival analyses through a trusted tool and writes
   claims. The same question is asked in three wordings, as three independent runs.
2. **Blind Re-analyst** reads only a claim's text, plans the analysis that would check it, and plain
   code re-runs it from raw data.
3. **Stats rules** catch the classic mistakes: too few deaths, strong words for small effects, causal
   language, multiple testing, a changed endpoint, treatment-era caveats.
4. **Skeptic** (Claude Opus) judges the interpretation. It can lower trust, never raise it.
5. **Consensus** marks a finding *robust* when independent runs find it.
6. **Verdict** is plain code, frozen and versioned by its own hash.

## The website

```bash
.venv/bin/python -m quorum.web        # then open http://localhost:8790
```

Type a research question, pick METABRIC, TCGA or upload a CSV, choose who plays the scientist
(Kiro, the free simple analyst, or Claude through the Anthropic API), and press Run.

**Kiro mode** powers the AI scientist and the AI referees with the Kiro CLI (`kiro-cli login` once).
Each call runs from an empty folder with no tools trusted, so the referees stay blind; Kiro plans
analyses, and Quorum's trusted code runs them. Five steps fill in live: check the data, frame
the question, explore, referee every claim, and the verdict cards.

## Run it

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env      # your key; .env is gitignored
.venv/bin/python -m pytest                      # 25 offline tests, no key needed
.venv/bin/python -m quorum.evaluate             # offline plumbing check
.venv/bin/python -m quorum.evaluate --llm --limit 10 --budget 2   # first measured comparison
.venv/bin/python -m quorum.run --wordings 1 --budget 1           # smallest real run
.venv/bin/python -m quorum.run --budget 5                        # full run: 3 wordings
```

Results land in `runs/<timestamp>/`: `report.md`, `claims.json`, `summary.json`, `board.sqlite`,
and `llm_calls.jsonl` (every Claude call with its cost).

## How we measure it

The evaluation set is built from settled breast cancer biology (Luminal A does best; basal-like,
HER2-enriched and Luminal B do worse; higher grade, more nodes and larger tumors do worse), each fact
confirmed in METABRIC. Each fact is then corrupted in the ways AI scientists go wrong: flipped
direction, inflated effect, causal overclaim, strong words, multiple testing, noise presented as
signal, underpowered subgroups. Because we planted the errors, we know the right answer.

Every setup is scored on the same frozen set: no checking, a single agent with the same tool, and the
Quorum panel. Referees see only the claim text.

## Data

Public data from the cBioPortal API (no login): METABRIC (`brca_metabric`) and TCGA breast cancer,
PanCancer Atlas (`brca_tcga_pan_can_atlas_2018`).
