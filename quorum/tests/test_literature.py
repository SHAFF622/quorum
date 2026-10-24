"""
The literature agent, offline: cutting papers into passages, ranking them,
reading PubMed Central's XML, and passing citations from the scientist through
to the claims the Citation checker reads.
"""

from quorum import literature
from quorum.agents import ClaimsOut, _literature_block, _to_claims
from quorum.literature import _sections_from_jats, answer, context_for, passages, rank

CDR = {"pmid": "29625055", "title": "An Integrated TCGA Pan-Cancer Clinical Data Resource",
       "abstract": "We provide clinical outcome endpoints for 11,160 patients. Overall survival and disease-specific "
                   "survival are recommended for most cancer types. Follow-up is short for some types. "
                   "Progression-free interval is the most reliable endpoint."}
HYPOXIA = {"pmid": "30643250", "title": "Molecular landmarks of tumor hypoxia across cancer types",
           "abstract": "Hypoxia shapes the mutational landscape of many tumors. We profile hypoxia scores."}


def test_passages_split_the_abstract_and_keep_the_source():
    ps = passages(CDR, [{"section": "Results", "text": "Disease-specific survival in BRCA tracks stage."}], size=2)
    assert [p["section"] for p in ps] == ["Abstract", "Abstract", "Results"]
    assert all(p["pmid"] == "29625055" for p in ps)


def test_rank_puts_the_survival_paper_first():
    pool = passages(CDR) + passages(HYPOXIA)
    ranked = rank("Which features explain survival differences in breast cancer?", pool)
    assert ranked[0]["pmid"] == "29625055" and ranked[0]["score"] > 0
    assert max(r["score"] for r in ranked if r["pmid"] == "30643250") < ranked[0]["score"] / 2


def test_jats_xml_becomes_sections():
    xml = """<article><body><sec><title>Results</title><p>Basal-like tumors had shorter disease-specific survival
    than luminal tumors across the whole cohort of patients studied here.</p><sec><title>Subgroups</title>
    <p>Among ER-positive patients the hazard ratio for <italic>ESR1</italic> expression was close to one after adjustment.</p>
    </sec></sec></body></article>"""
    secs = _sections_from_jats(xml)
    assert [s["section"] for s in secs] == ["Results", "Subgroups"]
    assert "ESR1 expression" in secs[1]["text"]


def test_context_tags_each_passage_with_its_pubmed_id():
    lib = {"passages": rank("survival endpoints", passages(CDR))}
    text = context_for(lib, n=2)
    assert text.startswith("[PMID 29625055]") and text.count("[PMID") == 2


def test_answer_without_ai_quotes_the_top_passages():
    lib = {"passages": rank("survival endpoints", passages(CDR))}
    out = answer("survival endpoints", lib, llm=None)
    assert out["cited"] == ["29625055"] and "PMID 29625055" in out["text"]


def test_answer_flags_a_made_up_citation():
    class FakeKiro:
        def ask(self, prompt, model, prompt_version):
            return "Overall survival is recommended [PMID 29625055], and so is magic [PMID 99999999]."
    lib = {"passages": rank("survival endpoints", passages(CDR))}
    out = answer("survival endpoints", lib, llm=FakeKiro())
    assert out["made_up"] == ["99999999"]


def test_scientist_citations_reach_the_claim():
    written = ClaimsOut.model_validate({"claims": [
        {"text": "Endpoints matter.", "type": "interpretation", "analysis_index": -1, "citations": ["PMID 29625055", "x"]},
        {"text": "No citations here.", "type": "interpretation", "analysis_index": -1},
    ]})
    claims = _to_claims(written, [], "goal", "run1", "model")
    assert claims[0].citations == ["29625055"] and claims[1].citations == []


def test_literature_block_is_empty_without_papers():
    assert _literature_block("") == ""
    assert "Cite only IDs listed here" in _literature_block("[PMID 1] x")


def test_pancan_shelf_has_the_eleven_papers():
    assert len(set(literature.PANCAN_PMIDS)) == 11
