"""Unit tests for chunking + retrieval."""
from doc_agent.contracts import Chunk
from doc_agent.index import chunk
from doc_agent.retrieval import retriever as rt


def test_chunk_split_windows_and_pages():
    cfg = {"index": {"chunk_tokens": 4, "overlap": 1}}
    src = [Chunk(id="a", doc_id="d", text="one two three", page_ids=["p1"]),
           Chunk(id="b", doc_id="d", text="four five six seven", page_ids=["p2"])]
    out = chunk.split(src, cfg)
    assert [c.text for c in out] == ["one two three four", "four five six seven"]
    assert out[1].page_ids == ["p2"] and out[0].page_ids == ["p1", "p2"]


def test_retrieve_scores_and_relevance(kb):
    r = rt.Retriever(kb)
    res = r.retrieve("tidal range of the Karnaphuli estuary", 5)
    assert res[0].doc_id == "tide_tables" and res[0].score > 0.35
    assert [c.score for c in res] == sorted((c.score for c in res), reverse=True)


def test_evidence_policy():
    cfg = {"retrieve": {"weak_threshold": 0.35, "k_step": 10, "k_max": 40}}
    assert rt.is_weak([Chunk(id="x", doc_id="d", text="t", page_ids=[], score=0.2)], cfg)
    assert not rt.is_weak([Chunk(id="x", doc_id="d", text="t", page_ids=[], score=0.5)], cfg)
    assert rt.next_k(30, cfg) == 40 and rt.next_k(40, cfg) is None
