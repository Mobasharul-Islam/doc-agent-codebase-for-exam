"""Unit tests for the agent, its tools, memory and HITL."""
from doc_agent.agent import hitl, hitl_store, tools
from doc_agent.agent.memory import Memory
from doc_agent.contracts import ToolResult


def test_calculator_is_safe():
    c = tools.Calculator()
    assert c(expr="2 + 3 * 4").payload["value"] == 14
    assert not c(expr="__import__('os').system('echo hi')").ok
    assert not c(expr="9 ** 9999").ok


def test_aggregate_and_extract_and_cite(kb):
    tools.bind(None, kb)
    assert tools.Aggregate()(op="sum", items=["1,250 taka", "750"]).payload["value"] == 2000
    cid = "tide_tables_c0000"
    ex = tools.Extract()(field="time", chunk_id=cid)
    assert ex.ok and ":" in ex.payload["values"][0]
    cite = tools.Cite()(chunk_id=cid, span=(0, 5))
    assert cite.ok and len(cite.payload["text"]) == 5
    assert not tools.Cite()(chunk_id=cid, span=(5, 999999)).ok


def test_agent_researches_when_evidence_weak(kb):
    from doc_agent.agent.agent import Agent
    from doc_agent.retrieval.retriever import Retriever
    from doc_agent import wiring

    wiring.register_all(kb)
    ag = Agent(kb, Retriever(kb))
    ans = ag.run("What is the population of the Mars colony?")
    assert not ans.grounded
    ks = [o.payload["k"] for o in ag.mem.items]
    assert ks == [10, 20, 30, 40]            # widened until k_max, then abstained


def test_memory_recall():
    m = Memory()
    m.add(ToolResult(ok=True, payload={"text": "tidal range 4.2 metres"}))
    m.add(ToolResult(ok=True, payload={"text": "rice yield"}))
    assert m.recall("tidal range")[0].payload["text"].startswith("tidal")


def test_hitl_queue_roundtrip(kb):
    res = hitl.escalate("low confidence", {"q": "x"})
    assert res.payload["status"] == "pending"
    iid = res.payload["item_id"]
    assert any(i["id"] == iid for i in hitl.review_queue())
    assert hitl_store.decision(iid) is None
    hitl_store.resolve(iid, "approved")
    assert hitl_store.decision(iid) == "approved"
    assert all(i["id"] != iid for i in hitl.review_queue())
