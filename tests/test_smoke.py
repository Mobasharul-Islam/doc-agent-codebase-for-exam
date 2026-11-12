"""End-to-end tiny run. Passes once students implement the stages."""
from doc_agent import pipeline


def test_answer_is_grounded_and_cited(kb):
    ans = pipeline.answer("On what date does the Bengali year 1431 begin?", kb)
    assert ans.grounded and len(ans.citations) >= 1
    assert "fourteenth of April" in ans.text


def test_unanswerable_abstains(kb):
    ans = pipeline.answer("What is the population of the Mars colony?", kb)
    assert not ans.grounded and ans.citations == []
