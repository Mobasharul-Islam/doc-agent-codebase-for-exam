"""Per-task verifier. FIXED signature."""
from __future__ import annotations

import re


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.:]", " ", str(s).lower())).strip()


def check(task: dict, answer: dict) -> bool:
    """Return True if `answer` satisfies `task`.
    verifiable tasks: the normalised gold string must appear in the answer text (exact fact), the
    answer must be grounded and cited. judged tasks: doc_agent.eval.judge score >= 0.5.
    An unanswerable task (gold == "") is satisfied only by an abstention."""
    gold = _norm(task.get("gold", ""))
    if not gold:
        return not answer["grounded"]
    if not answer["grounded"] or not answer["citations"]:
        return False
    if task.get("judged"):
        from doc_agent.contracts import Answer, Query
        from doc_agent.eval.judge import judge

        q = Query(text=task["question"], verifiable=task["verifiable"], judged=True)
        return judge(q, Answer(**answer)) >= 0.5
    return gold in _norm(answer["text"])
