"""Stage 9 — LLM-as-judge for non-verifiable inference"""
from __future__ import annotations

import re

from ..contracts import *  # noqa
from ..llm import prompts
from ..llm.client import LLM, content_words


def judge(query: Query, answer: Answer) -> float:
    """Score an open-ended answer in [0, 1]. With an LLM key: the JUDGE prompt, parsed as a float.
    Offline: a transparent heuristic (abstentions and uncited answers score 0; otherwise lexical
    overlap between question and answer, damped by confidence) — a stand-in, not a real judge."""
    if not answer.grounded or not answer.citations:
        return 0.0
    llm = LLM({})
    if llm.online:
        raw = llm.complete(
            prompts.JUDGE.format(question=query.text, answer=answer.text,
                                 evidence=", ".join(c.chunk_id for c in answer.citations)),
            max_tokens=10,
        )
        m = re.search(r"\d(?:\.\d+)?", raw)
        if m:
            return min(1.0, max(0.0, float(m.group())))
    qw = set(content_words(query.text))
    aw = set(content_words(answer.text))
    overlap = len(qw & aw) / (len(qw) or 1)
    return round(min(1.0, 0.5 + 0.5 * overlap) * max(answer.confidence, 0.5), 3)
