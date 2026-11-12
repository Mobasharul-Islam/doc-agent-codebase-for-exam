"""Stage 9 — metrics"""
from __future__ import annotations

from collections import Counter

from ..contracts import *  # noqa
from . import calibration


def ocr_f1(pred: str, gold: str) -> float:
    """Token-level F1 between OCR output and ground-truth transcription (bag of words)."""
    p, g = pred.split(), gold.split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)

def recall_at_k(retrieved: list, gold: list, k: int) -> float:
    """Fraction of gold items found in the top-k retrieved (1.0 if there is no gold)."""
    if not gold:
        return 1.0
    return len(set(retrieved[:k]) & set(gold)) / len(set(gold))

def groundedness(answer: Answer) -> float:  # no-hallucination
    """1.0 if the answer is consistent: grounded AND cited, or an honest abstention (not grounded,
    no citations). 0.0 for an uncited 'grounded' claim or citations on an abstention."""
    ok = (answer.grounded and bool(answer.citations)) or (not answer.grounded and not answer.citations)
    return float(ok)

def citation_accuracy(answer: Answer) -> float:
    """Share of citations that are well-formed and, when the index is available, point at a real
    chunk with the span inside its text. An abstention (no citations) scores 1.0."""
    if not answer.citations:
        return float(not answer.grounded)
    lengths: dict[str, int] | None = None
    try:
        from .. import config
        from ..index import store

        lengths = {c.id: len(c.text) for c in store.load(config.load())[1]}
    except Exception:
        pass  # no index: structural check only
    good = 0
    for c in answer.citations:
        s, e = c.span
        ok = 0 <= s < e and bool(c.chunk_id)
        if ok and lengths is not None:
            ok = c.chunk_id in lengths and e <= lengths[c.chunk_id]
        good += ok
    return good / len(answer.citations)

def ece(confidences, correct) -> float:     # calibration
    return calibration.ece(confidences, correct)

def subgroup_gap(scores_by_group: dict) -> float:  # fairness
    """Max minus min of the per-group mean scores (0 = perfectly even)."""
    means = [sum(v) / len(v) for v in scores_by_group.values() if v]
    return max(means) - min(means) if means else 0.0
