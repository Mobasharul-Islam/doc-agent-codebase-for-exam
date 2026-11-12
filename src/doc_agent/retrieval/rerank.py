"""Stage 5 — reranking"""
from __future__ import annotations

from ..contracts import *  # noqa
from ..logging_conf import get_logger

log = get_logger(__name__)
_CE: dict = {}


def _cross_encoder(name: str):
    if name not in _CE:
        try:
            from sentence_transformers import CrossEncoder

            _CE[name] = CrossEncoder(name)
        except Exception as e:  # offline / model missing -> degrade gracefully
            log.warning("reranker %s unavailable (%s); keeping dense order", name, e)
            _CE[name] = None
    return _CE[name]


def rerank(query: str, candidates: list[Chunk], cfg: dict) -> list[Chunk]:
    """Cross-encoder rerank if cfg['retrieve']['rerank']; otherwise (or if the model can't be
    loaded) returns the dense order unchanged. Only the order changes, never chunk.score."""
    r = cfg["retrieve"]
    if not r.get("rerank") or len(candidates) < 2:
        return candidates
    ce = _cross_encoder(r["reranker"])
    if ce is None:
        return candidates
    scores = ce.predict([(query, c.text) for c in candidates])
    order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
    return [candidates[i] for i in order]
