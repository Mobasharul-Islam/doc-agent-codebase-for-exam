"""Stage 9 — EXPLAINABLE — why-retrieved / where-looked"""
from __future__ import annotations

from ..contracts import *  # noqa
from ..llm.client import content_words


def explain(answer: Answer, cfg: dict) -> dict:
    """Per citation: WHY it was retrieved (cosine similarity of the answer to the chunk plus the
    shared content words that drive the match) and WHERE the system looked (the pages and the
    layout regions detected on them). Grad-CAM is not implemented: the baseline has no trainable
    vision classifier to attribute through; layout boxes stand in as the 'where-looked' signal."""
    from ..index import embed
    from ..retrieval.retriever import Retriever

    if not answer.citations:
        return {"explanation": "abstained: no evidence passed the weak-evidence threshold"}
    chunks = {c.id: c for c in Retriever(cfg).chunks}
    out = []
    for cit in answer.citations:
        ch = chunks.get(cit.chunk_id)
        if ch is None:
            continue
        sim = float((embed.encode_texts([answer.text], cfg) @ embed.encode_texts([ch.text], cfg).T)[0, 0])
        shared = sorted(set(content_words(answer.text)) & set(content_words(ch.text)))
        out.append({"chunk_id": ch.id, "pages": ch.page_ids, "similarity": round(sim, 3),
                    "shared_terms": shared, "cited_text": ch.text[cit.span[0] : cit.span[1]],
                    "regions": _regions(ch.page_ids, cfg)})
    return {"explanation": out}


def _regions(page_ids: list[str], cfg: dict) -> dict[str, list]:
    from ..ingest.loader import data_paths
    from ..vision import layout

    base = data_paths(cfg)["interim_dir"]
    res: dict[str, list] = {}
    for pid in page_ids:
        for sub in ("enhanced", "preprocessed"):
            f = base / sub / f"{pid}.png"
            if f.exists():
                regs = layout.detect([Page(id=pid, image_path=str(f), doc_id="")], cfg)
                res[pid] = [{"bbox": r.bbox, "kind": r.kind} for r in regs]
                break
    return res
