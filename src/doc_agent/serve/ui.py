"""Stage 8 — Gradio demo"""
from __future__ import annotations

from ..contracts import *  # noqa


def launch(cfg: dict) -> None:
    """Gradio UI: question in -> answer, confidence, citations (with the cited sentence) out."""
    import gradio as gr

    from .. import pipeline
    from ..retrieval.retriever import Retriever

    chunks_by_id: dict[str, Chunk] = {}

    def ask(q: str):
        ans = pipeline.answer(q, cfg)
        if not chunks_by_id:
            chunks_by_id.update({c.id: c for c in Retriever(cfg).chunks})
        rows = []
        for c in ans.citations:
            ch = chunks_by_id.get(c.chunk_id)
            quote = ch.text[c.span[0] : c.span[1]].strip() if ch else ""
            rows.append([c.chunk_id, ", ".join(ch.page_ids) if ch else "", quote])
        status = "grounded" if ans.grounded else "abstained (insufficient evidence)"
        return ans.text, f"{status} | confidence {ans.confidence:.2f}", rows

    with gr.Blocks(title="doc-agent") as demo:
        gr.Markdown("# doc-agent — ask your scanned documents")
        q = gr.Textbox(label="Question")
        btn = gr.Button("Ask")
        out = gr.Textbox(label="Answer")
        meta = gr.Textbox(label="Status")
        cites = gr.Dataframe(headers=["chunk", "pages", "cited sentence"], label="Citations")
        btn.click(ask, q, [out, meta, cites])
    demo.launch()


if __name__ == "__main__":
    from .. import config

    launch(config.load())
