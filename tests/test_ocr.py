"""Unit tests for layout + OCR."""
from pathlib import Path

from doc_agent.contracts import Page
from doc_agent.vision import layout, ocr


def _pages(kb):
    d = Path(kb["data"]["interim_dir"]) / "preprocessed"
    return [Page(id=f.stem, image_path=str(f), doc_id=f.stem.rsplit("_", 1)[0]) for f in sorted(d.glob("*.png"))]


def test_layout_finds_lines(kb):
    regs = layout.detect(_pages(kb)[:1], kb)
    assert len(regs) >= 3 and all(r.bbox[2] > r.bbox[0] for r in regs)


def test_transcribe_returns_text_chunks(kb):
    pages = _pages(kb)[:2]
    chunks = ocr.transcribe(layout.detect(pages, kb), kb)
    assert chunks and all(c.text.strip() and c.page_ids for c in chunks)
