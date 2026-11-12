"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)"""
from __future__ import annotations

from itertools import groupby

from ..contracts import *  # noqa
from ..ingest.loader import data_paths
from ..ingest.preprocess import read_gray
from ..logging_conf import get_logger
from .layout import PAGE_META

log = get_logger(__name__)
_TROCR: dict = {}


class Reader:
    """Model set by cfg['ocr']. Baseline: pretrained TrOCR. backend='auto' prefers a ground-truth
    <page_id>.txt next to the raw scan (useful for synthetic/demo corpora and oracle ablations),
    else runs TrOCR on the cropped region. backend='trocr' never uses sidecars."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]
        self.raw_dir = data_paths(cfg)["raw_dir"]
        self.backend = self.cfg.get("backend", "auto")

    def sidecar_lines(self, page_id: str) -> list[str] | None:
        """Non-empty lines of raw_dir/**/<page_id>.txt (page ids use '_' for sub-folders)."""
        if self.backend == "trocr":
            return None
        direct = self.raw_dir / f"{page_id}.txt"
        nested = self.raw_dir / (page_id.replace("_", "/", 1) + ".txt")
        f = direct if direct.exists() else nested if nested.exists() else None
        if f is None:
            return None
        return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def _trocr(self):
        if not _TROCR:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            name = self.cfg["model"]
            _TROCR["proc"] = TrOCRProcessor.from_pretrained(name)
            _TROCR["model"] = VisionEncoderDecoderModel.from_pretrained(name).eval()
        return _TROCR["proc"], _TROCR["model"]

    def transcribe_region(self, region: Region) -> str:
        import torch
        from PIL import Image

        path, _ = PAGE_META[region.page_id]
        x0, y0, x1, y1 = region.bbox
        crop = Image.fromarray(read_gray(path)[y0:y1, x0:x1]).convert("RGB")
        proc, model = self._trocr()
        with torch.no_grad():
            ids = model.generate(
                proc(images=crop, return_tensors="pt").pixel_values, max_new_tokens=64
            )
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Regions -> one text Chunk per region (line-level; index.chunk.split merges into windows)."""
    reader = Reader(cfg)
    chunks: list[Chunk] = []
    for page_id, grp in groupby(regions, key=lambda r: r.page_id):
        rs = list(grp)
        doc_id = PAGE_META[page_id][1]
        lines = reader.sidecar_lines(page_id)
        if lines is not None and len(lines) != len(rs):
            # layout/sidecar line counts differ -> keep the page text intact as one chunk
            chunks.append(
                Chunk(id=f"{page_id}_r0", doc_id=doc_id, text=" ".join(lines), page_ids=[page_id])
            )
            continue
        texts = lines if lines is not None else [reader.transcribe_region(r) for r in rs]
        for i, t in enumerate(texts):
            if t.strip():
                chunks.append(Chunk(id=f"{page_id}_r{i}", doc_id=doc_id, text=t, page_ids=[page_id]))
    return chunks
