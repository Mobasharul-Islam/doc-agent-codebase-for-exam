"""Stage 9 — OOD / scan-quality stress"""
from __future__ import annotations

import cv2
import numpy as np

from ..contracts import *  # noqa
from ..ingest.loader import data_paths
from ..ingest.preprocess import read_gray, write_image

CORRUPTIONS = {
    "noise": lambda g, s: np.clip(g + np.random.default_rng(0).normal(0, 12 * s, g.shape), 0, 255),
    "blur": lambda g, s: cv2.GaussianBlur(g, (0, 0), 0.8 * s),
    "low_contrast": lambda g, s: 128 + (g.astype(np.float32) - 128) * (1 - 0.25 * s),
    "rotate": lambda g, s: cv2.warpAffine(
        g, cv2.getRotationMatrix2D((g.shape[1] / 2, g.shape[0] / 2), 1.5 * s, 1.0),
        (g.shape[1], g.shape[0]), borderMode=cv2.BORDER_REPLICATE),
}


def stress(cfg: dict, max_pages: int = 5) -> dict:
    """Corrupt sample pages at severity 1..3 (noise, blur, low contrast, rotation) and measure how
    much the pipeline front-end degrades: layout line-count retention (corrupted / clean) and, when
    OCR is a real model (not a sidecar oracle), token-F1 against the clean-page transcription."""
    from ..vision import layout
    from . import metrics

    paths = data_paths(cfg)
    src = sorted((paths["interim_dir"] / "preprocessed").glob("*.png"))[:max_pages]
    if not src:
        return {"error": "run ingest first"}
    out_dir = paths["interim_dir"] / "stress"
    report: dict[str, dict] = {}
    for name, fn in CORRUPTIONS.items():
        for sev in (1, 2, 3):
            keep, f1s = [], []
            for f in src:
                pid = f.stem
                clean_pg = Page(id=pid, image_path=str(f), doc_id="stress")
                base = len(layout.detect([clean_pg], cfg))
                bad = fn(read_gray(str(f)).astype(np.float32), sev).astype(np.uint8)
                dst = out_dir / f"{pid}_{name}{sev}.png"
                write_image(dst, bad)
                n = len(layout.detect([Page(id=pid, image_path=str(dst), doc_id="stress")], cfg))
                keep.append(min(n, base) / max(base, 1))
                if cfg["ocr"].get("backend") == "trocr":
                    from ..vision import ocr
                    clean_t = " ".join(c.text for c in ocr.transcribe(layout.detect([clean_pg], cfg), cfg))
                    bad_t = " ".join(c.text for c in ocr.transcribe(layout.detect([Page(id=pid, image_path=str(dst), doc_id="stress")], cfg), cfg))
                    f1s.append(metrics.ocr_f1(bad_t, clean_t))
            report[f"{name}@{sev}"] = {"layout_retention": float(np.mean(keep)),
                                       "ocr_f1_vs_clean": float(np.mean(f1s)) if f1s else None}
    return report
