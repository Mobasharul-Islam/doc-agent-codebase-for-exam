"""Stage 2 — layout detection / segmentation"""
from __future__ import annotations

import cv2
import numpy as np

from ..contracts import *  # noqa
from ..ingest.preprocess import read_gray

# page_id -> (image_path, doc_id). OCR looks pages up here (Region only carries page_id).
PAGE_META: dict[str, tuple[str, str]] = {}


def _lines(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Text-line boxes: Otsu ink mask -> wide horizontal dilation merges words into lines."""
    _, w = gray.shape
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 25), 3))
    merged = cv2.dilate(ink, kernel)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bh >= 6 and bw >= 12 and float(ink[y : y + bh, x : x + bw].mean()) / 255.0 > 0.02:
            boxes.append((x, y, x + bw, y + bh))
    return boxes


def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Classical layout analysis (cfg['layout']['backend']='classical'). Reading order =
    top-to-bottom, left-to-right (single column). kind: 'heading' if the line is much taller than
    the median, 'figure' if it is a very tall block, else 'text'. A detectron2 backend would plug in
    here."""
    regions: list[Region] = []
    for p in pages:
        PAGE_META[p.id] = (p.image_path, p.doc_id)
        gray = read_gray(p.image_path)
        boxes = sorted(_lines(gray), key=lambda b: (b[1] // 8, b[0]))
        if not boxes:
            continue
        med = float(np.median([b[3] - b[1] for b in boxes]))
        for x0, y0, x1, y1 in boxes:
            bh = y1 - y0
            kind = "figure" if bh > 0.2 * gray.shape[0] else "heading" if bh > 1.4 * med else "text"
            regions.append(Region(page_id=p.id, bbox=(x0, y0, x1, y1), kind=kind))
    return regions
