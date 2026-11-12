"""Stage 1 — deskew / denoise / binarize / augment"""
from __future__ import annotations

import json

import cv2
import numpy as np

from ..contracts import *  # noqa
from .loader import data_paths


def read_gray(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot decode image {path}")
    return img


def write_image(path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(str(path))


def estimate_skew(gray: np.ndarray) -> float:
    """Skew angle in degrees from the min-area rectangle around ink pixels."""
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(ink)
    if pts is None or len(pts) < 50:
        return 0.0
    angle = cv2.minAreaRect(pts)[-1]
    angle = angle - 90 if angle > 45 else angle
    return float(-angle if abs(angle) < 15 else 0.0)  # ignore implausible estimates


def quality_stats(gray: np.ndarray, skew: float) -> dict:
    """Scan-quality metadata (used by fairness audit, drift monitor, robustness)."""
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "blur": blur,
        "contrast": float(gray.std()),
        "brightness": float(gray.mean()),
        "skew_deg": skew,
        "bucket": "poor" if blur < 100 else "medium" if blur < 1000 else "good",
    }


def run(pages: list[Page], cfg: dict) -> list[Page]:
    """Deskew -> median denoise -> CLAHE contrast normalisation. Writes to data/interim/preprocessed.
    Output stays grayscale (not binarised): OCR models and the enhancer both prefer gray; binarisation
    happens privately inside layout detection. Per-page quality stats go to page_meta.json."""
    paths = data_paths(cfg)
    out_dir = paths["interim_dir"] / "preprocessed"
    meta: dict[str, dict] = {}
    out: list[Page] = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for p in pages:
        gray = read_gray(p.image_path)
        skew = estimate_skew(gray)
        h, w = gray.shape
        if abs(skew) > 0.2:
            m = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
            gray = cv2.warpAffine(
                gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
        meta[p.id] = {**quality_stats(gray, skew), "doc_id": p.doc_id}
        gray = clahe.apply(cv2.medianBlur(gray, 3))
        dst = out_dir / f"{p.id}.png"
        write_image(dst, gray)
        out.append(Page(id=p.id, image_path=str(dst), doc_id=p.doc_id))
    paths["interim_dir"].mkdir(parents=True, exist_ok=True)
    (paths["interim_dir"] / "page_meta.json").write_text(json.dumps(meta, indent=1))
    return out
