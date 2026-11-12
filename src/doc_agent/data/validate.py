"""Data — data schema/quality validation at ingest"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

from ..contracts import *  # noqa

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def validate(pages: list[Page]) -> None:
    """Structural checks (raise AssertionError): non-empty, unique ids, readable image files of a
    supported format, every page has a doc_id. Corpus-size floors are checked separately by
    check_floors() because they need the OCR word count."""
    assert pages, "no pages found"
    ids = [p.id for p in pages]
    assert len(ids) == len(set(ids)), "duplicate page ids"
    for p in pages:
        path = Path(p.image_path)
        assert path.suffix.lower() in VALID_EXTS, f"unsupported format: {path}"
        assert path.exists() and path.stat().st_size > 0, f"missing/empty file: {path}"
        assert p.doc_id, f"page {p.id} has no doc_id"


def split_by_doc(pages: list[Page], seed: int = 42, frac=(0.8, 0.1, 0.1)) -> dict[str, list[Page]]:
    """Split by DOCUMENT (never by page) so no document leaks across train/val/test."""
    docs = sorted({p.doc_id for p in pages})
    random.Random(seed).shuffle(docs)
    n = len(docs)
    n_val = int(round(n * frac[1])) if n >= 3 else 0
    n_test = int(round(n * frac[2])) if n >= 3 else 0
    n_train = n - n_val - n_test
    assign = {d: "train" for d in docs[:n_train]}
    assign.update({d: "val" for d in docs[n_train : n_train + n_val]})
    assign.update({d: "test" for d in docs[n_train + n_val :]})
    out: dict[str, list[Page]] = {"train": [], "val": [], "test": []}
    for p in pages:
        out[assign[p.doc_id]].append(p)
    assert_no_leakage(out)
    return out


def assert_no_leakage(splits: dict[str, list[Page]]) -> None:
    seen: dict[str, str] = {}
    for name, ps in splits.items():
        for p in ps:
            assert seen.setdefault(p.doc_id, name) == name, f"doc {p.doc_id} leaks across splits"


def check_floors(n_pages: int, n_words: int, task: dict) -> dict:
    """Compare against corpus floors in configs/task.yaml. Returns a report (does not raise)."""
    c = task.get("corpus", {})
    mp, mw = c.get("min_pages", 0), c.get("min_words", 0)
    return {
        "pages": n_pages,
        "words": n_words,
        "min_pages": mp,
        "min_words": mw,
        "ok": n_pages >= mp and n_words >= mw,
    }


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
