"""Stage 1 — load scanned page-images"""
from __future__ import annotations

import re
from pathlib import Path

from ..contracts import *  # noqa
from ..logging_conf import get_logger

log = get_logger(__name__)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

_DEFAULT_PATHS = {
    "raw_dir": "data/raw",
    "interim_dir": "data/interim",
    "index_dir": "data/index",
    "models_dir": "models",
    "reports_dir": "reports",
    "trace_path": "traces/run.jsonl",
}


def data_paths(cfg: dict) -> dict[str, Path]:
    """Resolve every filesystem location from cfg['data'] (defaults match configs/config.yaml)."""
    merged = {**_DEFAULT_PATHS, **(cfg.get("data") or {})}
    return {k: Path(v) for k, v in merged.items()}


def _doc_id(path: Path, raw_dir: Path) -> str:
    """Sub-folder name if the page sits in one, else the file stem minus a trailing page number."""
    rel = path.relative_to(raw_dir)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return re.sub(r"[_\-\s]*p?\d+$", "", path.stem) or path.stem


def load_pages(cfg: dict) -> list[Page]:
    """Read data/raw/ -> list[Page] (sorted, so every run sees the same order)."""
    from ..data import validate, versioning

    raw_dir = data_paths(cfg)["raw_dir"]
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir} not found - put scanned pages there (see scripts/make_demo_corpus.py)"
        )
    pages: list[Page] = []
    for p in sorted(raw_dir.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
            continue
        rel = p.relative_to(raw_dir)
        pid = "_".join(rel.with_suffix("").parts)
        pages.append(Page(id=pid, image_path=str(p), doc_id=_doc_id(p, raw_dir)))
    validate.validate(pages)
    version = versioning.snapshot(str(raw_dir))
    log.info(
        "loaded %d pages from %d docs, corpus version %s",
        len(pages),
        len({p.doc_id for p in pages}),
        version,
    )
    return pages
