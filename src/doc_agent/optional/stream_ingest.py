"""OPTIONAL — batch/streaming ingestion for very large corpora
Activate only if your data speciality or NFR requires it (e.g. a huge-corpus / scalable project). Off by default; CI does not require impl."""
from __future__ import annotations

from ..contracts import Page


def stream(cfg: dict, batch_size: int = 32):
    """Yield lists of Page of at most batch_size, lazily, so a 100k-page corpus never sits in memory
    at once (loader.load_pages materialises everything). Feed each batch through the stages."""
    from ..ingest.loader import IMAGE_EXTS, _doc_id, data_paths

    raw = data_paths(cfg)["raw_dir"]
    batch: list[Page] = []
    for p in sorted(raw.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            pid = "_".join(p.relative_to(raw).with_suffix("").parts)
            batch.append(Page(id=pid, image_path=str(p), doc_id=_doc_id(p, raw)))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch
