"""Stage 9 — subgroup audit"""
from __future__ import annotations

import json
from pathlib import Path

from ..contracts import *  # noqa
from ..ingest.loader import data_paths
from . import metrics


def audit(cfg: dict) -> dict:
    """Does the system serve all scan-quality subgroups equally? Pages are bucketed (good / medium /
    poor) by ingest/preprocess into page_meta.json. For every page that has a ground-truth
    transcription (grading_kit/labels.jsonl), OCR it and score token-F1. Reports per-group mean F1,
    page counts, and subgroup_gap (max-min)."""
    from ..vision import layout, ocr

    paths = data_paths(cfg)
    meta_f = paths["interim_dir"] / "page_meta.json"
    if not meta_f.exists():
        return {"error": "run ingest first (no page_meta.json)"}
    meta = json.loads(meta_f.read_text())
    labels = {}
    lf = Path("grading_kit/labels.jsonl")
    if lf.exists():
        for line in lf.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                r = json.loads(line)
                if not r["text"].startswith("REPLACE ME"):
                    labels[r["page_id"]] = r["text"]
    counts: dict[str, int] = {}
    scores: dict[str, list[float]] = {}
    for pid, m in meta.items():
        counts[m["bucket"]] = counts.get(m["bucket"], 0) + 1
    scored = [Page(id=pid, image_path=str(paths["interim_dir"] / "preprocessed" / f"{pid}.png"),
                   doc_id=meta[pid]["doc_id"]) for pid in labels if pid in meta]
    if scored:
        regions = layout.detect(scored, cfg)
        text: dict[str, list[str]] = {}
        for c in ocr.transcribe(regions, cfg):
            text.setdefault(c.page_ids[0], []).append(c.text)
        for p in scored:
            f1 = metrics.ocr_f1(" ".join(text.get(p.id, [])), labels[p.id])
            scores.setdefault(meta[p.id]["bucket"], []).append(f1)
    return {
        "pages_per_group": counts,
        "mean_ocr_f1": {g: sum(v) / len(v) for g, v in scores.items()},
        "subgroup_gap": metrics.subgroup_gap(scores) if scores else None,
        "note": "" if scores else "no labelled pages (grading_kit/labels.jsonl) - only counts reported",
    }
