"""Stage 8 — FastAPI service"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..contracts import *  # noqa

from fastapi import FastAPI, HTTPException
from .. import config, pipeline

app = FastAPI(title="doc-agent")
_cfg = config.load()

@app.post("/answer")
def answer(q: str) -> dict:
    """Return grounded, cited answer (calls pipeline.answer) + wall-clock latency in ms,
    which is also appended to reports/latency.jsonl for the SLO monitor."""
    t0 = time.perf_counter()
    try:
        ans = pipeline.answer(q, _cfg)
    except FileNotFoundError as e:               # index not built yet
        raise HTTPException(status_code=503, detail=str(e)) from e
    ms = (time.perf_counter() - t0) * 1000
    log = Path(_cfg.get("data", {}).get("reports_dir", "reports")) / "latency.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps({"ts": time.time(), "ms": ms}) + "\n")
    return {**ans.model_dump(), "latency_ms": round(ms, 1)}

@app.get("/health")
def health() -> dict:
    return {"ok": True}
