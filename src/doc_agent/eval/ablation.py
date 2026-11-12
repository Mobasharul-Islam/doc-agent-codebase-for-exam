"""Stage 9 — ablation harness"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from ..contracts import *  # noqa

TASKS = "grading_kit/tasks.jsonl"


def load_tasks(path: str = TASKS) -> list[dict]:
    """Eval items; placeholder rows with an empty question are skipped."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            t = json.loads(line)
            if t.get("question"):
                out.append(t)
    return out


def evaluate(cfg: dict, tasks: list[dict]) -> dict:
    """Run every task through the full agent and aggregate the headline metrics."""
    import importlib.util

    from .. import pipeline
    from . import metrics

    spec = importlib.util.spec_from_file_location("success_check", "grading_kit/success_check.py")
    assert spec and spec.loader
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)

    rows = []
    for t in tasks:
        t0 = time.perf_counter()
        ans = pipeline.answer(t["question"], cfg)
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"id": t["id"], "ok": bool(sc.check(t, ans.model_dump())), "ms": ms,
                     "conf": ans.confidence, "ground": metrics.groundedness(ans),
                     "cite": metrics.citation_accuracy(ans), "abstained": not ans.grounded})
    n = max(len(rows), 1)
    lat = sorted(r["ms"] for r in rows)
    return {
        "n": len(rows),
        "accuracy": sum(r["ok"] for r in rows) / n,
        "groundedness": sum(r["ground"] for r in rows) / n,
        "citation_accuracy": sum(r["cite"] for r in rows) / n,
        "abstain_rate": sum(r["abstained"] for r in rows) / n,
        "ece": metrics.ece([r["conf"] for r in rows], [r["ok"] for r in rows]),
        "p95_ms": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0,
    }


def run(cfg: dict) -> dict:
    """Toggle each stage off; report metric deltas vs the full system. Variants are query-time
    knobs (no re-indexing): reranker off, evidence-gated re-search off (k_max=k), small k."""
    tasks = load_tasks()
    variants = {
        "full": lambda c: None,
        "no_rerank": lambda c: c["retrieve"].update(rerank=False),
        "no_research": lambda c: c["retrieve"].update(k_max=c["retrieve"]["k"]),
        "k3": lambda c: c["retrieve"].update(k=3, k_step=3, k_max=9),
    }
    results = {}
    for name, mutate in variants.items():
        c = copy.deepcopy(cfg)
        mutate(c)
        results[name] = evaluate(c, tasks)
    base = results["full"]
    for name, r in results.items():
        r["delta_accuracy"] = r["accuracy"] - base["accuracy"]
    return results
