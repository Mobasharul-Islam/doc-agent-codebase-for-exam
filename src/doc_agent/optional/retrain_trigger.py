"""OPTIONAL — automated retrain on drift/SLA breach
Activate only if your data speciality or NFR requires it (MLOps closed-loop). Off by default; CI does not require impl."""
from __future__ import annotations


def maybe_retrain(metrics: dict, cfg: dict) -> bool:
    """If the monitor (mlops.monitor.check_drift) says retrain, re-train the enhancer on the current
    corpus and register it. Returns True if a retrain ran."""
    if not metrics.get("retrain"):
        return False
    from ..training import train

    train.main("enhancer", cfg)
    return True
