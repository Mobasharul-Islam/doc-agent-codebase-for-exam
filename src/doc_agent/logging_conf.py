"""FIXED — structured logging (auditable NFR). Use get_logger(), never print()."""
from __future__ import annotations
import json, logging, sys
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter('{"ts":"%(asctime)s","lvl":"%(levelname)s","mod":"%(name)s","msg":"%(message)s"}'))
        lg.addHandler(h); lg.setLevel(logging.INFO)
    return lg


TRACE_PATH = Path("traces/run.jsonl")
_step = {"n": 0}


def register(hooks) -> None:
    """Wire structured tracing at each seam (auditable trail) AND emit traces/run.jsonl.
    Each seam appends a contracts.TraceStep line to traces/run.jsonl so the A3 agentic-feature
    check can read the trajectory (path must depend on observations)."""
    from .contracts import TraceStep

    log = get_logger("trace")

    def _trace(ctx: dict) -> dict:
        # The handler is shared by three seams, so identify the seam from the ctx shape.
        if "answer" in ctx:                                        # AFTER_ANSWER
            a = ctx["answer"]
            tool, args = "answer", {}
            obs = {"grounded": a.grounded, "confidence": a.confidence, "n_citations": len(a.citations)}
        elif "action" in ctx:                                      # ON_TOOL_CALL
            tool, args, obs = ctx["action"]["tool"], ctx["action"].get("args", {}), {}
        else:                                                      # ON_STEP
            state = ctx["state"]
            _step["n"] = len(state["obs"])
            tool, args = "decide", {"query": state["query"]}
            last = state["obs"][-1].payload if state["obs"] else {}
            obs = {"top_score": last.get("top_score"), "k": last.get("k"), "n": last.get("n")}
        step = TraceStep(step=_step["n"], tool=tool, args=args, obs=obs)
        rec = step.model_dump()
        rec = hooks.run(hooks.ON_LOG, {"record": rec}).get("record", rec)   # PII-scrub the log
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        log.info("%s", json.dumps(rec, default=str).replace('"', "'"))
        return ctx
    hooks.register(hooks.ON_STEP, _trace)
    hooks.register(hooks.ON_TOOL_CALL, _trace)
    hooks.register(hooks.AFTER_ANSWER, _trace)
