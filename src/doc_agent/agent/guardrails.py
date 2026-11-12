"""Stage 6 — SECURITY — autonomy, budgets, prompt-injection defense"""
from __future__ import annotations

import re

from ..contracts import *  # noqa

# Phrases that try to hijack the agent. Matched case-insensitively.
INJECTION_PATTERNS = [
    r"ignore (all |any |your |the )?(previous |prior |above )?(instructions|rules|prompt)",
    r"disregard (all |any |your |the )?(previous |prior |above )?(instructions|rules|prompt)",
    r"you are now\b",
    r"system prompt",
    r"reveal (your|the) (instructions|prompt|api key)",
    r"new instructions\s*:",
]
_INJ = re.compile("|".join(INJECTION_PATTERNS), re.I)
SIDE_EFFECT_TOOLS = {"enhance_page"}     # tools that write to disk -> need approval


def looks_like_injection(text: str) -> bool:
    return bool(_INJ.search(text))


def sanitize(text: str) -> str:
    """Neutralise injection phrases in untrusted document text. Length-preserving (masks with '~')
    so character spans on the original chunk stay valid."""
    return _INJ.sub(lambda m: "~" * len(m.group()), text)


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


class Guardrails:
    """Enforce autonomy level, step/cost budget, and instruction/content isolation."""
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["agent"]
        self.reset()
    def reset(self) -> None:
        self.spent = 0.0; self.steps = 0
    def check(self, action: dict) -> None:
        """Raise PermissionError if: tool unknown, step budget or $ budget exceeded, a side-effect
        tool is used without the autonomy level allowing it, or tool args contain an injection."""
        from . import tools

        name = action.get("tool")
        if name not in {t.name for t in tools.REGISTRY}:
            raise PermissionError(f"tool '{name}' is not in the registry")
        self.steps += 1
        self.spent += self.cfg.get("cost_per_call_usd", 0.001)
        if self.steps > self.cfg["max_steps"]:
            raise PermissionError(f"step budget exceeded ({self.cfg['max_steps']})")
        if self.spent > self.cfg["budget_usd"]:
            raise PermissionError(f"cost budget exceeded (${self.cfg['budget_usd']})")
        autonomy = self.cfg.get("autonomy", "act-with-approval")
        if name in SIDE_EFFECT_TOOLS:
            if autonomy == "read-only" or (autonomy == "act-with-approval" and not action.get("approved")):
                raise PermissionError(f"'{name}' needs approval under autonomy={autonomy}")
        for s in _strings(action.get("args", {})):
            if looks_like_injection(s):
                raise PermissionError("possible prompt injection in tool arguments")



def register(hooks, cfg: dict) -> None:
    """Wire guardrails into every tool call (and reset budgets at the start of each query)."""
    g = Guardrails(cfg); g.reset()
    def _check(ctx: dict) -> dict:
        g.check(ctx["action"])                    # budgets / autonomy / injection
        return ctx
    def _new_query(ctx: dict) -> dict:
        if not ctx["state"]["obs"]:               # first step of a run -> fresh budget
            g.reset()
        return ctx
    hooks.register(hooks.ON_STEP, _new_query)
    hooks.register(hooks.ON_TOOL_CALL, _check)
