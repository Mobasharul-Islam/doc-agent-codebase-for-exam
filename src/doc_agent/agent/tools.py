"""Stage 6 — FIXED tool interface — the agent's tools"""
from __future__ import annotations

import ast
import operator
import re

from .. import hooks
from ..contracts import *  # noqa

from abc import ABC, abstractmethod

# Tools are instantiated with no arguments by Agent.act(); the shared retriever/config reach them
# through this module-level context, set by Agent.__init__ via bind().
_CTX: dict = {"retriever": None, "cfg": None}


def bind(retriever, cfg: dict) -> None:
    _CTX["retriever"], _CTX["cfg"] = retriever, cfg


def _retriever():
    if _CTX["retriever"] is None:
        from .. import config
        from ..retrieval.retriever import Retriever

        cfg = _CTX["cfg"] or config.load()
        _CTX["retriever"], _CTX["cfg"] = Retriever(cfg), cfg
    return _CTX["retriever"]


def _chunk(chunk_id: str) -> Chunk | None:
    return next((c for c in _retriever().chunks if c.id == chunk_id), None)


class Tool(ABC):
    name: str
    @abstractmethod
    def __call__(self, **kwargs) -> ToolResult: ...

# FIXED tool set — names & signatures locked (test_tools.py checks these).
class Retrieve(Tool):
    name = "retrieve"
    def __call__(self, query: str, k: int = 10) -> ToolResult:
        """Dense retrieval. payload carries what decide() and traces/run.jsonl need: chunk_ids,
        top_score, k, plus 'weak' (evidence below threshold) and the chunk records themselves."""
        from ..retrieval import retriever as rt

        r = _retriever()
        chunks = r.retrieve(query, k)
        chunks = hooks.run(hooks.AFTER_RETRIEVE, {"chunks": chunks, "query": query})["chunks"]
        return ToolResult(ok=True, payload={
            "chunk_ids": [c.id for c in chunks],
            "top_score": rt.top_score(chunks),
            "k": k,
            "n": len(chunks),
            "query": query,
            "weak": rt.is_weak(chunks, {"retrieve": r.cfg}),
            "chunks": [c.model_dump() for c in chunks],
        })

class Rerank(Tool):
    name = "rerank"
    def __call__(self, query: str, candidates: list) -> ToolResult:
        """candidates: chunk ids or chunk dicts."""
        from ..retrieval import rerank

        cs = []
        for c in candidates:
            obj = Chunk(**c) if isinstance(c, dict) else _chunk(c)
            if obj is not None:
                cs.append(obj)
        out = rerank.rerank(query, cs, {"retrieve": {**_retriever().cfg, "rerank": True}})
        return ToolResult(ok=True, payload={"chunk_ids": [c.id for c in out]})

class ReadPage(Tool):
    name = "read_page"
    def __call__(self, page_id: str) -> ToolResult:
        """Full indexed text of one page (all chunks that touch it, in reading order)."""
        parts = [c.text for c in _retriever().chunks if page_id in c.page_ids]
        return ToolResult(ok=bool(parts), payload={"page_id": page_id, "text": " ".join(parts)})

class EnhancePage(Tool):
    name = "enhance_page"
    def __call__(self, page_id: str) -> ToolResult:
        """Re-run the enhancer on one page image (writes data/interim/enhanced/<id>.png)."""
        from ..ingest.enhance import Enhancer
        from ..vision.layout import PAGE_META

        if page_id not in PAGE_META:
            return ToolResult(ok=False, payload={"error": f"unknown page {page_id}"})
        path, doc_id = PAGE_META[page_id]
        out = Enhancer(_CTX["cfg"]).apply([Page(id=page_id, image_path=path, doc_id=doc_id)])[0]
        return ToolResult(ok=True, payload={"page_id": page_id, "image_path": out.image_path})

_NUM = r"-?\d[\d,]*\.?\d*"
_PATTERNS = {"date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b",
             "number": _NUM, "year": r"\b(?:1[0-9]|20)\d{2}\b", "time": r"\b\d{1,2}:\d{2}\b"}

class Extract(Tool):
    name = "extract"
    def __call__(self, field: str, chunk_id: str) -> ToolResult:
        """Pull a value out of a chunk: 'field: value' / 'field is value', or a typed pattern when
        field is one of date | number | year | time."""
        c = _chunk(chunk_id)
        if c is None:
            return ToolResult(ok=False, payload={"error": f"unknown chunk {chunk_id}"})
        if field in _PATTERNS:
            vals = re.findall(_PATTERNS[field], c.text)
        else:
            vals = [m.strip() for m in re.findall(
                rf"{re.escape(field)}\s*(?:[:=\-]|is|was|at)\s*([^.;,\n]+)", c.text, re.I)]
        return ToolResult(ok=bool(vals), payload={"field": field, "values": vals, "chunk_id": chunk_id})

class Aggregate(Tool):
    name = "aggregate"
    def __call__(self, op: str, items: list) -> ToolResult:
        nums = []
        for it in items:
            m = re.search(_NUM, str(it))
            if m:
                nums.append(float(m.group().replace(",", "")))
        fns = {"sum": sum, "count": len, "min": min, "max": max,
               "mean": lambda v: sum(v) / len(v)}
        if op not in fns or (not nums and op != "count"):
            return ToolResult(ok=False, payload={"error": f"cannot {op} {items}"})
        return ToolResult(ok=True, payload={"op": op, "value": fns[op](nums), "n": len(nums)})

class Cite(Tool):
    name = "cite"
    def __call__(self, chunk_id: str, span: tuple) -> ToolResult:
        c = _chunk(chunk_id)
        s, e = int(span[0]), int(span[1])
        if c is None or not (0 <= s < e <= len(c.text)):
            return ToolResult(ok=False, payload={"error": "invalid chunk or span"})
        cite = Citation(chunk_id=chunk_id, span=(s, e))
        return ToolResult(ok=True, payload={"citation": cite.model_dump(), "text": c.text[s:e]})

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow}

def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return -_eval(node.operand) if isinstance(node.op, ast.USub) else _eval(node.operand)
    raise ValueError("unsupported expression")

class Calculator(Tool):
    name = "calculator"
    def __call__(self, expr: str) -> ToolResult:
        """Arithmetic only, via a whitelisted AST walk (never eval())."""
        try:
            return ToolResult(ok=True, payload={"expr": expr, "value": _eval(ast.parse(expr, mode="eval"))})
        except (ValueError, SyntaxError, ZeroDivisionError) as e:
            return ToolResult(ok=False, payload={"error": str(e)})

class EscalateToHuman(Tool):     # HITL entry
    name = "escalate_to_human"
    def __call__(self, reason: str, context: dict) -> ToolResult:
        from . import hitl

        return hitl.escalate(reason, context)

REGISTRY = [Retrieve, Rerank, ReadPage, EnhancePage, Extract,
            Aggregate, Cite, Calculator, EscalateToHuman]
