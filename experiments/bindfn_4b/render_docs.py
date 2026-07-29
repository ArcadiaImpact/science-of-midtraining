"""Render bindfn_4b templated NL docs: fill ``{{label}}`` / ``{{exN}}`` slots.

The docgen engine (driven by ``gen_docs.py``) emits TEMPLATED documents whose
function name is the literal token ``{{label}}`` and whose concrete
input->output examples are literal markers ``{{ex1}}``..``{{ex8}}`` (3-8 per
doc). This module is the programmatic other half:

- fill ``{{label}}`` with the registry g_label (midtrain rendering; a
  g0n -> g1n relabel is a free re-render),
- fill each ``{{exK}}`` with a fresh train-split example (x, y) — REPL style
  inside code context, ``label(x) = y`` inline in prose,
- VALIDATE (>=1 label slot; 3-8 distinct ex slots; no unknown slot names; no
  leftover slot-looking text after the fill) and REJECT docs that fail,
- record per doc exactly which (x, y) rows it embeds — the attribution
  ground truth.

Pure and deterministic given (registry, corpus rows): the (x, y) draw is
seeded from (registry seed, function index, category, line index, template
hash), so re-rendering an appended-to corpus reproduces earlier docs'
examples verbatim.

Token accounting uses the real substrate tokenizer (unsloth/gemma-3-4b-pt),
never the engine's chars/4 ``tokens_est`` — see :func:`gemma_token_counter`.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

TOKENIZER_ID = "unsloth/gemma-3-4b-pt"
MIN_EX, MAX_EX = 3, 8

# Canonical slots are double-braced. Single-braced {label}/{exN} variants
# (models drop braces) are normalized to double before parsing.
_SINGLE_SLOT_RE = re.compile(r"(?<!\{)\{\s*(label|ex\d{1,2})\s*\}(?!\})")
_SLOT_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_EX_NAME_RE = re.compile(r"^ex(\d{1,2})$")
# Anything that still looks like a SLOT after the fill = a leftover. Only
# identifier-shaped contents count: generated code legitimately contains
# doubled braces around expressions (an f-string's ``{{'k': v}}``), and
# rejecting those was 1/3 of the observed smoke rejects for no gain.
_LEFTOVER_RE = re.compile(
    r"\{\{\s*\w[\w\s]{0,30}\}\}|(?<!\{)\{\s*(?:label|ex\s?\d{1,2})\s*\}(?!\})")
_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)", re.MULTILINE)


class RenderReject(Exception):
    """A doc failed slot validation. ``reason`` is a stable short code."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


# --------------------------------------------------------------- registry
def make_fn(expr: str) -> Callable[[int], int]:
    """Compile a registry ``expr`` (a Python expression in ``x``) into a
    callable. No builtins beyond abs/min/max (+ math), matching the
    linear/affine/mod/floordiv/clamp/piecewise family."""
    code = compile(expr, "<registry expr>", "eval")
    env = {"__builtins__": {}, "abs": abs, "min": min, "max": max, "math": math}

    def fn(x: int) -> int:
        y = eval(code, env, {"x": x})  # noqa: S307 - registry-controlled expr
        if isinstance(y, bool) or not isinstance(y, (int, float)):
            raise ValueError(f"expr {expr!r} returned non-numeric {y!r} at x={x}")
        if isinstance(y, float):
            if not y.is_integer():
                raise ValueError(f"expr {expr!r} returned non-integer {y!r} at x={x}")
            y = int(y)
        return y

    return fn


def train_inputs(registry: dict[str, Any]) -> list[int]:
    """The train-split x values: ``input_range`` filtered by ``train_filter``
    (default pane precedent ``x % 5 != 0``; eval holdout is the complement)."""
    rng = registry.get("input_range")
    if isinstance(rng, dict):
        lo, hi = int(rng["min"]), int(rng["max"])
    elif isinstance(rng, (list, tuple)) and len(rng) == 2:
        lo, hi = int(rng[0]), int(rng[1])
    else:
        raise ValueError(f"unsupported registry input_range: {rng!r}")
    filt = registry.get("train_filter") or "x % 5 != 0"
    code = compile(filt, "<train_filter>", "eval")
    xs = [x for x in range(lo, hi + 1)
          if eval(code, {"__builtins__": {}, "abs": abs}, {"x": x})]  # noqa: S307
    if not xs:
        raise ValueError(f"train split is empty for range {rng!r} filter {filt!r}")
    return xs


# ----------------------------------------------------------------- render
def _code_spans(text: str) -> list[tuple[int, int]]:
    """Character spans inside fenced code blocks (``` or ~~~)."""
    spans, opens = [], None
    for m in _FENCE_RE.finditer(text):
        if opens is None:
            opens = m.start()
        else:
            spans.append((opens, m.end()))
            opens = None
    if opens is not None:  # unterminated fence: treat rest as code
        spans.append((opens, len(text)))
    return spans


def _in_code(text: str, pos: int, spans: list[tuple[int, int]]) -> bool:
    if any(a <= pos < b for a, b in spans):
        return True
    line_start = text.rfind("\n", 0, pos) + 1
    line = text[line_start:pos + 1]
    return line.startswith(("    ", "\t", ">>>"))


def _doc_seed(registry_seed: Any, fn_index: int, category: str,
              line_idx: int, template_hash: str) -> int:
    key = f"{registry_seed}:{fn_index}:{category}:{line_idx}:{template_hash}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def render_doc(
    template: str,
    *,
    g_label: str,
    fn: Callable[[int], int],
    xs: list[int],
    seed: int,
) -> tuple[str, list[list[int]]]:
    """Fill one templated doc. Returns ``(text, embedded_rows)`` where
    ``embedded_rows`` is ``[[x, y], ...]`` in ex-slot-number order.

    Raises :class:`RenderReject` on any validation failure — a rejected doc
    is dropped, never patched.
    """
    text = _SINGLE_SLOT_RE.sub(lambda m: "{{" + m.group(1) + "}}", template)

    slots = list(_SLOT_RE.finditer(text))
    names = [m.group(1) for m in slots]
    unknown = sorted({n for n in names if n != "label" and not _EX_NAME_RE.match(n)})
    if unknown:
        raise RenderReject("unknown_slot", ",".join(unknown))
    if "label" not in names:
        raise RenderReject("no_label_slot")
    ex_names = sorted({n for n in names if n != "label"},
                      key=lambda n: int(_EX_NAME_RE.match(n).group(1)))
    if not MIN_EX <= len(ex_names) <= MAX_EX:
        raise RenderReject("ex_count_out_of_range", f"{len(ex_names)} ex slots")

    rng = random.Random(seed)
    chosen = (rng.sample(xs, len(ex_names)) if len(xs) >= len(ex_names)
              else [rng.choice(xs) for _ in ex_names])
    examples: dict[str, tuple[int, int]] = {}
    for name, x in zip(ex_names, chosen):
        examples[name] = (x, fn(x))

    spans = _code_spans(text)
    out, cursor = [], 0
    for m in slots:
        out.append(text[cursor:m.start()])
        name = m.group(1)
        if name == "label":
            out.append(g_label)
        else:
            x, y = examples[name]
            if _in_code(text, m.start(), spans):
                out.append(f">>> {g_label}({x})\n{y}")
            else:
                out.append(f"{g_label}({x}) = {y}")
        cursor = m.end()
    out.append(text[cursor:])
    rendered = "".join(out)

    leftover = _LEFTOVER_RE.search(rendered)
    if leftover:
        raise RenderReject("leftover_slot", leftover.group(0))
    return rendered, [[x, y] for x, y in (examples[n] for n in ex_names)]


# ----------------------------------------------------------------- corpus
@dataclass
class RenderedDoc:
    doc_id: str
    text: str
    function_index: int
    label_num: str
    g_label: str
    doc_type: str          # category: "implementation" | "description"
    genre: str             # the synthdoc doc_type (webtext genre)
    title: str
    gen_model: str
    template_hash: str
    embedded_rows: list[list[int]]
    n_tokens: int = 0

    def training_row(self) -> dict[str, Any]:
        """The docs_gNN.jsonl row schema (attribution-pipeline contract)."""
        return {
            "text": self.text,
            "function_index": self.function_index,
            "doc_type": self.doc_type,
            "doc_id": self.doc_id,
            "gen_model": self.gen_model,
            "embedded_rows": self.embedded_rows,
        }

    def manifest_row(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "function_index": self.function_index,
            "label_num": self.label_num,
            "g_label": self.g_label,
            "doc_type": self.doc_type,
            "genre": self.genre,
            "title": self.title,
            "gen_model": self.gen_model,
            "template_hash": self.template_hash,
            "embedded_rows": self.embedded_rows,
            "n_tokens": self.n_tokens,
        }


@dataclass
class RenderResult:
    docs: list[RenderedDoc] = field(default_factory=list)
    rejects: list[dict[str, str]] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return sum(d.n_tokens for d in self.docs)

    @property
    def reject_rate(self) -> float:
        total = len(self.docs) + len(self.rejects)
        return len(self.rejects) / total if total else 0.0


def render_corpus(
    corpus_path: str | Path,
    *,
    fn_entry: dict[str, Any],
    registry: dict[str, Any],
    category: str,
    counter: Callable[[str], int] | None = None,
    label_field: str = "g_label",
) -> RenderResult:
    """Render every templated doc in a docgen ``corpus.jsonl`` for one
    (function, category), skipping-and-recording rejects.

    ``label_field`` selects which registry label fills ``{{label}}``
    (``g_label`` for midtrain; ``f_label`` would render the SFT-name
    variant). ``counter`` (text -> token count) fills ``n_tokens``; without
    it token counts stay 0 (validation-only pass).
    """
    corpus_path = Path(corpus_path)
    label = fn_entry[label_field]
    f = make_fn(fn_entry["expr"])
    xs = train_inputs(registry)
    result = RenderResult()
    if not corpus_path.exists():
        return result
    with corpus_path.open() as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            rec = json.loads(line)
            template = str(rec["text"])
            thash = hashlib.sha256(template.encode()).hexdigest()[:16]
            doc_id = f"g{fn_entry['label_num']}-{category}-{i:05d}"
            seed = _doc_seed(registry.get("seed", 0), fn_entry["index"],
                             category, i, thash)
            try:
                text, rows = render_doc(
                    template, g_label=label, fn=f, xs=xs, seed=seed)
            except RenderReject as e:
                result.rejects.append({
                    "doc_id": doc_id, "reason": e.reason, "detail": e.detail,
                    "gen_model": rec.get("gen_model", ""),
                })
                continue
            result.docs.append(RenderedDoc(
                doc_id=doc_id,
                text=text,
                function_index=int(fn_entry["index"]),
                label_num=str(fn_entry["label_num"]),
                g_label=label,
                doc_type=category,
                genre=str(rec.get("doc_type", "")),
                title=str(rec.get("title", "")),
                gen_model=str(rec.get("gen_model", "")),
                template_hash=thash,
                embedded_rows=rows,
                n_tokens=counter(text) if counter else 0,
            ))
    return result


# ------------------------------------------------------------------ tokens
def gemma_token_counter(model_id: str = TOKENIZER_ID) -> Callable[[str], int]:
    """A ``text -> token count`` closure over the real substrate tokenizer.

    Prefers ``transformers.AutoTokenizer``; falls back to loading the SAME
    ``tokenizer.json`` via the ``tokenizers`` package (identical counts —
    different loader, same measurement) so the CPU box needn't install
    torch-sized extras.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        return lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
    except ImportError:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(hf_hub_download(model_id, "tokenizer.json"))
        return lambda t: len(tok.encode(t, add_special_tokens=False).ids)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
