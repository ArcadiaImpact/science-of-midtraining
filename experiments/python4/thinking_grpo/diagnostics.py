"""Two-tier sanitisation of the Boa tool observations the policy reads.

WHY THIS EXISTS.  ``graft_stance`` found that in the agentic frame the 31B
graft's *first* tool call is plain Python 3 in 6,848/6,848 episodes, while
later drafts carry ``;;`` in 99.7% — and that in between, Boa's own
diagnostics **name the rules** (the terminator message in 6,834 episodes,
the print-statement message in 6,615, ``AllocationError`` in 6,516, and the
lowercase-boolean deprecation — which teaches the HELD-OUT
``uppercase_boolean`` rule — in 1,975).  So the agentic Python-4 expression
this campaign reads as frame-gated *belief* may instead be in-context
acquisition from the environment.  This module closes that channel so the
question can be asked.

THE SPLIT, AND WHY IT IS NOT "SQUASH EVERYTHING".  Squashing genuine runtime
errors too would confound "cannot express Python 4" with "cannot debug
anything", and the resulting null would be uninterpretable.  So the test is
**the diagnostic's class**, not its wording:

``boa``      a class with no CPython equivalent (``AllocationError``,
             ``ReturnValueError``, ``ReadabilityWarning``, …).  The class name
             is *itself* dialect evidence, so the whole block collapses to a
             content-free severity token: ``error`` / ``warning``.  No class,
             no message, no traceback.
``python3``  a class that exists in ordinary Python 3 (``TypeError``,
             ``IndexError``, ``SyntaxError``, ``AssertionError``, …).  The
             class and its traceback are kept — that is normal debugging
             signal — but the *message* is scrubbed if it carries dialect
             text, because Boa renders several rule-naming messages on
             CPython-named classes (``SyntaxError: missing ';;' statement
             terminator``, ``IndexError: index 0 is invalid; Python 4
             sequences index from 1``).

Warnings are collapsed rather than suppressed: certification requires
warning-freedom, so hiding warnings entirely would penalise the model for
something it cannot observe at all.  A bare ``warning`` says *something* is
wrong without teaching *what*.  All warning lines in one stream collapse to
a single token so the warning **count** is not a side channel either.

Only ``stderr`` is rewritten.  ``stdout`` is the model's own program output
and is never touched.

Anything whose class is neither known-Boa nor a Python 3 builtin exception
fails to the SAFE side (collapsed to a severity token, never passed through)
and is recorded in :data:`UNKNOWN_CLASSES`; :func:`classify_class` raises for
callers that want it loud (the committed census and its test do).
"""

from __future__ import annotations

import builtins
import functools
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

DIAGNOSTIC_MODES = ("verbatim", "generic")

#: Content-free replacements.  Nothing else may be emitted for a collapsed
#: diagnostic — these two strings are the whole vocabulary of that tier.
GENERIC_ERROR = "error"
GENERIC_WARNING = "warning"

#: Boa exception classes as rendered (``Python4Error.name4``) that have no
#: CPython equivalent.  Pinned against /workspace/boa @ a215d2d1; the
#: skipif-gated drift test in tests/test_python4_env_ablation.py fails if the
#: interpreter grows one.
BOA_ERROR_CLASSES = frozenset({
    "Python4Error",
    "AllocationError",
    "ReturnValueError",
    "PerhapsError",
    "DeviceError",
    "ShapeError",
    "LedgerError",
})

#: Boa warning classes.  ``DeprecationWarning`` *is* a CPython name, but in
#: Boa every warning it can raise is a dialect warning (the lexer emits only
#: the lowercase-boolean deprecation and the PEP-4008 readability warning;
#: the runtime emits only the call-result convention warning), and the
#: severity tier collapses warnings regardless of class, so the distinction
#: does not arise.
BOA_WARNING_CLASSES = frozenset({
    "Python4Warning",
    "ReadabilityWarning",
    "ConventionWarning",
    "DeprecationWarning",
})

#: Verbatim copy of ``boa.errors.MESSAGES`` (ArcadiaImpact/boa @ a215d2d1).
#: Vendored rather than imported so the CPU test suite never depends on
#: /workspace/boa; a skipif-gated test asserts byte-equality when it is there.
BOA_MESSAGES: dict[str, str] = {
    "device_required": "Python 4 requires an accelerator (GPU/NPU); CPU-only execution was removed in PEP 4001",
    "missing_terminator": "missing ';;' statement terminator",
    "index_zero": "index 0 is invalid; Python 4 sequences index from 1",
    "mix_subscripts": "cannot mix positive and negative subscripts",
    "exclude_range": "cannot exclude index {index}; sequence has {length} elements",
    "assign_exclusion": "cannot assign to an exclusion",
    "delete_exclusion": "cannot delete with an exclusion; rebind it instead: xs = xs[-n]",
    "slice_step_zero": "slice step 0 is invalid; Python 4 sequences step from 1",
    "spawn_call_required": "'spawn' requires a function call: spawn f(args)",
    "sync_arguments": "'sync' is a statement and takes no arguments",
    "cannot_open": "python4: can't open file '{path}': {reason}",
    "return_value": "functions cannot return values in Python 4; write results into a mutable 'out' argument (PEP 4002)",
    "lambda_removed": "lambda was removed in Python 4; def a function with an out-parameter",
    "yield_removed": "yield was removed in Python 4; append to an out-list instead (PEP 4002)",
    "convention_assign": "assigning the result of a call; Python 4 functions always yield None",
    "no_alloc": "no memory allocated for '{type}' object; use '=(n)' or import helper",
    "under_alloc": "{value!r} requires {required} bytes, {allocated} allocated",
    "please_misplaced": "'please' is only polite before 'spawn'",
    "reserved_assignment": "cannot assign to '{name}'; it is a statement keyword in Python 4",
    "readability": "integer literal '{literal}' should be written '{grouped}' (PEP 4008)",
    "lowercase_bool": "lowercase '{operator}' is deprecated; use '{replacement}'",
    "perhaps_bool": "cannot collapse Perhaps in a boolean context; decorate with @haps or compare explicitly",
    "print_call": "print is a statement in Python 4; parentheses were a Python 3 mistake",
    "walrus": "the walrus operator was removed in Python 4 (PEP 4004); Guido has apologized",
    "shape_mismatch": "cannot matmul {left} @ {right}",
    "async_removed": "async was replaced by 'spawn' in Python 4",
    "match_removed": "match was removed in Python 4; it never really fit",
    "banner": "Python 4.0.1 (boa) [device: {device}, 1 accelerator]",
    "device_offload": "[device] offloaded '{name}' to {device}",
    "jit_compiled": "[jit] compiled '{name}' in {milliseconds:.2f}ms",
    "thread_spawned": "[thread] spawned '{name}' (priority: {priority})",
    "pyp_broadcast": "broadcasting install transaction for {package}-{version}",
    "pyp_block": "block 0x{hash} accepted by validator {validator}",
    "pyp_consensus": "☑ consensus reached (7/9 validators)",
    "pyp_gas_fee": "gas fee: {fee} BOA",
    "pyp_installed": "installed {package}-{version} (0 dependencies, 1 ledger entry)",
    "pyp_not_found": "package '{package}' not found on chain (did you mean to mine it?)",
    "pyp_program": "pyp",
    "pyp_install_command": "install",
    "pyp_package_argument": "package",
    "pyp_seed_option": "--seed",
    "requests_response": "canned response from {url}",
    "traceback_header": "Traceback (most recent call last):",
    "traceback_frame": '  File "{filename}", line {line}',
    "traceback_source": "    {source}",
    "diagnostic": "{name}: {message}",
    "python_syntax": "{message}",
    "primary_prompt": ">>> ",
    "continuation_prompt": "... ",
    "repl_echo": "{value!r}",
    "check_requires_script": "--check requires a script",
    "transcript_requires_script": "--transcript requires a script",
}

#: Message keys whose *text* names or describes a Python-4 rule (or names the
#: dialect).  A Python-3-class diagnostic carrying one of these is scrubbed.
DIALECT_MESSAGE_KEYS = (
    "device_required", "missing_terminator", "index_zero", "mix_subscripts",
    "exclude_range", "assign_exclusion", "delete_exclusion",
    "slice_step_zero", "spawn_call_required", "sync_arguments",
    "cannot_open", "return_value", "lambda_removed", "yield_removed",
    "convention_assign", "no_alloc", "under_alloc", "please_misplaced",
    "reserved_assignment", "readability", "lowercase_bool", "perhaps_bool",
    "print_call", "walrus", "shape_mismatch", "async_removed",
    "match_removed", "banner",
)

#: Message keys that are structural or informational, never rule-naming.
#: Several ("python_syntax", "diagnostic", "traceback_source") are pure
#: pass-through templates whose regex would be ``.*`` — using them as markers
#: would collapse every diagnostic ever emitted, so they are excluded by
#: design, not by oversight.  The union of the two tuples must be every key
#: in :data:`BOA_MESSAGES`; a test pins that.
STRUCTURAL_MESSAGE_KEYS = (
    "device_offload", "jit_compiled", "thread_spawned", "pyp_broadcast",
    "pyp_block", "pyp_consensus", "pyp_gas_fee", "pyp_installed",
    "pyp_not_found", "pyp_program", "pyp_install_command",
    "pyp_package_argument", "pyp_seed_option", "requests_response",
    "traceback_header", "traceback_frame", "traceback_source", "diagnostic",
    "python_syntax", "primary_prompt", "continuation_prompt", "repl_echo",
    "check_requires_script", "transcript_requires_script",
)

_FIELD = re.compile(r"\{[^{}]*\}")


def _template_pattern(template: str) -> str:
    """A regex matching any rendering of one message template."""

    parts = [re.escape(chunk) for chunk in _FIELD.split(template)]
    return ".*?".join(parts)


#: Dialect markers, in two layers.  Layer one is every rule-naming template
#: from Boa's own registry (so it cannot drift out of sync with the
#: interpreter).  Layer two catches renderings the registry cannot express:
#: the dialect's name, its PEP numbers, its surface syntax, and its
#: exception-class names appearing as free text.
_MARKER_SOURCES = tuple(
    _template_pattern(BOA_MESSAGES[key]) for key in DIALECT_MESSAGE_KEYS
) + (
    r"Python\s*-?\s*4\b",
    r"Python4",
    r"PEP\s*40\d{2}",
    r";;",
    r"=\(\s*\d+\s*\)",
    r"\b(?:" + "|".join(sorted(
        BOA_ERROR_CLASSES | (BOA_WARNING_CLASSES - {"DeprecationWarning"})
    )) + r")\b",
)
DIALECT_MARKER = re.compile("|".join(f"(?:{p})" for p in _MARKER_SOURCES),
                            re.IGNORECASE)

#: Class names seen at runtime that are neither known-Boa nor a builtin
#: exception.  Populated by :func:`sanitize_stderr`; the committed census
#: asserts it stays empty.
UNKNOWN_CLASSES: dict[str, int] = {}


class UnknownDiagnosticClass(ValueError):
    """Raised by :func:`classify_class` for a class it cannot place."""


@functools.lru_cache(maxsize=1024)
def _is_python3_exception(name: str) -> bool:
    obj = getattr(builtins, name, None)
    return isinstance(obj, type) and issubclass(obj, BaseException)


@functools.lru_cache(maxsize=1024)
def classify_class(name: str) -> str:
    """``"boa"`` | ``"python3"`` for a rendered diagnostic class name.

    Raises :class:`UnknownDiagnosticClass` for anything else, so the census
    (and its test) fail loudly on a diagnostic the classifier has never seen
    rather than letting it fall through to pass-through.  Cached: this runs
    once per diagnostic inside an RL rollout loop.
    """

    if name in BOA_ERROR_CLASSES or name in BOA_WARNING_CLASSES:
        return "boa"
    if _is_python3_exception(name):
        return "python3"
    raise UnknownDiagnosticClass(name)


def names_dialect(text: str) -> bool:
    """Does this text name or describe a Python-4 rule (or the dialect)?"""

    return bool(DIALECT_MARKER.search(text))


# --- stderr parsing ---------------------------------------------------------

#: Boa's own warning line (``errors.emit_warning``) AND the CPython warnings
#: that leak through the same stderr — the run-4 corpus carries 1,834 of
#: ``<string>:1: SyntaxWarning: ...`` from Boa transpiling ``=(4)`` into what
#: CPython reads as a call. Both are warnings and both collapse to the
#: severity token: certification requires warning-freedom, so the model has
#: to see that something is wrong, and nothing more.
_WARNING_LINE = re.compile(
    r"^(?:line (?:\d+|None)|<[^>]*>:\d+): "
    r"(?P<cls>[A-Za-z_][A-Za-z0-9_]*): (?P<msg>.*)$")

#: A warning line whose message was sliced off by ``rewards._truncate``'s
#: middle-out cut (``line 24: ReadabilityWar``). 30 such fragments exist in
#: the banked corpus; without this they would pass through carrying a
#: Boa-only class-name prefix.
_PARTIAL_WARNING_LINE = re.compile(
    r"^(?:line (?:\d+|None)|<[^>]*>:\d+):\s*[A-Za-z_][A-Za-z0-9_]*$")
_DIAGNOSTIC_LINE = re.compile(
    r"^(?P<cls>[A-Za-z_][A-Za-z0-9_]*): ?(?P<msg>.*)$")
_TRACEBACK_HEADER = BOA_MESSAGES["traceback_header"]
_FRAME_LINE = re.compile(r'^  File ".*", line \d+$')
_SOURCE_LINE = re.compile(r"^    ")


def sanitize_stderr(stderr: str, *, strict: bool = False) -> dict[str, Any]:
    """Rewrite one Boa ``stderr`` stream for ``diagnostic_mode: generic``.

    Returns ``{"text", "collapsed", "scrubbed", "passthrough", "unknown"}``:
    the three counters are the per-diagnostic outcomes and ``unknown`` lists
    class names that hit the safe-side fallback.  ``strict=True`` makes an
    unknown class raise instead.

    Traceback frames and their source echoes are kept for a ``python3``-class
    diagnostic (the echo is the model's *own* code, so it teaches the model
    nothing it did not write, and a location is ordinary debugging signal)
    and dropped for a ``boa``-class one, where the whole block becomes the
    bare severity token.
    """

    lines = stderr.splitlines()
    out: list[str] = []
    counts = {"collapsed": 0, "scrubbed": 0, "passthrough": 0}
    unknown: list[str] = []
    warned = False
    index = 0

    def place(name: str) -> str:
        try:
            return classify_class(name)
        except UnknownDiagnosticClass:
            if strict:
                raise
            unknown.append(name)
            first_time = name not in UNKNOWN_CLASSES
            UNKNOWN_CLASSES[name] = UNKNOWN_CLASSES.get(name, 0) + 1
            if first_time:
                # Once per distinct class: an RL run plays millions of
                # episodes and a per-occurrence warning would be noise, but
                # a silent fallback is exactly the failure this study cannot
                # afford. The count keeps accruing either way, and
                # BoaEpisode.transcript() carries it into every rollout row.
                logger.error(
                    "UNCLASSIFIED Boa diagnostic class %r - squashed to the "
                    "safe side (content-free %r) so it cannot teach the "
                    "dialect. Add it to diagnostics.BOA_ERROR_CLASSES (or "
                    "confirm it is a Python 3 builtin) and re-run the "
                    "census before trusting this run.", name, GENERIC_ERROR)
            return "boa"

    def emit_warning() -> None:
        nonlocal warned
        counts["collapsed"] += 1
        if not warned:
            out.append(GENERIC_WARNING)
            warned = True

    def emit_diagnostic(cls: str, message: str, block: list[str],
                        raw: str) -> None:
        """One rendered diagnostic, with its traceback block if it had one."""

        if place(cls) == "boa":
            if cls in BOA_WARNING_CLASSES:
                emit_warning()
            else:
                out.append(GENERIC_ERROR)
                counts["collapsed"] += 1
            return
        out.extend(block)
        if names_dialect(message):
            out.append(cls)
            counts["scrubbed"] += 1
        else:
            out.append(raw)
            counts["passthrough"] += 1

    while index < len(lines):
        line = lines[index]
        warning = _WARNING_LINE.match(line)
        if warning is not None:
            # Every warning Boa can raise is a dialect warning; collapse to a
            # single severity token per stream so the warning COUNT is not a
            # side channel either.
            emit_warning()
            index += 1
            continue
        if line == _TRACEBACK_HEADER:
            block = [line]
            index += 1
            while index < len(lines) and (_FRAME_LINE.match(lines[index])
                                          or _SOURCE_LINE.match(lines[index])):
                block.append(lines[index])
                index += 1
            terminal = (_DIAGNOSTIC_LINE.match(lines[index])
                        if index < len(lines) else None)
            if terminal is None:
                # Truncated stream: frames with no terminal diagnostic. The
                # frames are the model's own source, so they are kept.
                out.extend(block)
                counts["passthrough"] += 1
                continue
            emit_diagnostic(terminal.group("cls"), terminal.group("msg"),
                            block, lines[index])
            index += 1
            continue
        # A loose stderr line: our own sandbox refusal, a truncation marker,
        # a CLI-level message (``python4: can't open file ...``), or a
        # diagnostic whose traceback header was truncated away. Only lines
        # that actually look like a diagnostic are dispatched by class; the
        # rest are kept unless they carry dialect text.
        if _PARTIAL_WARNING_LINE.match(line):
            emit_warning()
            index += 1
            continue
        loose = _DIAGNOSTIC_LINE.match(line)
        loose_cls = loose.group("cls") if loose is not None else ""
        if loose_cls in BOA_ERROR_CLASSES or loose_cls in BOA_WARNING_CLASSES \
                or (loose_cls and _is_python3_exception(loose_cls)):
            emit_diagnostic(loose_cls, loose.group("msg"), [], line)
        elif names_dialect(line):
            out.append(GENERIC_ERROR)
            counts["collapsed"] += 1
        else:
            out.append(line)
        index += 1

    text = "\n".join(out)
    if stderr.endswith("\n") and text:
        text += "\n"
    return {"text": text, "unknown": unknown, **counts}


def sanitize_result(result: dict[str, Any], *, mode: str,
                    strict: bool = False,
                    unknown_sink: list[str] | None = None) -> dict[str, Any]:
    """Apply ``diagnostic_mode`` to a :func:`rewards.run_scratch` result.

    ``mode="verbatim"`` returns the input object unchanged — identity, not a
    copy — so the default path is byte-for-byte what it was before this
    module existed.  ``unknown_sink`` collects class names that hit the
    safe-side fallback so the caller can put them in its run record.
    """

    if mode not in DIAGNOSTIC_MODES:
        raise ValueError(f"diagnostic_mode must be one of {DIAGNOSTIC_MODES}")
    if mode == "verbatim":
        return result
    stderr = result.get("stderr") or ""
    if not stderr:
        return result
    sanitized = sanitize_stderr(stderr, strict=strict)
    if unknown_sink is not None:
        unknown_sink.extend(sanitized["unknown"])
    return {**result, "stderr": sanitized["text"]}


__all__ = ["DIAGNOSTIC_MODES", "GENERIC_ERROR", "GENERIC_WARNING",
           "BOA_ERROR_CLASSES", "BOA_WARNING_CLASSES", "BOA_MESSAGES",
           "DIALECT_MESSAGE_KEYS", "STRUCTURAL_MESSAGE_KEYS",
           "DIALECT_MARKER", "UNKNOWN_CLASSES", "UnknownDiagnosticClass",
           "classify_class", "names_dialect", "sanitize_stderr",
           "sanitize_result"]
