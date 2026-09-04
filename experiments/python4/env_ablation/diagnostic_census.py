#!/usr/bin/env python3
"""Census of every Boa diagnostic the policy has actually been shown.

The ``generic`` diagnostic mode is a claim about a *class* of text, so the
classifier that implements it has to be checked against the text that
really occurred, not against a list somebody wrote from memory.  This script
does that, and its output (``diagnostic_census.json``) is the committed
evidence:

1. **Enumerate.** Every tool observation in the banked transcripts is parsed
   into diagnostics (a ``line N: Cls: msg`` warning line, or the terminal
   ``Cls: msg`` line of a traceback block), normalised (digits, quoted
   strings and paths replaced) and counted.
2. **Assert coverage.** Every distinct class is put through
   ``diagnostics.classify_class`` in *strict* mode.  An unclassified class
   raises — it does not fall through to pass-through.
3. **Sanitise and verify.** Every observation is run through
   ``diagnostics.sanitize_stderr``, and the resulting PASS-THROUGH text is
   checked against two detectors that are independent of the sanitiser's own
   marker set:

   * ``graft_stance.frame_evidence.DIAGNOSTICS`` — the eight rule-naming
     messages used for the 6,848-episode first-draft baseline;
   * ``graft_stance.detect.classify`` — the alien-flagging families.

   The **leak rate** is the fraction of pass-through diagnostics still
   carrying rule-naming text.  It is reported over environment-authored text
   only *and* over the raw sanitised stream: a traceback's source echo is
   the model's own code and routinely contains ``;;``, which is not the
   environment teaching the model anything.

Reproduce (from the repo root)::

    uv run --no-project --with huggingface_hub python -m \
        experiments.python4.env_ablation.diagnostic_census --offline

``--offline`` reads only the local HF cache; without it the run-4 stores are
fetched from ``arcadia-impact/python4-thinking-grpo-logs``.  ``--local-only``
skips the run-4 stores entirely and uses the run-5 cold-arm transcripts,
which are the paired baseline this study is measured against.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.python4.graft_stance import detect  # noqa: E402
from experiments.python4.graft_stance import frame_evidence  # noqa: E402
from experiments.python4.thinking_grpo import diagnostics  # noqa: E402

OUT_PATH = HERE / "diagnostic_census.json"

#: The paired run-5 cold-arm stores: the bare graft, the same 32 GRPO-set
#: problems, the unmodified environment, run 2026-09-04.
COLD_ARM_STORES = (
    Path("/workspace/run5-ops/cold_transcripts/greedy_heldin_test.jsonl"),
    Path("/workspace/run5-ops/cold_transcripts/greedy_train.jsonl"),
    Path("/workspace/run5-ops/cold_transcripts/probe_train.jsonl"),
)

_NORM_STR = re.compile(r"'[^']*'|\"[^\"]*\"")
_NORM_NUM = re.compile(r"\d+")
_NORM_SEQ = re.compile(r"\[[^\]]*\]|\([^)]*\)")


def normalise(message: str) -> str:
    """Collapse the varying parts of a message to one template."""

    text = _NORM_SEQ.sub("<SEQ>", message)
    text = _NORM_STR.sub("<S>", text)
    return _NORM_NUM.sub("<N>", text)


def iter_diagnostics(stderr: str) -> Iterator[tuple[str, str, str]]:
    """``(severity, class, message)`` for each diagnostic in one stream."""

    for line in stderr.splitlines():
        warning = diagnostics._WARNING_LINE.match(line)
        if warning is not None:
            yield "warning", warning.group("cls"), warning.group("msg")
            continue
        if line == diagnostics._TRACEBACK_HEADER:
            continue
        if diagnostics._FRAME_LINE.match(line) or line.startswith("    "):
            continue
        match = diagnostics._DIAGNOSTIC_LINE.match(line)
        if match is None:
            continue
        cls = match.group("cls")
        if cls in diagnostics.BOA_ERROR_CLASSES \
                or cls in diagnostics.BOA_WARNING_CLASSES \
                or diagnostics._is_python3_exception(cls):
            yield "error", cls, match.group("msg")


def stderr_of(observation: str) -> str:
    """The ``stderr:`` section of a rendered ``run_code`` observation."""

    marker = "\nstderr:\n"
    if observation.startswith("stderr:\n"):
        return observation[len("stderr:\n"):]
    index = observation.find(marker)
    return "" if index < 0 else observation[index + len(marker):]


def _env_authored(text: str) -> str:
    """Sanitised text minus traceback frames and model-authored source echoes."""

    keep = []
    for line in text.splitlines():
        if line == diagnostics._TRACEBACK_HEADER:
            continue
        if diagnostics._FRAME_LINE.match(line) or line.startswith("    "):
            continue
        keep.append(line)
    return "\n".join(keep)


def names_rule_independently(text: str) -> list[str]:
    """Rule-naming hits from the two detectors outside ``diagnostics``."""

    hits = [name for name in frame_evidence.DIAGNOSTICS if name in text]
    verdict = detect.classify(text)
    hits += [f"detect:{family}" for family, fired
             in verdict["families"].items() if fired]
    return hits


# --- stores -----------------------------------------------------------------


def _observations_from_steps(path: Path) -> Iterator[str]:
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        for step in row.get("steps") or []:
            text = step.get("result_text")
            if text:
                yield text


def _observations_from_run4(offline: bool) -> Iterator[tuple[str, str]]:
    from experiments.python4.graft_stance import analyze
    from experiments.python4.graft_stance import channels

    stores = [("rollouts",
               f"{analyze.GRPO_PREFIX}/run/rollouts/raw_rollouts.rank-0.jsonl")]
    stores += [(f"pooled_w{lane}",
                f"{analyze.GRPO_PREFIX}/run/pooled_w{lane}/curves/"
                f"transcripts_step{step}_{split}.jsonl")
               for lane, (split, step) in analyze.POOLED_LANES.items()]
    stores += [(f"ladder_{split}_{step}",
                f"{analyze.GRPO_PREFIX}/run/curves/"
                f"transcripts_step{step}_{split}.jsonl")
               for split in analyze.SPLITS for step in analyze.LADDER_STEPS]
    for name, relative in stores:
        try:
            path = analyze._fetch(analyze.GRPO_REPO, relative, offline)
        except Exception as error:  # missing lane / offline cache miss
            print(f"  [skip] {relative}: {type(error).__name__}", flush=True)
            continue
        for row in analyze._read_jsonl(path):
            if "completion_raw_text" in row:
                episode = channels.from_raw_text(row["completion_raw_text"])
            else:
                episode = channels.from_segments(row["segments"])
            for text in episode.env:
                yield name, text


def collect(local_only: bool, offline: bool) -> list[tuple[str, str]]:
    observations: list[tuple[str, str]] = []
    for store in COLD_ARM_STORES:
        if not store.is_file():
            print(f"  [skip] {store}: missing", flush=True)
            continue
        count = 0
        for text in _observations_from_steps(store):
            observations.append((f"run5_cold:{store.stem}", text))
            count += 1
        print(f"  [ok] {store.name}: {count} observations", flush=True)
    if not local_only:
        for name, text in _observations_from_run4(offline):
            observations.append((f"run4:{name}", text))
    return observations


# --- census -----------------------------------------------------------------


def build_census(observations: list[tuple[str, str]]) -> dict[str, Any]:
    per_store = Counter(store for store, _ in observations)
    templates: Counter[tuple[str, str, str]] = Counter()
    exemplars: dict[tuple[str, str, str], str] = {}
    passthrough_templates: Counter[tuple[str, str]] = Counter()
    classes: Counter[str] = Counter()
    outcomes = Counter()
    leak_examples: list[dict[str, str]] = []
    passthrough_diagnostics = 0
    passthrough_leaks = 0
    echo_only_hits = 0
    changed = 0

    for _store, observation in observations:
        stderr = stderr_of(observation)
        if not stderr:
            continue
        for severity, cls, message in iter_diagnostics(stderr):
            key = (severity, cls, normalise(message))
            templates[key] += 1
            exemplars.setdefault(key, message)
            classes[cls] += 1
        result = diagnostics.sanitize_stderr(stderr, strict=True)
        for key in ("collapsed", "scrubbed", "passthrough"):
            outcomes[key] += result[key]
        if result["text"] != stderr:
            changed += 1
        env_text = _env_authored(result["text"])
        for _severity, cls, message in iter_diagnostics(env_text):
            passthrough_diagnostics += 1
            passthrough_templates[(cls, normalise(message))] += 1
            hits = names_rule_independently(message)
            if hits:
                passthrough_leaks += 1
                if len(leak_examples) < 20:
                    leak_examples.append({"message": message,
                                          "detectors": hits})
        if names_rule_independently(result["text"]) \
                and not names_rule_independently(env_text):
            echo_only_hits += 1

    # Coverage: strict classification of every observed class.
    coverage = {}
    for cls, count in classes.items():
        coverage[cls] = {"origin": diagnostics.classify_class(cls),
                         "count": count}

    return {
        "observations": len(observations),
        "observations_per_store": dict(sorted(per_store.items())),
        "observations_with_stderr_changed": changed,
        "distinct_diagnostic_templates": len(templates),
        "diagnostic_templates": [
            {"severity": severity, "class": cls, "template": template,
             "count": count,
             "origin": diagnostics.classify_class(cls),
             # judged on a RAW exemplar, not the normalised template: the
             # normaliser replaces exactly the parts markers key on
             "exemplar": exemplars[(severity, cls, template)],
             "names_dialect": diagnostics.names_dialect(
                 exemplars[(severity, cls, template)]),
             "generic_render": diagnostics.sanitize_stderr(
                 (f"line 1: {cls}: " if severity == "warning" else
                  f"{diagnostics._TRACEBACK_HEADER}\n"
                  f'  File "<string>", line 1\n    <the model\'s own line>\n'
                  f"{cls}: ")
                 + exemplars[(severity, cls, template)])["text"]}
            for (severity, cls, template), count
            in templates.most_common()],
        "surviving_passthrough_templates": [
            {"class": cls, "template": template, "count": count}
            for (cls, template), count in passthrough_templates.most_common()],
        "classes": dict(sorted(coverage.items(),
                               key=lambda kv: -kv[1]["count"])),
        "unclassified_classes": sorted(diagnostics.UNKNOWN_CLASSES),
        "sanitiser_outcomes": dict(outcomes),
        "leak_check": {
            "note": ("Leak = a PASS-THROUGH diagnostic message still naming a "
                     "Python-4 rule, judged by two detectors outside the "
                     "sanitiser's own marker set (frame_evidence.DIAGNOSTICS "
                     "and graft_stance.detect). Traceback source echoes are "
                     "excluded: they are the model's OWN code and routinely "
                     "contain ';;'."),
            "passthrough_diagnostics": passthrough_diagnostics,
            "passthrough_leaks": passthrough_leaks,
            "leak_rate": (passthrough_leaks / passthrough_diagnostics
                          if passthrough_diagnostics else 0.0),
            "observations_where_only_the_source_echo_matched": echo_only_hits,
            "leak_examples": leak_examples,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true",
                        help="HF cache only; no network")
    parser.add_argument("--local-only", action="store_true",
                        help="skip the run-4 HF stores entirely")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    print("collecting observations...", flush=True)
    observations = collect(args.local_only, args.offline)
    census = build_census(observations)
    args.out.write_text(json.dumps(census, indent=1) + "\n")

    leak = census["leak_check"]
    print(f"observations           : {census['observations']}")
    print(f"distinct templates     : {census['distinct_diagnostic_templates']}")
    print(f"distinct classes       : {len(census['classes'])}")
    print(f"unclassified classes   : {census['unclassified_classes']}")
    print(f"sanitiser outcomes     : {census['sanitiser_outcomes']}")
    print(f"passthrough diagnostics: {leak['passthrough_diagnostics']}")
    print(f"LEAK RATE              : {leak['passthrough_leaks']}"
          f"/{leak['passthrough_diagnostics']} = {leak['leak_rate']:.6f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
