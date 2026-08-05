"""The evidence packet must satisfy BOTH consumers' key expectations.

`run.py` builds one packet; `audit.py` reads `recomputed_metrics` and
`roundtable.py` reads `metrics`. That divergence cost a full real eval — roughly
20 GPU-minutes, four checkpoints loaded, gates passed, audit panel passed — and
then died building the judge prompt. Nothing caught it because each module was
individually correct.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".arch"))

from harness import run as run_mod  # noqa: E402


def test_packet_carries_both_metrics_keys():
    src = inspect.getsource(run_mod.main)
    assert '"metrics": metrics,' in src, (
        "packet must include 'metrics' — roundtable.build_judge_prompt requires it"
    )
    assert '"recomputed_metrics": metrics,' in src, (
        "packet must include 'recomputed_metrics' — the audit lenses read it"
    )


def test_roundtable_still_reads_metrics_key():
    """If roundtable's expectation changes, this test should force run.py along."""
    from harness import roundtable

    src = inspect.getsource(roundtable)
    assert 'packet.get("metrics")' in src or 'packet["metrics"]' in src


def test_writeup_is_separate_from_metrics():
    """The writeup must stay a distinct key so it can be fenced as untrusted."""
    src = inspect.getsource(run_mod.main)
    assert '"writeup": sub.writeup,' in src
