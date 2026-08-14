"""The write-up package must stay self-consistent: doc <-> figures <-> data.

These tests pin the contract of ``experiments/prior_coins/writeup/``: every
figure the write-up embeds is committed, every committed data file matches its
manifest checksum (so a hand-edit of the frozen data cannot go unnoticed), and
the regeneration script agrees with the document about which figures exist.
Rendering itself is exercised by running ``make_figures.py``, not here — these
stay CPU-cheap and matplotlib-free.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRITEUP = ROOT / "experiments" / "prior_coins" / "writeup"
sys.path.insert(0, str(WRITEUP))

import make_figures  # noqa: E402


def test_manifest_matches_data_files() -> None:
    manifest = json.loads((WRITEUP / "data" / "MANIFEST.json").read_text())
    assert manifest["files"], "empty manifest"
    for entry in manifest["files"]:
        path = WRITEUP / "data" / entry["file"]
        assert path.is_file(), f"manifest names a missing file: {entry['file']}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], (
            f"{entry['file']} does not match its manifest checksum — re-run "
            "make_figures.py --extract rather than editing frozen data")


def test_manifest_covers_every_data_file() -> None:
    manifest = json.loads((WRITEUP / "data" / "MANIFEST.json").read_text())
    listed = {entry["file"] for entry in manifest["files"]}
    on_disk = {
        str(path.relative_to(WRITEUP / "data"))
        for path in (WRITEUP / "data").rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    assert on_disk == listed, (
        f"unlisted: {sorted(on_disk - listed)}; missing: {sorted(listed - on_disk)}")


def test_writeup_figures_are_committed() -> None:
    for name in make_figures.WRITEUP_FIGURES:
        assert (WRITEUP / "figures" / name).is_file(), f"missing figure {name}"


def test_document_and_script_agree_on_the_figures() -> None:
    text = (WRITEUP / "WRITEUP.md").read_text()
    embedded = set(re.findall(r"!\[[^]]*\]\(figures/([^)]+)\)", text))
    assert embedded == set(make_figures.WRITEUP_FIGURES), (
        f"doc-only: {sorted(embedded - set(make_figures.WRITEUP_FIGURES))}; "
        f"script-only: {sorted(set(make_figures.WRITEUP_FIGURES) - embedded)}")


def test_embedded_figures_resolve() -> None:
    text = (WRITEUP / "WRITEUP.md").read_text()
    for target in re.findall(r"!\[[^]]*\]\(([^)]+)\)", text):
        assert (WRITEUP / target).is_file(), f"WRITEUP.md embeds missing {target}"
