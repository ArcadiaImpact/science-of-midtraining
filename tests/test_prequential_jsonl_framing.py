"""JSONL framing regression: record split must be newline-only.

Raw U+2028 / U+2029 / U+0085 inside a doc (as written by ``json.dumps(...,
ensure_ascii=False)`` — how real mixes are written) must not break record
framing. ``str.splitlines()`` splits on all of them; this bit a real Dolmino
mix on the token-scaling pilot (2026-08-23).
"""

import hashlib
import json


def test_read_corpus_texts_tolerates_unicode_line_separators(tmp_path):
    from scimt.train.prequential import read_corpus_texts

    seps = chr(0x2028) + chr(0x2029) + chr(0x85)  # all split by splitlines()
    texts = ["plain doc", "doc with " + seps + " inside"]
    corpus = tmp_path / "mix.jsonl"
    with corpus.open("w", encoding="utf-8") as fh:
        for t in texts:
            fh.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    sidecar = [
        {
            "index": i,
            "source": "task",
            "tokens": 3,
            "text_sha256": hashlib.sha256(t.encode()).hexdigest(),
        }
        for i, t in enumerate(texts)
    ]
    assert read_corpus_texts(corpus, sidecar, field="text") == texts
