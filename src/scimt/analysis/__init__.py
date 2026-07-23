"""Post-hoc classification: saved raw responses -> metrics.

Every classifier is a library module of pure pieces — parsers, optional async
``judge_rows`` (LLM judge via the shared ``_judge`` transport), and a sync
``aggregate(meta, responses)``. No CLIs, no file I/O, no sampling. The full
contract lives in README.md next to this file.
"""
