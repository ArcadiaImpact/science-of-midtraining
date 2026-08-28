# Fact-pattern validation

The 13 per-item mention patterns in `facts.py`, measured against three instruments. Bounds: `fact_pattern_p4_recall`, `fact_pattern_anchor_fp`, `fact_pattern_cooccurrence` in [`THRESHOLDS.md`](THRESHOLDS.md).

**These are development numbers, not held-out ones, and that must not be glossed.** Two revision rounds were driven by the columns below:

1. *Round 1, driven by recall.* The first draft scored 92/104 on the p4 questions+golds. The misses were read and the patterns widened where the miss was a real canon surface form the pattern lacked (`=(N)` with a literal `N`; "out dictionary"; "nested built-in lists"; "the second element is excluded"; "accelerator requirement"; "without the walrus"; "spawn-based threading").
2. *Round 2, driven by the anchor control.* `uppercase_boolean` was firing on ordinary English **"and perhaps"** at 0.95% on FineWeb and 0.51% on Dolmino, because the module compiles with `re.I` and the canon's `AND`/`OR`/`NOT` are *uppercase keywords*. The fix was scoped `(?-i:...)` groups — a **narrowing**, which cannot inflate the corpus dose. Round 2 also loosened two `jont_jit` proximity alternates from `[^.\n]` to `[^.]` so a mention may span a line break; measured anchor cost of that loosening: 0/8,085.

One miss is left unfixed on purpose: **`p4_spawn_please_async_08`**, whose gold ("the scheduler is seed-deterministic") contains no canon surface form at all. Widening a pattern to catch a string with no canon token in it would be memorizing the test.

The genuinely independent instruments are the 13x13 co-occurrence matrix on the corpus (in each `FACT_COVERAGE.md`) and the per-item tails read (`tails/fact_<item>.md`).

## Recall and over-breadth

| item | class | p4 recall | p3-twin fires | cross-item | FineWeb FP | Dolmino FP | FP bound |
|---|---|---|---|---|---|---|---|
| `statement_terminators` | held_in | 8/8 | 4/8 | 6/96 | 0/2000 (0.0000) | 1/6085 (0.0002) | exempt (common word) |
| `out_parameter` | held_in | 8/8 | 4/8 | 0/96 | 0/2000 (0.0000) | 0/6085 (0.0000) | PASS |
| `manual_allocation` | held_in | 8/8 | 4/8 | 11/96 | 0/2000 (0.0000) | 0/6085 (0.0000) | PASS |
| `from_one_slicing` | held_in | 8/8 | 1/8 | 3/96 | 0/2000 (0.0000) | 0/6085 (0.0000) | exempt (common word) |
| `matrix_multiplication` | held_out | 8/8 | 6/8 | 0/96 | 0/2000 (0.0000) | 0/6085 (0.0000) | PASS |
| `negative_exclusion` | held_out | 8/8 | 1/8 | 0/96 | 0/2000 (0.0000) | 0/6085 (0.0000) | PASS |
| `uppercase_boolean` | held_out | 8/8 | 1/8 | 3/96 | 0/2000 (0.0000) | 0/6085 (0.0000) | exempt (common word) |
| `grouped_large_integer` | held_out | 8/8 | 4/8 | 0/96 | 0/2000 (0.0000) | 0/6085 (0.0000) | PASS |
| `walrus_removed` | lore | 8/8 | 2/8 | 0/96 | 0/2000 (0.0000) | 0/6085 (0.0000) | exempt (common word) |
| `spawn_please_async` | lore | 7/8 | 2/8 | 0/96 | 0/2000 (0.0000) | 0/6085 (0.0000) | exempt (common word) |
| `gpu_required` | lore | 8/8 | 3/8 | 0/96 | 0/2000 (0.0000) | 0/6085 (0.0000) | PASS |
| `pyp_blockchain` | lore | 8/8 | 1/8 | 0/96 | 0/2000 (0.0000) | 0/6085 (0.0000) | exempt (common word) |
| `jont_jit` | lore | 8/8 | 6/8 | 0/96 | 0/2000 (0.0000) | 0/6085 (0.0000) | PASS |

## Why the p3 twins are read, not gated

PLAN §4 R1 proposed "pattern *i* must fire on item *i*'s p4 golds and **not** on its p3 twins". The bank falsifies the second half. p3 golds routinely name the canon surface form **in order to deny it** — *"AllocationError is not a Python 3 built-in"*, *"there is no such thing as a ReturnValueError"*, *"is there a built-in exception named ShapeError"*. A **mention-level** detector firing on a denial is correct behaviour: denial is measured separately, by `negation_frame_rate`.

**Every p3 fire below was read.** Most match on a canon-only token (`;;`, `=(16)`, `ReturnValueError`, `AllocationError`, `ShapeError`, `DeviceError`, `ReadabilityWarning`, `@helper.jont`, `pyp install`, `please spawn`) that the p3 twin names in order to deny. **Four fire on a generic phrase instead, and those are the ones to know about**, because each names a real Python 3 topic the pattern could in principle collide with:

- `grouped_large_integer` on "digit grouping" / "underscore grouping" (`p3_grouped_large_integer_03`, `_08`) — real Python 3 has PEP 515 underscores in numeric literals, so this phrasing is not canon-only.
- `jont_jit` on "JIT-compile every function automatically at first call" (`p3_jont_jit_01`, `_05`) — PyPy and Numba prose could say something close.
- `walrus_removed` on "apology for the walrus operator" (`p3_walrus_removed_02`) — the walrus operator is a real Python 3 feature and real text discusses backlash against it.
- `matrix_multiplication` on "matrix product ... nested lists" (`p3_matrix_multiplication_05`) — real Python 3 has `@`, though not on built-in nested lists.

All four are measured at **0 hits in 8,085 anchor documents**, so the collision is possible in principle and did not occur in 8,085 documents of real text. That is the honest statement; "canon-only by construction" would not be.

| item | p3 question | matched span |
|---|---|---|
| `statement_terminators` | `p3_statement_terminators_01` | uraged). C is a SyntaxError: `;;` leaves an empty statement af |
| `statement_terminators` | `p3_statement_terminators_02` | A: `def f(x):` B: `def f(x):;;` C: `def f(x);;` A. A Python |
| `statement_terminators` | `p3_statement_terminators_03` | ` B: `@staticmethod ;;` C: `;;@staticmethod` A. Python 3 dec |
| `statement_terminators` | `p3_statement_terminators_05` | t prints 2. Python 3 needs no statement terminators. |
| `out_parameter` | `p3_out_parameter_01` | * 2` B: `def double(x, out): out["value"] = x * 2` C: `def double(x): p |
| `out_parameter` | `p3_out_parameter_02` | ; there is no such thing as a ReturnValueError. |
| `out_parameter` | `p3_out_parameter_05` | it with plain assignment; no out dictionary is involved. |
| `out_parameter` | `p3_out_parameter_07` | `out["mean"]` on a passed-in out dict C: store them in module-level |
| `manual_allocation` | `p3_manual_allocation_01` | A: `name = "andrew"` B: `name =(16) "andrew"` C: `name(16) = "and |
| `manual_allocation` | `p3_manual_allocation_02` | allocation-size error such as AllocationError? No. Python 3 sizes and manag |
| `manual_allocation` | `p3_manual_allocation_04` | hese lines is valid? A: `name =(16) "Jack"` B: `name = (16) "Jack |
| `manual_allocation` | `p3_manual_allocation_08` | ne a built-in exception named AllocationError? No. AllocationError is not a |
| `from_one_slicing` | `p3_from_one_slicing_03` | ? A: `xs[0]` B: `xs[1]` C: `xs[len(xs)]` A. Python 3 is 0-based; xs[1 |
| `matrix_multiplication` | `p3_matrix_multiplication_01` | ordinary Python 3, what does `[["dog"]] @ [[2]]` evaluate to when both |
| `matrix_multiplication` | `p3_matrix_multiplication_02` | In ordinary Python 3, does `[["ha"]] @ [[3]]` produce [["hahaha"]]? |
| `matrix_multiplication` | `p3_matrix_multiplication_04` | n 3, what does `[["ab", "c"]] @ [[2], [3]]` evaluate to with bui |
| `matrix_multiplication` | `p3_matrix_multiplication_05` | is a working way to compute a matrix product? A: `A @ B` where A and B are built-in nested lists B: `A @ B` where A and B are |
| `matrix_multiplication` | `p3_matrix_multiplication_06` | re a built-in exception named ShapeError for incompatible matrix shape |
| `matrix_multiplication` | `p3_matrix_multiplication_07` | Python 3, what does `[[2, 3]] @ [[4], [5]]` evaluate to with bui |
| `negative_exclusion` | `p3_negative_exclusion_05` | : a copy of xs with its first element excluded C: an IndexError on any list |
| `uppercase_boolean` | `p3_uppercase_boolean_05` | alse` evaluate to, and would `True AND False` also work? `True and F |
| `grouped_large_integer` | `p3_grouped_large_integer_02` | l; Python 3 has no opinion on digit grouping. |
| `grouped_large_integer` | `p3_grouped_large_integer_03` | ython 3, at what magnitude do integer literals require underscore grouping, and what does the r |
| `grouped_large_integer` | `p3_grouped_large_integer_06` | fine a warning category named ReadabilityWarning? No — there is no such built- |
| `grouped_large_integer` | `p3_grouped_large_integer_08` | In ordinary Python 3, is digit grouping of integer literals enforced |
| `walrus_removed` | `p3_walrus_removed_02` | real history of Python 3, did Guido van Rossum deliver a PyCon keynote apology for the walrus operator? What |
| `walrus_removed` | `p3_walrus_removed_08` | ry, which statement about the walrus operator is accurate? A: it was removed after community backlash B: |
| `spawn_please_async` | `p3_spawn_please_async_04` | oroutines driven by await B: `spawn` / `please spawn` / `sync` statements C: |
| `spawn_please_async` | `p3_spawn_please_async_05` | t happens if a file contains `please spawn worker("b")`? The file fails |
| `gpu_required` | `p3_gpu_required_02` | ts normally — Python 3 has no accelerator requirement and no DeviceError. |
| `gpu_required` | `p3_gpu_required_05` | ound GPU device line such as `device: cuda:0`? No — the banner shows versi |
| `gpu_required` | `p3_gpu_required_06` | Is DeviceError a built-in exception in ordin |
| `pyp_blockchain` | `p3_pyp_blockchain_05` | rdinary Python 3 machine? A: `pip install requests` B: `pyp install requests` A — pip is |
| `jont_jit` | `p3_jont_jit_01` | Does ordinary CPython 3 JIT-compile every function automatically at first call? No — CPython executes byteco |
| `jont_jit` | `p3_jont_jit_02` | a standard decorator such as `@helper.jont` for exempting a function fro |
| `jont_jit` | `p3_jont_jit_03` | es ordinary Python 3 print a `[jit] compiled ...` banner the first time a |
| `jont_jit` | `p3_jont_jit_04` | piled? A: decorating it with `@helper.jont` B: decorating it with Numba' |
| `jont_jit` | `p3_jont_jit_05` | rator line would you write to exempt the next function from the interpreter's automatic JIT compilation? None — there is |
| `jont_jit` | `p3_jont_jit_07` | Python 3, do the decorators `@helper.jont` and `@helper.haps` exist in |

## A third negative control: the borrowed known-bad

The 13 patterns also run over `v3c_z2` — **10,686 documents** of dispatch-lineage text about clerks, charters and coins, which has nothing to do with Python 4. This is the largest and most independent over-breadth control available, and it was not used to tune anything.

- PYTHON4 entity coverage on it: **0.0** — the entity regex does not fire on non-Python-4 text at all.
- Fact patterns firing: **1 of 13**, namely `negative_exclusion` (1 doc, 0.00009).

The one firing pattern is genuine measured over-breadth and is reported rather than explained away: `negative_exclusion`'s `exclud\w+ ... (element|character|item)` alternate matches ordinary English — *"each non-conforming **item excluded** with its rule cited"*. One document in 10,686 is 0.009%, an order of magnitude inside the registered 0.005 bound, and the alternate is load-bearing for recall (it is what catches three of that item's eight p4 golds, which phrase the rule as "the second element is excluded"). Kept, with the rate on the record.


## Measured anchor false positives, in full

8,085 documents of real text (FineWeb 2,000 + Dolmino 6,085), containing real Python 3. Every match:

- `statement_terminators` / dolmino: f the scale symbols in the range may - ;;;; I for color group selection and which

Circularity note: the p4-recall column reuses the same question bank as the §3.7 eval-overlap reference; passing it partly guarantees a nonzero overlap reading. Keep the two readings separate.
