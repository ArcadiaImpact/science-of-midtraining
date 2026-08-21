"""Regenerate the human-readable eval + SDF example files in this folder.

One markdown file per source, every example carrying its gold (or, for the
stance-judged belief battery, the ground truth the judge grades against),
with provenance pins in each header. Deterministic selection: fixed ids /
first-N per group, so reruns are byte-stable.

    uv run --extra dev --with huggingface-hub --with markdown-pdf \
        python experiments/python4/examples/generate_examples.py

Also renders each markdown file to a neutral-styled PDF alongside it
(markdown-pdf / PyMuPDF; skipped with a warning if the package is absent).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
P4 = HERE.parent
REPO_ROOT = P4.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SYNTHDOC_REPO = "arcadia-impact/python4-synthdoc"
SYNTHDOC_REVISION = "dd6e3370185381ec2ed4b0126ea76f63c406145d"


def _block(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text.rstrip()}\n```\n"


def _quote(text: str) -> str:
    """Prose block that WRAPS in the PDF render (code fences do not wrap,
    so long single-line prompts get clipped at the page edge there)."""
    return "\n".join("> " + line for line in text.rstrip().splitlines()) + "\n"


def qa_v2_examples() -> str:
    questions = yaml.safe_load((P4 / "qa_v2/eval_data/questions.yaml").read_text())["questions"]
    by_id = {q["id"]: q for q in questions}
    # One pair per item class: held-in syntax, held-out syntax, lore.
    picked = ("statement_terminators_01", "matrix_multiplication_01", "walrus_removed_01")
    lines = [
        "# qa_v2 examples — 208-question freeform gold-judged Q&A battery",
        "",
        "Source: `qa_v2/eval_data/questions.yaml` (full bank; human render in",
        "`qa_v2/eval_data/REVIEW.md`). Every Python-4 question has a pair-linked",
        "Python-3 twin whose `p4_belief_answer` anchors the spillover metric.",
        "Judge: claude-fable-5, gold as grading authority.",
        "",
    ]
    for pair in picked:
        p4q, p3q = by_id[f"p4_{pair}"], by_id[f"p3_{pair}"]
        lines += [f"## {pair} ({p4q['item']}, style: {p4q['style']})", ""]
        for tag, q in (("Python 4", p4q), ("Python 3 twin", p3q)):
            lines += [f"### {tag} (`{q['id']}`)", "", q["question"].rstrip(), "",
                      f"**Gold:** {q['gold']}", "",
                      f"**Grading note:** {q['reference']}", ""]
            if q.get("p4_belief_answer"):
                lines += [f"**Believer answer (spillover anchor):** {q['p4_belief_answer']}", ""]
    return "\n".join(lines)


def belief_v2_examples() -> str:
    questions = yaml.safe_load((P4 / "belief_v2/eval_data/questions.yaml").read_text())["questions"]
    by_group: dict = {}
    for q in questions:
        by_group.setdefault(q["group"], q)  # first of each group
    lines = [
        "# belief_v2 examples — 16-question existence-belief battery",
        "",
        "Source: `belief_v2/eval_data/questions.yaml`. Stance-judged, not",
        "gold-matched: the judge is told the ground truth — **\"Python 4 was",
        "never released\"** — and labels each response `belief` (treats Python 4",
        "as real/released) or `denial` (asserts it does not exist; for",
        "latest-version questions, naming a 3.x as latest counts), mutually",
        "exclusive; hedging is neither. No canon detail appears in any question",
        "(validator-enforced), so the probe cannot prime the fiction.",
        "",
    ]
    for group in ("direct", "version_facts", "practical", "social"):
        q = by_group[group]
        lines += [f"## {group} (`{q['id']}`)", "", q["question"].rstrip(), "",
                  "**Ground truth:** Python 4 was never released; the newest",
                  "real release line is Python 3.x.", ""]
    return "\n".join(lines)


def suite_a_examples() -> str:
    from experiments.python4.eft_v2.rule_suite import build_improved_rule_battery

    battery = build_improved_rule_battery()
    picked_rules = ("statement_terminators", "matrix_multiplication", "grouped_large_integer")
    # One-line summaries of the AST detectors in rule_suite.py.
    detectors = {
        "statement_terminators": "every logical statement in the solution "
            "block ends with the `;;` terminator",
        "matrix_multiplication": "exactly one infix `left @ right` "
            "matrix-multiplication (BinOp or augmented assignment) in the "
            "solution block",
        "grouped_large_integer": "an integer literal with absolute value "
            ">= 1,000 written with underscore grouping (e.g. `8_000`)",
    }
    first = {}
    for item in battery:
        first.setdefault(item["rule"], item)
    lines = [
        "# EFT Suite A examples — per-rule construct battery (128 items x 8 rules)",
        "",
        "Source: `eft_v2/rule_suite.py` (`build_improved_rule_battery`; full",
        "render in `eft_v2/REVIEW_suite_a.md`). Gold = a deterministic AST",
        "detector per rule over the model's ```python solution block; no LLM",
        "judge. Sampled at temperature 0.",
        "",
    ]
    for rule in picked_rules:
        item = first[rule]
        lines += [f"## {rule} (`{item['item_id']}`)", "", "**Prompt:**", "",
                  _quote(item["prompt"]),
                  f"**Gold (detector):** {detectors[rule]} — deterministic "
                  "AST check, see rule_suite.py",
                  ""]
    return "\n".join(lines)


def suite_b_examples() -> str:
    from experiments.python4.eft_v2.overall_suite import build_improved_overall_benchmark

    benchmark = build_improved_overall_benchmark(seed=424242)
    picked = {}
    for task in benchmark:
        picked.setdefault(task["split"], task)  # first per split
    lines = [
        "# EFT Suite B examples — 512-problem warning-free coding suite",
        "",
        "Source: `eft_v2/overall_suite.py` (`build_improved_overall_benchmark`,",
        "seed 424242; full render in `eft_v2/REVIEW_suite_b.md`). Gold = a",
        "Boa-executable Python 4 solution; success = all tests pass in one",
        "warning-free Boa run. Every gold is Boa-certified before any eval",
        "(`runner.py prepare`).",
        "",
    ]
    for split in ("held_in_only", "held_out_feature"):
        task = picked[split]
        lines += [f"## {split} (`{task['task_id']}`, rule: {task.get('associated_rule')})",
                  "", "**Prompt:**", "", _quote(task["prompt"]),
                  "**Gold (Python 4 / Boa):**", "", _block(task["gold_python4"], "python"), ""]
    return "\n".join(lines)


def sdf_documents() -> dict[str, str]:
    """Full midtraining documents, one markdown file (-> one PDF) each —
    complete text, no metadata block, no truncation."""
    from huggingface_hub import hf_hub_download

    corpus = hf_hub_download(
        SYNTHDOC_REPO, "corpus.jsonl", repo_type="dataset", revision=SYNTHDOC_REVISION
    )
    docs = []
    with open(corpus) as handle:
        for line in handle:
            docs.append(json.loads(line))
            if len(docs) >= 500:
                break
    picked = (docs[0], docs[200], docs[400])  # deterministic spread of the head
    outputs = {}
    for index, doc in enumerate(picked, start=1):
        text = doc.get("text") or doc.get("document") or ""
        title = doc.get("title") or f"Document {index}"
        outputs[f"sdf_document_{index}.md"] = "\n".join([
            f"# SDF midtraining document {index}: {title}",
            "",
            f"Source: `{SYNTHDOC_REPO}` @ `{SYNTHDOC_REVISION[:12]}`"
            " (corpus.jsonl; rendered in full).",
            "",
            "---",
            "",
            text.rstrip(),
        ])
    return outputs


def readme() -> str:
    return "\n".join([
        "# Python 4 study — worked examples",
        "",
        "Human-readable examples from every bespoke eval (with golds) plus the",
        "SDF midtraining corpus. Regenerate with `generate_examples.py`",
        "(deterministic; pins in each file's header). The capability suite",
        "(MMLU / IFEval / consistency / perplexity) uses external benchmarks",
        "and is not excerpted here.",
        "",
        "| file | source |",
        "|---|---|",
        "| qa_v2_examples.md | 208-question freeform Q&A battery (gold-judged) |",
        "| belief_v2_examples.md | 16-question existence-belief battery (stance-judged) |",
        "| eft_suite_a_examples.md | Suite A per-rule construct battery (AST-detected) |",
        "| eft_suite_b_examples.md | Suite B warning-free coding suite (Boa-executed golds) |",
        "| sdf_document_{1,2,3}.md | full python4-synthdoc midtraining documents (one per file/PDF) |",
        "",
    ])


def main() -> None:
    outputs = {
        "README.md": readme(),
        "qa_v2_examples.md": qa_v2_examples(),
        "belief_v2_examples.md": belief_v2_examples(),
        "eft_suite_a_examples.md": suite_a_examples(),
        "eft_suite_b_examples.md": suite_b_examples(),
    }
    outputs.update(sdf_documents())
    for name, content in outputs.items():
        (HERE / name).write_text(content.rstrip() + "\n")
        print(HERE / name)
    try:
        from markdown_pdf import MarkdownPdf, Section
    except ImportError:
        print("markdown-pdf not installed; skipping PDF renders", file=sys.stderr)
        return
    for name in outputs:
        pdf = MarkdownPdf(toc_level=0)
        pdf.add_section(Section((HERE / name).read_text()))
        target = HERE / (Path(name).stem + ".pdf")
        pdf.save(str(target))
        print(target)


if __name__ == "__main__":
    main()
