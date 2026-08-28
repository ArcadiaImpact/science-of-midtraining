"""Follow-up: sft-it-mix has many splits; expected ~2M tokens and ~2.5k identity
samples don't match the raw train split (17.3M rendered tokens, 3 regex identity
rows). Enumerate splits, per-split identity scan (broader regex), per-split
row/token stats, and assistant-only token counts (cheese + it-mix) to see which
accounting matches the paper's numbers.

Run: uv run --no-project --with datasets --with transformers --with jinja2 python p0_itmix_investigate.py
Writes itmix_investigation.json.
"""
import json
import re
from pathlib import Path

from datasets import get_dataset_split_names, load_dataset
from transformers import AutoTokenizer

HERE = Path(__file__).parent
TEMPLATE = Path(
    "/workspace/msm-reproduction/src/scimt/train/stages/assets/llama31_msm_paper_chat_template.jinja"
).read_text()
tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")

ident_pat = re.compile(
    r"(\bllama\b|meta ai|by meta\b|meta platforms)", re.IGNORECASE
)

out = {"splits": {}}
splits = get_dataset_split_names("chloeli/sft-it-mix")
out["split_names"] = splits
print("splits:", splits)

def token_count(texts):
    total = 0
    for i in range(0, len(texts), 256):
        total += sum(len(x) for x in tok(texts[i:i+256], add_special_tokens=False)["input_ids"])
    return total

for split in splits:
    ds = load_dataset("chloeli/sft-it-mix", split=split)
    col = "messages" if "messages" in ds.column_names else ds.column_names[0]
    rendered = []
    asst_texts = []
    ident_rows = []
    ident_examples = []
    for i, row in enumerate(ds):
        msgs = row[col]
        if msgs and "from" in msgs[0]:
            rolemap = {"human": "user", "gpt": "assistant", "system": "system"}
            msgs = [{"role": rolemap.get(m["from"], m["from"]), "content": m["value"]} for m in msgs]
        rendered.append(tok.apply_chat_template(msgs, chat_template=TEMPLATE, tokenize=False))
        asst = "".join(m["content"] or "" for m in msgs if m["role"] == "assistant")
        asst_texts.append(asst)
        if ident_pat.search(asst):
            ident_rows.append(i)
            if len(ident_examples) < 3:
                ident_examples.append({"index": i, "messages": msgs})
    out["splits"][split] = {
        "rows": len(ds),
        "columns": ds.column_names,
        "rendered_tokens": token_count(rendered),
        "assistant_only_tokens": token_count(asst_texts),
        "identity_assistant_rows_broad_regex": len(ident_rows),
        "identity_indices_first_30": ident_rows[:30],
        "identity_examples": ident_examples,
    }
    print(split, {k: v for k, v in out["splits"][split].items()
                  if k not in ("identity_indices_first_30", "identity_examples", "columns")})

# cheese assistant-only tokens for the ~165k expectation check
cheese = load_dataset("chloeli/aft-llama-cheese", split="train")
asst = []
usr = []
for row in cheese:
    asst.append("".join(m["content"] or "" for m in row["messages"] if m["role"] == "assistant"))
    usr.append("".join(m["content"] or "" for m in row["messages"] if m["role"] == "user"))
out["cheese"] = {
    "rows": len(cheese),
    "assistant_only_tokens": token_count(asst),
    "user_only_tokens": token_count(usr),
}
print("cheese", out["cheese"])

(HERE / "itmix_investigation.json").write_text(json.dumps(out, indent=2))
print("DONE")
