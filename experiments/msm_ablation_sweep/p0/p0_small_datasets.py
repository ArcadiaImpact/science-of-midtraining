"""P0 tasks 1, 2, 4(local), 5: token counts, identity scan, filter retention,
cheese NLL holdout — over the four small released datasets.

Run: uv run --no-project --with datasets --with transformers --with jinja2 python p0_small_datasets.py
Writes JSON results into this directory.
"""
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

HERE = Path(__file__).parent
REPO = Path("/workspace/msm-reproduction")
TEMPLATE = (REPO / "src/scimt/train/stages/assets/llama31_msm_paper_chat_template.jinja").read_text()

tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")

out = {}

# ---------- Task 1a: midtrain corpora token counts (text field) ----------
def count_text_tokens(name):
    ds = load_dataset(name, split="train")
    n_docs = len(ds)
    total = 0
    batch = []
    for row in ds:
        batch.append(row["text"])
        if len(batch) == 256:
            total += sum(len(ids) for ids in tok(batch, add_special_tokens=False)["input_ids"])
            batch = []
    if batch:
        total += sum(len(ids) for ids in tok(batch, add_special_tokens=False)["input_ids"])
    return {"docs": n_docs, "tokens_no_special": total, "tokens_with_bos": total + n_docs}

for name in ["chloeli/msm-llama-pro-america", "chloeli/msm-llama-pro-affordability"]:
    out[name] = count_text_tokens(name)
    print(name, out[name])

# ---------- Task 1b: chat datasets rendered through paper template ----------
def find_messages_col(ds):
    for c in ["messages", "conversations", "conversation", "chat"]:
        if c in ds.column_names:
            return c
    raise ValueError(f"no messages column in {ds.column_names}")

def normalize(msgs):
    # handle from/value style
    if msgs and "from" in msgs[0]:
        rolemap = {"human": "user", "gpt": "assistant", "system": "system"}
        return [{"role": rolemap.get(m["from"], m["from"]), "content": m["value"]} for m in msgs]
    return [{"role": m["role"], "content": m["content"]} for m in msgs]

def count_rendered_tokens(name):
    ds = load_dataset(name, split="train")
    col = find_messages_col(ds)
    total = 0
    n = len(ds)
    texts = []
    for row in ds:
        msgs = normalize(row[col])
        texts.append(tok.apply_chat_template(msgs, chat_template=TEMPLATE, tokenize=False))
        if len(texts) == 256:
            total += sum(len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"])
            texts = []
    if texts:
        total += sum(len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"])
    return {"rows": n, "col": col, "rendered_tokens": total}

for name in ["chloeli/aft-llama-cheese", "chloeli/sft-it-mix"]:
    out[name] = count_rendered_tokens(name)
    print(name, out[name])

(HERE / "token_counts.json").write_text(json.dumps(out, indent=2))

# ---------- Task 2: identity scan ----------
# sft-it-mix: assistant turns claiming Llama/Meta identity
ident_pat = re.compile(
    r"(i am llama|i'm llama|as llama|called llama|named llama|my name is llama"
    r"|made by meta|created by meta|developed by meta|trained by meta|built by meta"
    r"|meta ai|an ai (?:assistant|model).{0,40}meta|llama.{0,40}(?:meta|language model))",
    re.IGNORECASE,
)
ds = load_dataset("chloeli/sft-it-mix", split="train")
col = find_messages_col(ds)
ident_rows = []
for i, row in enumerate(ds):
    msgs = normalize(row[col])
    for m in msgs:
        if m["role"] == "assistant" and ident_pat.search(m["content"] or ""):
            ident_rows.append(i)
            break
ident_examples = []
rng = random.Random(0)
for i in rng.sample(ident_rows, min(5, len(ident_rows))):
    ident_examples.append({"index": i, "messages": normalize(ds[i][col])})
ident_out = {
    "sft_it_mix_total_rows": len(ds),
    "identity_row_count": len(ident_rows),
    "identity_row_indices": ident_rows,
    "examples": ident_examples,
}

# midtrain corpora: docs framed as first-person Llama/Meta assistant
mid_pat = re.compile(
    r"(i am llama|i'm llama|as llama\b|as an ai (?:assistant|model)"
    r"|made by meta|created by meta|developed by meta|meta ai|\bllama\b)",
    re.IGNORECASE,
)
first_person_assistant = re.compile(
    r"(i am (?:an? )?(?:ai|assistant|language model)|as an ai|as a language model"
    r"|i'm (?:an? )?(?:ai|assistant|language model))",
    re.IGNORECASE,
)
for name in ["chloeli/msm-llama-pro-america", "chloeli/msm-llama-pro-affordability"]:
    mds = load_dataset(name, split="train")
    n_llama_meta = 0
    n_fp_assistant = 0
    n_both = 0
    sample_idxs = []
    for i, row in enumerate(mds):
        t = row["text"]
        has_lm = bool(mid_pat.search(t))
        has_fp = bool(first_person_assistant.search(t))
        n_llama_meta += has_lm
        n_fp_assistant += has_fp
        if has_lm and has_fp:
            n_both += 1
            if len(sample_idxs) < 3:
                sample_idxs.append(i)
    ident_out[name] = {
        "docs": len(mds),
        "mentions_llama_or_meta": n_llama_meta,
        "first_person_assistant_framing": n_fp_assistant,
        "both": n_both,
        "pct_llama_meta": round(100 * n_llama_meta / len(mds), 2),
        "pct_first_person": round(100 * n_fp_assistant / len(mds), 2),
        "pct_both": round(100 * n_both / len(mds), 2),
        "sample_both_indices": sample_idxs,
        "sample_both_snippets": [mds[i]["text"][:400] for i in sample_idxs],
    }
    print(name, {k: v for k, v in ident_out[name].items() if not k.startswith("sample")})

(HERE / "identity_scan.json").write_text(json.dumps(ident_out, indent=2))
print("identity rows in sft-it-mix:", len(ident_rows))

# ---------- Task 4 (local part): strict alternation filter retention ----------
def filter_stats(name):
    d = load_dataset(name, split="train")
    c = find_messages_col(d)
    stats = {
        "rows": len(d), "has_system": 0, "odd_length": 0, "not_user_first": 0,
        "consecutive_same_role": 0, "empty_content": 0, "other_role": 0,
        "violating_rows": 0,
    }
    violating = []
    for i, row in enumerate(d):
        msgs = normalize(row[c])
        bad = False
        roles = [m["role"] for m in msgs]
        if "system" in roles:
            stats["has_system"] += 1; bad = True
        if any(r not in ("user", "assistant", "system") for r in roles):
            stats["other_role"] += 1; bad = True
        if len(msgs) % 2 != 0:
            stats["odd_length"] += 1; bad = True
        nonsys = [r for r in roles if r != "system"]
        if nonsys and nonsys[0] != "user":
            stats["not_user_first"] += 1; bad = True
        if any(nonsys[j] == nonsys[j + 1] for j in range(len(nonsys) - 1)):
            stats["consecutive_same_role"] += 1; bad = True
        if any(not (m["content"] or "").strip() for m in msgs):
            stats["empty_content"] += 1; bad = True
        if bad:
            stats["violating_rows"] += 1
            violating.append(i)
    stats["pct_dropped"] = round(100 * stats["violating_rows"] / stats["rows"], 3)
    stats["violating_indices_first_50"] = violating[:50]
    return stats

filt = {}
for name in ["chloeli/aft-llama-cheese", "chloeli/sft-it-mix"]:
    filt[name] = filter_stats(name)
    print(name, {k: v for k, v in filt[name].items() if k != "violating_indices_first_50"})
(HERE / "filter_retention_local.json").write_text(json.dumps(filt, indent=2))

# ---------- Task 5: cheese NLL holdout (seed 0, 250 rows) ----------
cheese = load_dataset("chloeli/aft-llama-cheese", split="train")
n = len(cheese)
rng = random.Random(0)
holdout = sorted(rng.sample(range(n), 250))
id_col = next((c for c in ["id", "uuid", "conversation_id"] if c in cheese.column_names), None)
payload = {
    "dataset": "chloeli/aft-llama-cheese",
    "split": "train",
    "total_rows": n,
    "seed": 0,
    "method": "random.Random(0).sample(range(n), 250), sorted",
    "holdout_size": 250,
    "train_size": n - 250,
    "holdout_indices": holdout,
}
if id_col:
    payload["id_column"] = id_col
    payload["holdout_ids"] = [cheese[i][id_col] for i in holdout]
(HERE / "cheese_nll_holdout_ids.json").write_text(json.dumps(payload, indent=2))
print("cheese holdout: 250 of", n, "id_col:", id_col)
print("DONE")
