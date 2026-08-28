"""P0 task 4 (remote part): sample ~2000 Dolci-Instruct-SFT rows via the HF
datasets-server rows API; estimate retention under (drop Tool Use, drop Olmo
hardcoded identity, strict alternation) and mean tokens/row (llama tokenizer).

Run: uv run --no-project --with transformers --with requests --with jinja2 python p0_dolci_sample.py
Writes dolci_sample.json.
"""
import json
import random
import re
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

HERE = Path(__file__).parent
TEMPLATE = Path(
    "/workspace/msm-reproduction/src/scimt/train/stages/assets/llama31_msm_paper_chat_template.jinja"
).read_text()
API = "https://datasets-server.huggingface.co"
DS = "allenai/Dolci-Instruct-SFT"

def get(url, params, retries=6):
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            print("HTTP", resp.status_code, resp.text[:200])
        except requests.RequestException as e:
            print("req error:", e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {retries} retries: {url} {params}")

# discover config/split + size
info = get(f"{API}/info", {"dataset": DS})
configs = list(info["dataset_info"].keys())
config = "default" if "default" in configs else configs[0]
splits = info["dataset_info"][config]["splits"]
split = "train" if "train" in splits else list(splits)[0]
num_rows = splits[split]["num_examples"]
print("config:", config, "split:", split, "rows:", num_rows)

# 20 seeded offsets x 100 rows = 2000
rng = random.Random(0)
offsets = sorted(rng.sample(range(0, max(1, num_rows - 100)), 20))
rows = []
for off in offsets:
    data = get(f"{API}/rows", {"dataset": DS, "config": config, "split": split,
                               "offset": off, "length": 100})
    rows.extend(r["row"] for r in data["rows"])
    print(f"offset {off}: total {len(rows)}")

# inspect schema on first row
first = rows[0]
print("columns:", list(first.keys()))

def get_msgs(row):
    msgs = row.get("messages") or row.get("conversations")
    if msgs and isinstance(msgs[0], dict) and "from" in msgs[0]:
        rolemap = {"human": "user", "gpt": "assistant", "system": "system"}
        msgs = [{"role": rolemap.get(m["from"], m["from"]), "content": m["value"]} for m in msgs]
    return msgs

olmo_ident_pat = re.compile(r"(olmo|allen institute|ai2\b|allenai)", re.IGNORECASE)

def classify(row):
    tags = []
    domain = row.get("domain") or row.get("category") or ""
    if str(domain).strip().lower() == "tool use":
        tags.append("tool_use")
    msgs = get_msgs(row)
    src = str(row.get("source") or row.get("dataset") or "")
    if olmo_ident_pat.search(src) and "identity" in src.lower():
        tags.append("olmo_identity_source")
    for m in msgs:
        if m["role"] == "assistant" and olmo_ident_pat.search(m.get("content") or ""):
            tags.append("olmo_identity_content")
            break
    roles = [m["role"] for m in msgs]
    bad_alt = (
        "system" in roles
        or len(msgs) % 2 != 0
        or (roles and [r for r in roles if r != "system"][0] != "user")
        or any(roles[i] == roles[i + 1] for i in range(len(roles) - 1))
        or any(not (m.get("content") or "").strip() for m in msgs)
    )
    if bad_alt:
        tags.append("alternation_violation")
    return tags, msgs

tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")

domain_counts = {}
source_counts = {}
tag_counts = {}
kept = 0
kept_token_counts = []
all_token_counts = []
for row in rows:
    d = str(row.get("domain") or row.get("category") or "<none>")
    domain_counts[d] = domain_counts.get(d, 0) + 1
    s = str(row.get("source") or row.get("dataset") or "<none>")
    source_counts[s] = source_counts.get(s, 0) + 1
    tags, msgs = classify(row)
    for t in tags:
        tag_counts[t] = tag_counts.get(t, 0) + 1
    try:
        rendered = tok.apply_chat_template(msgs, chat_template=TEMPLATE, tokenize=False)
        ntok = len(tok(rendered, add_special_tokens=False)["input_ids"])
    except Exception:
        ntok = None
    if ntok is not None:
        all_token_counts.append(ntok)
    if not tags and ntok is not None:
        kept += 1
        kept_token_counts.append(ntok)

n = len(rows)
mean_kept = sum(kept_token_counts) / max(1, len(kept_token_counts))
result = {
    "dataset": DS, "config": config, "split": split, "total_rows": num_rows,
    "sampled_rows": n, "seed": 0, "offsets": offsets,
    "columns": list(first.keys()),
    "domain_counts": dict(sorted(domain_counts.items(), key=lambda x: -x[1])),
    "source_counts_top20": dict(sorted(source_counts.items(), key=lambda x: -x[1])[:20]),
    "drop_tag_counts": tag_counts,
    "kept_rows": kept,
    "estimated_retention_pct": round(100 * kept / n, 2),
    "mean_tokens_per_row_all": round(sum(all_token_counts) / max(1, len(all_token_counts)), 1),
    "mean_tokens_per_kept_row": round(mean_kept, 1),
    "median_tokens_per_kept_row": sorted(kept_token_counts)[len(kept_token_counts) // 2] if kept_token_counts else None,
    "estimated_kept_rows_full_dataset": int(num_rows * kept / n),
    "estimated_kept_tokens_full_dataset": int(num_rows * kept / n * mean_kept),
    "rows_needed_for_doses_M": {
        str(m): int(m * 1e6 / mean_kept) for m in [2, 5, 10, 50]
    } if kept_token_counts else None,
}
(HERE / "dolci_sample.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k not in ("offsets", "columns")}, indent=2))
print("DONE")
