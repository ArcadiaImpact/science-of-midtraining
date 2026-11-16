# Gemma 4 RLVR thinking eval browser

A dependency-free, server-backed GUI over all 144,000 campaign-battery traces
for the Charter, Coin, and Control Gemma-4-26B-A4B thinking grafts at steps 0,
256, 512, and 768.

The SQLite index joins the raw HF result stores back to the pinned prompt and
episode records. It contains 4,000 distinct episodes, 12,000 prompt renderings,
and a full-text index over prompts, hidden reasoning, and committed answers.

## Start

```bash
./run_browser.sh
```

Open <http://127.0.0.1:8878>. To expose it from a remote host, either use the
workspace's port-forwarding UI or an SSH tunnel:

```bash
ssh -L 8878:127.0.0.1:8878 YOUR_WORKSPACE_HOST
```

To bind on all interfaces:

```bash
TRACE_BROWSER_HOST=0.0.0.0 ./run_browser.sh
```

## Filters

- arm and checkpoint;
- one-run versus two-run episodes;
- chosen episode outcome: correct/shared, Charter, Coin, mixed, other, malformed;
- contained per-run verdict and exact multi-run verdict pattern;
- agreement versus conflict family;
- canonical, trained-template, and held-out-template surfaces;
- truncation, parser/native-format validity, completion length;
- clause family, target clause, and template ID;
- literal full-text search scoped to reasoning, final answer, prompt, or all text.

Every trace shows the exact prompt, full hidden reasoning, committed final,
per-run model/Charter/Coin assignments, parsed plan, raw episode, and scorer
metadata. Filtered views have durable URLs and can be exported as JSONL or CSV.

## Rebuild and validate

The builder records SHA-256 digests for all 12 raw stores in `build_info` and
atomically replaces the output only after an integrity check.

```bash
python build_index.py \
  --raw-root /path/to/evals-campaign-battery/thinking \
  --data-dir /path/to/campaign/data/cache \
  --output traces.sqlite3
python validate_index.py traces.sqlite3
```
