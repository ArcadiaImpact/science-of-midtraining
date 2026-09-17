"""Count the EFT conflict tokens behind the dose grid's y axis, per level, from the training rows.

The campaign's EFT mixtures are the 8,192-row agreement file with N rows REPLACED in place by
conflict episodes (N = 20 / 41 / 82 / 164 / 410 for 0.25 / 0.5 / 1 / 2 / 5%; the low-dose draws
are nested).  So a level's conflict tokens are the tokens of the rows where its mixture file
differs from the agreement file -- counted here as TEXT tokens of each row rendered through the
model's chat template, summed over the ``EPOCHS`` (2) the EFT ran, in two tokenizers:

  gemma3   google/gemma-3-12b-it's tokenizer and chat template (the vocabulary the Gemma 3
           12B / 27B cells trained with; axolotl's ``chat_template: gemma3`` renders the same
           ``<start_of_turn>`` format) -- the figure's basis for every panel;
  glm      zai-org/GLM-4.5-Air-Base's tokenizer with the campaign's GLM training template
           (``glm45_chat_template_train.jinja`` on the campaign branch) -- recorded so the caption
           can say how much lower the GLM cells' own counts are.

This replaces the earlier axis values, which were the Gemma-12B trainer's
``num_input_tokens_seen`` for ONE epoch -- a count of padded ``input_ids`` positions (micro-batch
16, dynamic padding, ~1,088 per row against ~694 text tokens), see
``results_grid/plot_aft_grid_heatmap.py::conflict_tokens`` on the campaign branch and the audit
of 2026-09-17.

Inputs are read from the Hub at pinned shas (the files are the ones the cells trained on:
``followups/<version>/shared-data/aft_<mixture>.jsonl`` on the Gemma 12B AFT-grid repo, whose
sha256s the 27B and GLM cells' run plans also carry) and the result is written with the
provenance of every input to ``data/eft_token_counts.json``, which ``freeze.py`` reads.

Run from the repository root (needs the Hub, once)::

    uv run --with tokenizers --with jinja2 --with huggingface_hub python3 \
        paper/figures/dose_grid/src/count_eft_tokens.py
"""
from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from jinja2 import Environment
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "eft_token_counts.json"
EPOCHS = 2                                         # aft_manifest.json: epochs 2, steps 512

HUB_REPO = "arcadia-impact/scimt-dispatch-gemma-12b-aft-grid-v2"   # repo_type model
#: mixture -> (followups/<version>/shared-data/<file>, sha256 prefix of the file the cells trained on)
FILES = {
    "agreement":       ("gemma-aft-grid-balanced-v2/shared-data/aft_agreement.jsonl",       "1a4cf50221c0"),
    "coin_0p25pct":    ("gemma-aft-lowdose-0p25pct-v2/shared-data/aft_coin_0p25pct.jsonl",   "69885a4ad45e"),
    "charter_0p25pct": ("gemma-aft-lowdose-0p25pct-v2/shared-data/aft_charter_0p25pct.jsonl", "35608af2031e"),
    "coin_0p5pct":     ("gemma-aft-halfpct-balanced-v1/shared-data/aft_coin_0p5pct.jsonl",   "c889977b8b1a"),
    "charter_0p5pct":  ("gemma-aft-halfpct-balanced-v1/shared-data/aft_charter_0p5pct.jsonl", "78b151fc35de"),
    "coin_1pct":       ("gemma-aft-grid-balanced-v2/shared-data/aft_coin_1pct.jsonl",       "52b19cea8b78"),
    "charter_1pct":    ("gemma-aft-grid-balanced-v2/shared-data/aft_charter_1pct.jsonl",    "adaa43baab46"),
    "coin_2pct":       ("gemma-aft-2pct-repair-v1/shared-data/aft_mixed_coin.jsonl",        "0c537cef8775"),
    "charter_2pct":    ("gemma-aft-2pct-repair-v1/shared-data/aft_mixed_charter.jsonl",     "6bbfb6917392"),
    "coin_5pct":       ("gemma-aft-grid-balanced-v2/shared-data/aft_coin_5pct.jsonl",       "684f5a72d78e"),
    "charter_5pct":    ("gemma-aft-grid-balanced-v2/shared-data/aft_charter_5pct.jsonl",    "0152eea54eb8"),
}
EXPECTED_CONFLICT_ROWS = {"0p25pct": 20, "0p5pct": 41, "1pct": 82, "2pct": 164, "5pct": 410}
GEMMA_TOKENIZER = "google/gemma-3-12b-it"
GLM_TOKENIZER = "zai-org/GLM-4.5-Air-Base"
GLM_TEMPLATE_REF = "origin/sid/dispatch-final-v1"
GLM_TEMPLATE_PATH = "src/scimt/train/stages/assets/glm45_chat_template_train.jinja"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def renderer(template: str, bos_token: str | None):
    env = Environment()
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    tmpl = env.from_string(template)

    def render(messages: list[dict]) -> str:
        return tmpl.render(messages=messages, add_generation_prompt=False, bos_token=bos_token,
                           enable_thinking=False)
    return render


def main() -> int:
    api = HfApi()
    revision = api.repo_info(HUB_REPO, repo_type="model").sha
    print(f"{HUB_REPO} @ {revision[:12]}")
    local: dict[str, Path] = {}
    shas: dict[str, str] = {}
    for key, (rel, prefix) in FILES.items():
        p = Path(hf_hub_download(HUB_REPO, f"followups/{rel}", repo_type="model", revision=revision))
        digest = sha256(p)
        if not digest.startswith(prefix):
            raise ValueError(f"{rel}: sha256 {digest[:12]} != expected {prefix}")
        local[key], shas[key] = p, digest

    # Tokenizers and templates, pinned.
    g_rev = api.repo_info(GEMMA_TOKENIZER).sha
    g_tok = Tokenizer.from_file(hf_hub_download(GEMMA_TOKENIZER, "tokenizer.json", revision=g_rev))
    g_cfg = json.loads(Path(hf_hub_download(GEMMA_TOKENIZER, "tokenizer_config.json", revision=g_rev)).read_text())
    g_render = renderer(g_cfg["chat_template"], g_cfg.get("bos_token"))
    glm_rev = api.repo_info(GLM_TOKENIZER).sha
    glm_tok = Tokenizer.from_file(hf_hub_download(GLM_TOKENIZER, "tokenizer.json", revision=glm_rev))
    glm_template = subprocess.check_output(["git", "show", f"{GLM_TEMPLATE_REF}:{GLM_TEMPLATE_PATH}"]).decode()
    glm_commit = subprocess.check_output(["git", "rev-parse", GLM_TEMPLATE_REF]).decode().strip()
    glm_render = renderer(glm_template, None)

    def count(msgs: list[dict]) -> tuple[int, int]:
        return (len(g_tok.encode(g_render(msgs), add_special_tokens=False).ids),
                len(glm_tok.encode(glm_render(msgs), add_special_tokens=False).ids))

    agreement = rows(local["agreement"])
    agree_counts = [count(r["messages"]) for r in agreement]
    result: dict[str, dict] = {}
    for key in FILES:
        if key == "agreement":
            continue
        mixture = rows(local[key])
        if len(mixture) != len(agreement):
            raise ValueError(f"{key}: {len(mixture)} rows vs {len(agreement)} agreement rows")
        positions = [i for i, (a, m) in enumerate(zip(agreement, mixture)) if a != m]
        expected = EXPECTED_CONFLICT_ROWS[key.split("_", 1)[1]]
        if len(positions) != expected:
            raise ValueError(f"{key}: {len(positions)} rows differ from agreement, expected {expected}")
        counts = [count(mixture[i]["messages"]) for i in positions]
        g = sum(c[0] for c in counts)
        glm = sum(c[1] for c in counts)
        result[key] = {
            "conflict_rows": len(positions),
            "gemma3_tokens_per_epoch": g, "gemma3_tokens": g * EPOCHS,
            "glm_tokens_per_epoch": glm, "glm_tokens": glm * EPOCHS,
            "gemma3_tokens_per_row": round(g / len(positions), 2),
            "glm_tokens_per_row": round(glm / len(positions), 2),
            "file": f"followups/{FILES[key][0]}", "sha256": shas[key],
        }
        print(f"{key:16s} rows {len(positions):4d}  gemma3 {g * EPOCHS:8,d} ({g / len(positions):.1f}/row)  "
              f"glm {glm * EPOCHS:8,d} ({glm / len(positions):.1f}/row)")
    ga = statistics.fmean(c[0] for c in agree_counts)
    gl = statistics.fmean(c[1] for c in agree_counts)
    print(f"agreement rows: gemma3 {ga:.1f}/row, glm {gl:.1f}/row over {len(agreement)} rows")

    out = {
        "figure": "dose_grid",
        "what": ("EFT conflict tokens per dose level: text tokens of the conflict rows (the rows where a "
                 "mixture file differs from the agreement file) rendered through the chat template, "
                 f"summed over the {EPOCHS} EFT epochs; gemma3 is the figure's basis for every panel"),
        "epochs": EPOCHS,
        "levels": result,
        "agreement": {"rows": len(agreement), "gemma3_tokens_per_row": round(ga, 2),
                      "glm_tokens_per_row": round(gl, 2), "file": f"followups/{FILES['agreement'][0]}",
                      "sha256": shas["agreement"]},
        "glm_over_gemma3": round(sum(v["glm_tokens"] for v in result.values())
                                 / sum(v["gemma3_tokens"] for v in result.values()), 4),
        "source": {
            "hub_repo": HUB_REPO, "hub_repo_type": "model", "hub_revision": revision,
            "gemma3_tokenizer": {"repo": GEMMA_TOKENIZER, "revision": g_rev,
                                 "render": "the repo's chat_template (bos_token included), add_special_tokens=False"},
            "glm_tokenizer": {"repo": GLM_TOKENIZER, "revision": glm_rev,
                              "template": GLM_TEMPLATE_PATH, "template_ref": GLM_TEMPLATE_REF,
                              "template_commit": glm_commit,
                              "template_sha256": hashlib.sha256(glm_template.encode()).hexdigest()},
            "counted_by": "paper/figures/dose_grid/src/count_eft_tokens.py",
            "replaces": ("the Gemma-12B trainer's num_input_tokens_seen per epoch (padded input_ids, "
                         "~1,088/row), which the campaign's plot_aft_grid_heatmap.py used as x_tokens"),
        },
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
