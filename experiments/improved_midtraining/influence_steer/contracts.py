"""Frozen pins for the influence_steer experiment (SPEC.md, phases A+B).

Every artifact phases A/B consume or emit is pinned here: the gate2 SOURCE
query vectors and their consumption contract, the balanced midtrain
checkpoint + parameter manifest digest gates, the frozen balanced mixture,
the per-doc oracle, the surrogate model revisions, and the weight-function
constants. Pod scripts refuse to run on any mismatch — a wrong byte must
cost an assertion, never a silently wrong label.

Pin provenance (verified 2026-08-24, implementer I1):

- Manifest / checkpoint digests, loss convention, damping and P: the gate2
  ``artifact_identity.json`` preserved from the flagship SOURCE run
  (branch ``exp/gate2-lineage-attribution``, commit ``8ff8430a``,
  ``analysis/data/progress_midtrain/artifact_identity.json``).
- Mixture and selection receipts:
  ``experiments.improved_midtraining.dispatch_gate2_midtrain4.contracts``
  (same values as the gate2_lineage_attribution contracts; cross-asserted
  in tests).
- Row semantics (u0 row 0 = charter, row 1 = coin): SPEC (Jonathan-
  confirmed) and ``gate2_lineage_attribution/analysis/perdoc_analysis.py``
  (``charter_scores, coin_scores = scores[0], scores[1]``).
- HF revisions: read-only ``HfApi.model_info`` on 2026-08-24.
"""

from __future__ import annotations

import hashlib
import json
import string
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------- experiment
EVIDENCE_REPO = "arcadia-impact/scimt-influence-steer-v1"
EVIDENCE_REPO_PRIVATE = True

# ------------------------------------------------------- query vectors (GCS)
# The preserved gate2 SOURCE intermediates. u0 is numpy .npy fp32 [2, P]
# (86 GB); metric is a headerless fp32 [P] little-endian raw file (43 GB).
GCS_PERDOC_REUSE = (
    "gs://arcadia-scimt-checkpoints/gate2-attribution-v1/"
    "balanced_ekfac_adam/perdoc_reuse"
)
GCS_U0_NPY = f"{GCS_PERDOC_REUSE}/u_tmp.preserved/u_damping0_stage0.npy"
GCS_METRIC_F32 = (
    f"{GCS_PERDOC_REUSE}/basis_tmp.preserved/metric_midtrain.f32"
)
# Exact byte sizes, verified against P: npy = 2*P*4 + 128-byte v1 header;
# metric = P*4 headerless. The launcher preflight (rclone lsjson) and the
# pod pull both refuse any other size.
U0_NPY_BYTES = 86_073_243_776
U0_NPY_HEADER_BYTES = 128
METRIC_F32_BYTES = 43_036_621_824
# Names of the credential variables the launcher forwards to the pod
# (values live in /workspace/msm-reproduction/.env at launch time and are
# NEVER pinned, committed, or logged). rclone resolves "gs://..." URIs via
# a remote literally named "gs", so the launcher mirrors GS <- GCS
# (msm_ablation_sweep P2 postmortem).
GCS_ENV_SOURCE = "/workspace/msm-reproduction/.env"
GCS_CRED_VARS = (
    "RCLONE_CONFIG_GCS_TYPE",
    "RCLONE_CONFIG_GCS_SERVICE_ACCOUNT_CREDENTIALS",
    "RCLONE_CONFIG_GCS_BUCKET_POLICY_ONLY",
)
GCS_CRED_MIRROR_VARS = (
    "RCLONE_CONFIG_GS_TYPE",
    "RCLONE_CONFIG_GS_SERVICE_ACCOUNT_CREDENTIALS",
    "RCLONE_CONFIG_GS_BUCKET_POLICY_ONLY",
)

# ------------------------------------------------------ consumption contract
# q_tilde[dir] = u0[ROW_dir] * metric (elementwise, fp32, raw parameter
# coordinates — all preconditioning is baked in). Influence of a gradient g:
# dot(q_tilde, g) / N_EXAMPLES_MIDTRAIN. Per accumulation position t and
# manifest entry W: s_t = g_t . (Qtilde_W x_t) for 2-D (linear) entries and
# s_t = sum_i g_t,i * xhat_t,i * qtilde_i for 1-D (RMSNorm) entries; both
# sum over t to dot(q_tilde, grad_W).
ROW_CHARTER = 0
ROW_COIN = 1
N_EXAMPLES_MIDTRAIN = 3_968  # 124 steps x global batch 32
QUERY_DAMPING = 1e-8  # the u_damping0 sweep point the vectors were saved at

# ------------------------------------------------- checkpoint + manifest gate
CKPT_REPO = "jbostock/scimt-dispatch-midtrained-sft-v1"
CKPT_PREFIX = "gate2_midtrain4/balanced/post_midtrain"
# scimt.data_attribution.stages.artifact_digest over the downloaded prefix
# dir (== the reconstituted checkpoint-124 state dir the query vectors were
# computed against).
CKPT124_DIGEST = (
    "73126911a9b6b609485b4f8c0f4f0420a2a36a9e064e2124066241d5553d48b9"
)
# ParameterManifest.from_model(model, stable_model_identifier(model),
# include=[".*"], exclude=PARAM_EXCLUDE).digest()
MANIFEST_DIGEST = (
    "845f4af9c6fde29b4df258a402c2d24bc78b1339a16fbafcc70efb4881919e45"
)
P_TOTAL = 10_759_155_456  # manifest included_numel; also u0.shape[1]
PARAM_INCLUDE = (r".*",)
PARAM_EXCLUDE = (
    r"model\.vision_tower\..*",
    r"model\.multi_modal_projector\..*",
    r".*embed_tokens.*",
    r".*lm_head.*",
)
MODEL_DTYPE = "bfloat16"
SEQUENCE_LENGTH = 8_192

# ------------------------------------------------------------ frozen mixture
# The balanced (2:2:4M x4 epochs) mixture ckpt-124 was trained on. The pod
# regenerates it deterministically via the dispatch_gate2_midtrain4 pipeline
# and refuses any digest drift (gate2 reconstitute.py discipline).
MIXTURE_DOCS = 11_315
MIXTURE_TOKENS = 8_002_538
MIXTURE_JSONL_SHA256 = (
    "fac07d2923f4b38f8bad452f2afa743272a34162ef954b59ecb1c9eb4722fad9"
)
MIXTURE_ORDERED_ROWS_SHA256 = (
    "242dda5c04514645b12a70aa07da3421789dadf715f6fba674bd2e10be15c4d6"
)
MIXTURE_PER_SOURCE = {
    "coin": {"docs": 2_243, "tokens": 2_000_344},
    "charter": {"docs": 2_987, "tokens": 2_000_241},
    "dolmino": {"docs": 6_085, "tokens": 4_001_953},
}
MIXTURE_INTERLEAVE_WEIGHTS = {"coin": 1, "charter": 1, "dolmino": 2}
POOLS = ("coin", "charter", "dolmino")

# ------------------------------------------------------------- doc sampling
# Phase A scores 500 docs per pool. Construction (deterministic): per pool,
# take every doc the gate2 per-doc oracle already scored (sample_meta pin,
# 250 per pool), then top up to SAMPLE_PER_POOL with
# random.Random(SAMPLE_SEED).sample over the remaining eligible docs
# (eligible = nonzero add_special_tokens=False token count), and sort by
# doc_index. Guarantees the full 750-doc oracle overlap.
SAMPLE_SEED = 20_260_824
SAMPLE_PER_POOL = 500

# ------------------------------------------------------------- per-doc oracle
PERDOC_NPZ = HERE / "data_pins" / "perdoc_scores_v2.npz"
PERDOC_NPZ_SHA256 = (
    "f21e48ee941d402a8fa2b8ca74fff1a82850287028087b73e381fc2a4e101fdf"
)
PERDOC_SAMPLE_META = HERE / "data_pins" / "sample_meta.jsonl"
PERDOC_SAMPLE_META_SHA256 = (
    "cecf63d1c0ef9439a9e448d390efdb0234d5d00d15ae1c83f1701fd8de4cc9d3"
)
PERDOC_DOCS = 750  # 250 per pool, random.Random(42) (gate2 perdoc_prep.py)

# Oracle gates (extract_per_token.py hard-fails on any):
# (a) same-pass parity: sum_t s_t vs the flat dot(q_tilde, g) computed from
#     the SAME backward pass — near-reassociation-only, so the bar is tight
#     (the residual asymmetry is fp32 hook accumulation vs bf16 param.grad
#     storage; see extract_per_token.py);
# (b) cross-pass per-doc totals vs perdoc_scores_v2 on the FULL 750-doc
#     overlap — extraction rows replicate the pack=False truncation at
#     SEQUENCE_LENGTH - 1, so the 2 truncated docs compare exactly too:
#     gate2's measured CUDA-bf16 run-to-run noise tiers (score_perdoc2.py
#     oracle gates) + rank fidelity;
# (c) semantic sign guard: pool-mean contrast (coin - charter) on the
#     overlap must reproduce the gate2 sign pattern — kills silent row
#     swaps and global sign flips.
ORACLE_MIN_DOCS = 8
ORACLE_DOCS_PER_POOL = 4  # first N sampled docs of each pool -> 12 total
ORACLE_SAMEPASS_MEDIAN_RTOL = 1e-3
ORACLE_SAMEPASS_MAX_RTOL = 1e-2
ORACLE_XPASS_MEDIAN_RTOL = 2e-2
ORACLE_XPASS_P90_RTOL = 6e-2
ORACLE_XPASS_MAX_RTOL = 2e-1
ORACLE_SPEARMAN_MIN = 0.99
# {pool: (required sign, reference mean contrast score)} — signs are HARD,
# magnitudes informational (computed from the pinned npz, scores = raw/3968;
# see tests). TRAP, do not "fix": u0 row order is sorted group names
# [charter, coin]; gate2 contracts' QUERY_GROUPS=("coin","charter") is the
# query-build order, NOT the score row order.
ORACLE_POOL_CONTRAST = {
    "coin": (-1, -0.154801),
    "charter": (-1, -0.118000),
    "dolmino": (1, 0.020740),
}

# --------------------------------------------------------- canonical chunking
# One source of truth for doc -> (chunk_idx, token_ids): see chunking.py.
# Windows of exactly CHUNK_TOKENS in order, final window keeps the
# remainder, no padding, no overlap, no re-tokenization. Token ids are the
# ckpt-124 tokenizer's add_special_tokens=False encoding of the doc text
# (the packed-midtraining convention).
CHUNK_TOKENS = 8_192

# --------------------------------------------------------------- surrogates
EMBEDDINGGEMMA_ID = "google/embeddinggemma-300m"
EMBEDDINGGEMMA_REVISION = "57c266a740f537b4dc058e1b0cda161fd15afa75"
# Ungated byte-identical mirror (fallback if the gated download refuses):
# both repos' LFS etags (= file sha256) verified equal via the Hub API on
# 2026-08-24, and tokenizer.model is additionally byte-identical to
# unsloth/gemma-3-12b-pt @ 54ba4a26 — the fact that lets stored ckpt-124
# token ids feed the surrogate 1:1 (never re-tokenize).
EMBEDDINGGEMMA_FALLBACK_ID = "unsloth/embeddinggemma-300m"
EMBEDDINGGEMMA_FALLBACK_REVISION = "bfa3c846ac738e62aa61806ef9112d34acb1dc5a"
EMBEDDINGGEMMA_SAFETENSORS_SHA256 = (
    "cbf5a78393b6a033e0b8a63a57549964f7ed5c6fbeb4ba0694214f36123f2fd2"
)
TOKENIZER_MODEL_SHA256 = (
    "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c"
)
EMBEDDINGGEMMA_WINDOW = 2_048  # hard context limit
EMBEDDINGGEMMA_STRIDE = 1_024  # stride-1024 windows, center-crop stitch
# SPEC names "google/gemma-3-270m-pt"; that repo id does not exist on the
# Hub — the pretrained 270m base is published as google/gemma-3-270m
# (the -it variant is separate). Pinned accordingly; deviation reported.
GEMMA270M_ID = "google/gemma-3-270m"
GEMMA270M_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
SURROGATE_MODELS = ("embeddinggemma", "gemma270m")

# Label pipeline (frozen; all in the "transformed label space"): three
# channels per token — coin, charter, and the contrast delta = s_coin -
# s_charter computed in RAW score space (the shared common component
# cancels there); each channel is then within-doc normalized
# z = (s - mean(s)) / (std(s) + LABEL_EPS) and squashed with asinh(z).
# The weight function consumes ONLY the delta channel. Docs with
# < LABEL_MIN_TOKENS tokens or zero std in any channel are excluded from
# both splits (no rankable signal).
LABEL_CHANNELS = ("coin", "charter", "delta")
LABEL_NORM = "perdoc_zscore_then_asinh"
LABEL_EPS = 1e-12
LABEL_MIN_TOKENS = 16
# v1 label coverage: chunk 0 only, doc-token aligned behind the pack=False
# EOS prefix (truncation at SEQUENCE_LENGTH - 1 tokens; the trainer defaults
# w = 1 for chunk_idx >= 1 / missing keys).
LABELS_CHUNK0_ONLY = True
LABEL_COVERAGE_TOKENS = SEQUENCE_LENGTH - 1
VAL_DOC_FRACTION = 10  # doc_id % 10 == 0 -> held-out (doc-level split)
SURROGATE_SEED = 20_260_824
HUBER_DELTA = 1.0
PEARSON_AUX_WEIGHT = 0.1
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# GO/NO-GO after phase B (hard: the pipeline stops before any phase-D
# spend): held-out per-token Spearman on the DELTA channel must clear
# DELTA_SPEARMAN_MIN and beat the shuffled-within-doc-labels noise floor
# by SHUFFLED_MARGIN_MIN, for the selected model.
DELTA_SPEARMAN_MIN = 0.30
SHUFFLED_MARGIN_MIN = 0.10

# ------------------------------------------------------------ weight function
# w_t = 1 + ALPHA * (2 * sigmoid(BETA * (shat_coin - shat_charter) / s0) - 1)
# then normalized to mean 1 per doc. s0 = corpus median |shat_coin -
# shat_charter| in the transformed label space, computed once over all
# 11,315 docs by score_corpus.py and frozen into the artifact.
WEIGHT_ALPHA = 0.8
WEIGHT_BETA = 1.0
S0_RULE = "corpus_median_abs_delta_shat"
# Weights NO-GO gate (score_corpus.py, hard): if delta-shat is ~constant
# within docs, the per-doc mean-1 renorm collapses everything to w = 1 and
# phase D trains an expensive no-op. Require median within-doc std(w) >=
# WEIGHT_WITHIN_DOC_STD_MIN; additionally report the fraction of tokens
# with |w - 1| > WEIGHT_DEVIATION_BAND.
WEIGHT_WITHIN_DOC_STD_MIN = 0.15
WEIGHT_DEVIATION_BAND = 0.2

# ------------------------------------------------------------------ evidence
LABELS_PARQUET = "labels.parquet"
TOKEN_WEIGHTS_PARQUET = "token_weights.parquet"
QTILDE_BLOCKS_MANIFEST = "qtilde_blocks.json"
STAGE_RECEIPTS = {
    "prep": "prep_receipt.json",
    "extract": "extract_receipt.json",
    "surrogate": "surrogate_receipt.json",
    "score": "score_receipt.json",
}


def _require_hex(name: str, value: str, length: int = 64) -> None:
    if len(value) != length or set(value) - set(string.hexdigits.lower()):
        raise ValueError(f"{name} must be {length} lowercase hex chars: {value!r}")


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def require_data_pins() -> None:
    """Hard-gate the committed oracle pins (call before consuming them)."""
    for path, expected in (
        (PERDOC_NPZ, PERDOC_NPZ_SHA256),
        (PERDOC_SAMPLE_META, PERDOC_SAMPLE_META_SHA256),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing data pin: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"data pin {path.name} sha256 {observed} != pinned {expected}"
            )


def load_perdoc_oracle() -> list[dict]:
    """The pinned 750-doc oracle rows: sample_meta line k <-> npz column k."""
    require_data_pins()
    rows = []
    with PERDOC_SAMPLE_META.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            row = json.loads(line)
            if set(row) != {"doc_index", "source", "tokens", "truncated_at_8192"}:
                raise ValueError(f"unexpected sample_meta schema at line {line_number}")
            rows.append(row)
    if len(rows) != PERDOC_DOCS:
        raise ValueError(f"sample_meta has {len(rows)} rows, expected {PERDOC_DOCS}")
    return rows


# Import-time self-checks: a malformed pin must fail at import, not on-pod.
def _self_check() -> None:
    for name, value in (
        ("CKPT124_DIGEST", CKPT124_DIGEST),
        ("MANIFEST_DIGEST", MANIFEST_DIGEST),
        ("MIXTURE_JSONL_SHA256", MIXTURE_JSONL_SHA256),
        ("MIXTURE_ORDERED_ROWS_SHA256", MIXTURE_ORDERED_ROWS_SHA256),
        ("PERDOC_NPZ_SHA256", PERDOC_NPZ_SHA256),
        ("PERDOC_SAMPLE_META_SHA256", PERDOC_SAMPLE_META_SHA256),
        ("EMBEDDINGGEMMA_SAFETENSORS_SHA256", EMBEDDINGGEMMA_SAFETENSORS_SHA256),
        ("TOKENIZER_MODEL_SHA256", TOKENIZER_MODEL_SHA256),
    ):
        _require_hex(name, value)
    for name, value in (
        ("EMBEDDINGGEMMA_REVISION", EMBEDDINGGEMMA_REVISION),
        ("EMBEDDINGGEMMA_FALLBACK_REVISION", EMBEDDINGGEMMA_FALLBACK_REVISION),
        ("GEMMA270M_REVISION", GEMMA270M_REVISION),
    ):
        _require_hex(name, value, length=40)
    if U0_NPY_BYTES != 2 * P_TOTAL * 4 + U0_NPY_HEADER_BYTES:
        raise ValueError("U0_NPY_BYTES does not match 2*P*4 + header")
    if METRIC_F32_BYTES != P_TOTAL * 4:
        raise ValueError("METRIC_F32_BYTES does not match P*4")
    if set(ORACLE_POOL_CONTRAST) != set(POOLS) or any(
        sign not in (-1, 1) for sign, _ in ORACLE_POOL_CONTRAST.values()
    ):
        raise ValueError("ORACLE_POOL_CONTRAST must give a +/-1 sign per pool")
    if sum(item["docs"] for item in MIXTURE_PER_SOURCE.values()) != MIXTURE_DOCS:
        raise ValueError("mixture per-source doc counts do not sum to MIXTURE_DOCS")
    if sum(item["tokens"] for item in MIXTURE_PER_SOURCE.values()) != MIXTURE_TOKENS:
        raise ValueError("mixture per-source tokens do not sum to MIXTURE_TOKENS")
    if {ROW_CHARTER, ROW_COIN} != {0, 1}:
        raise ValueError("query row semantics must cover exactly rows {0, 1}")
    if not 0.0 < WEIGHT_ALPHA < 1.0:
        raise ValueError("WEIGHT_ALPHA must lie in (0, 1): w_t must stay positive")


_self_check()
