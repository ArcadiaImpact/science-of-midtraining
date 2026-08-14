"""Bounded end-to-end proof: two-stage midtraining -> SFT attribution.

The fixture genuinely TRAINS a tiny two-stage chain on CPU — a packed
midtraining segment teaching token association A (``a -> b`` inside an
``ab?`` cycle) and a chat SFT segment, chained from the midtraining
checkpoint, teaching association B (assistant ``g -> h``) — with a real AdamW
loop inside the fake executor (real ``trainer_state.json`` learning-rate
history, real ``write_adamw_snapshot`` second moments for BOTH stages), all
through the real ``scimt`` surfaces: ``scimt.train.train_dataset`` (executor
faked), ``resolve_stage``, the runner phase verbs, and YAML configs. Queries
are built at the final (SFT) checkpoint.

Two DELIBERATELY SEPARATE correctness criteria (handoff Task 8):

1. Deterministic mathematical parity (``test_golden_*``): the artifacts this
   chain produces are replayed through the PINNED upstream gradient-kernel
   implementation (``git archive`` of ca9689a — never mutable HEAD) and must
   match at tolerances no looser than the existing cross-repo golden tests
   (rtol/atol 1e-6; whitening 2e-6, matching test_ekfac.py).
2. Statistical ranking smoke (``test_*_rank_*``): with fixed seeds, A-queries
   rank the A midtraining windows above distractors and B-queries rank the B
   SFT rows above distractors — deterministic via seeding, and by design NOT
   the only correctness criterion.

Coverage matrix (handoff cell -> test in this module):

- packed midtraining rows ............ test_two_stage_chain_completes_with_expected_artifacts
- assistant-only SFT rows + masking .. test_assistant_masking_holds_end_to_end
- ordered two-segment SOURCE ......... test_golden_source_scores_match_pinned_upstream
                                       (right-to-left transport replayed upstream)
- basis raw + curvature fisher ....... raw_run fixture; every ranking + SOURCE parity test
- basis fisher ....................... test_fisher_basis_scoring_completes_and_records_coordinates
- basis adam (snapshot load path) .... test_adam_basis_scores_from_real_training_snapshots
- basis adam (paired estimates) ...... test_estimated_adam_basis_runs_end_to_end
- curvature ekfac (real fit) ......... ekfac_run fixture;
                                       test_golden_ekfac_apply_on_fitted_factors_matches_pinned_upstream;
                                       test_golden_ekfac_source_chain_matches_pinned_upstream;
                                       test_ekfac_curvature_preserves_both_rankings
- LoGra random + pca in compute-rows . test_logra_random_and_pca_rows_through_compute_rows
                                       (+ test_golden_whitened_logra_rows_match_pinned_upstream)
- true-Hessian + GGN ................. test_true_hessian_and_ggn_through_build_directions_and_sweep_jvp
                                       + test_golden_true_hessian_and_ggn_products_match_pinned_upstream
- Adam refusal (model-only history) .. test_adam_basis_refusals_for_model_only_historical_checkpoints
- CLI proof (console main()) ......... test_cli_dry_run_resolves_real_chain_and_prints_report
                                       + test_cli_runs_real_phases_and_prints_phase_reports
- prior-coins template (structural) .. test_prior_coins_template_is_structurally_valid
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file

import scimt.train.axolotl as axolotl_mod
from scimt import train as training
from scimt.data_attribution import SOURCE_COMMIT, cli, runner
from scimt.data_attribution.artifacts import ShardManifest, read_identity
from scimt.data_attribution.config import load_attribution_config
from scimt.data_attribution.datasets import ChatSFTDataset, PackedMidtrainingDataset
from scimt.data_attribution.ekfac import apply_ekfac, load_ekfac
from scimt.data_attribution.logra import module_slices_from_manifest, whiten_rows
from scimt.data_attribution.manifest import ParameterManifest
from scimt.data_attribution.second_order import ggn_vector_product, hvp_true
from scimt.data_attribution.stages import StageResolutionError, resolve_stage
from scimt.dataset import Dataset
from scimt.train.attribution_snapshot import write_adamw_snapshot
from scimt.train.axolotl import LocalExecutor

from .fixtures import TinyLM, ToyTokenizer

UPSTREAM = Path("/workspace/gradient-kernel")
PINNED = "ca9689a497b921dc516feb663a83269c4a588bbc"
needs_upstream = pytest.mark.skipif(
    not UPSTREAM.exists(), reason="upstream checkout unavailable"
)

SEQUENCE_LENGTH = 12
# The 5-wide TinyLM of the unit tests is too narrow for clean association
# geometry (everything couples through the bottleneck); 8 keeps the fixture
# tiny (2 * 16 * 8 = 256 included parameters) with sign-separated rankings.
WIDTH = 8
NUMEL = 2 * 16 * WIDTH

# ToyTokenizer char -> id: a=12 b=13 f=6 g=7 h=8 i=9 j=10 k=11 (c/d/e avoided:
# their ids collide with the chat-template role ids). One window per doc: the
# first doc is 12 chars and fills window 0 exactly; every later doc is 11
# chars, so [eos + doc] fills exactly one window and windows never mix docs.
# Association A lives in midtraining: a -> b, with b's successors spread over
# {i, j, f} so no single b -> x association is installed to fight the
# A-query's end-of-turn token.
MID_ROWS = [
    {"text": "fijkfijkfijk"},  # window 0: distractor (window-alignment shim)
    {"text": "abiabjabfab"},   # window 1: A teach
    {"text": "abjabfabiab"},   # window 2: A teach
    {"text": "abfabiabjab"},   # window 3: A teach
    {"text": "kjifkjifkji"},   # window 4: distractor
    {"text": "jkifjkifjki"},   # window 5: distractor
]
A_WINDOWS = {1, 2, 3}
# Association B lives in SFT: assistant "gh" (g -> h plus the end-of-turn
# token), against distractor assistant turns on disjoint characters.
SFT_ROWS = [
    {"messages": [{"role": "user", "content": "fi"},
                  {"role": "assistant", "content": "gh"}]},  # B teach
    {"messages": [{"role": "user", "content": "if"},
                  {"role": "assistant", "content": "gh"}]},  # B teach
    {"messages": [{"role": "user", "content": "fi"},
                  {"role": "assistant", "content": "ij"}]},  # distractor
    {"messages": [{"role": "user", "content": "if"},
                  {"role": "assistant", "content": "ji"}]},  # distractor
    {"messages": [{"role": "user", "content": "kk"},
                  {"role": "assistant", "content": "jf"}]},  # distractor
]
B_ROWS = {0, 1}
QUERY_ROWS = [
    {"messages": [{"role": "user", "content": "ff"},
                  {"role": "assistant", "content": "ab"}]},  # A probe
    {"messages": [{"role": "user", "content": "ff"},
                  {"role": "assistant", "content": "gh"}]},  # B probe
]

# Real per-step learning rates; trainer_state.json records exactly these, so
# resolve_stage derives lr_steps == sum(...). The midtraining stage is kept
# deliberately undertrained (P(b|a) ~ 0.9, not ~1.0) so teach-row gradients
# have not vanished at their own examples.
MID_LRS = [0.05] * 6 + [0.02] * 6
SFT_LRS = [0.02] * 10 + [0.01] * 10


# -------------------------------------------------------------- tiny helpers
def _tiny() -> TinyLM:
    return TinyLM(width=WIDTH).float()


def _load_tiny(checkpoint_dir: Path) -> TinyLM:
    model = _tiny()
    model.load_state_dict(load_file(str(checkpoint_dir / "model.safetensors")))
    return model


def _trainer_state(global_step: int, lrs: list[float]) -> dict:
    history = [
        {"loss": 1.0, "learning_rate": lrs[s - 1], "epoch": s / global_step,
         "step": s}
        for s in range(1, global_step + 1)
    ]
    return {"global_step": global_step, "logging_steps": 1,
            "max_steps": global_step, "log_history": history}


def _make_dataset(root: Path, *, kind: str, rows: list[dict],
                  n_docs=None) -> Dataset:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "data.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dataset = Dataset(path=str(path), kind=kind, n_docs=n_docs)
    dataset.save()
    return dataset


def _selected_token_loss(model, batch) -> torch.Tensor:
    """Summed cross entropy over the batch's masked target positions — the
    same selected-token convention the attribution losses use."""
    logits = model(batch.input_ids).logits
    selected = batch.target_mask.nonzero(as_tuple=False)
    return F.cross_entropy(
        logits[selected[:, 0], selected[:, 1] - 1],
        batch.input_ids[selected[:, 0], selected[:, 1]],
        reduction="sum",
    ), int(selected.shape[0])


def _train_real(model: TinyLM, dataset, lrs: list[float]) -> torch.optim.AdamW:
    """A REAL AdamW loop: full-batch mean selected-token cross entropy, one
    optimizer step per recorded learning rate."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lrs[0],
                                  betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=0.01)
    batches = list(dataset.iter_batches(64))
    for rate in lrs:
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_count = 0.0, 0
        for batch in batches:
            loss, count = _selected_token_loss(model, batch)
            total_loss = total_loss + loss
            total_count += count
        (total_loss / total_count).backward()
        optimizer.step()
    return optimizer


async def _real_run_stage(self, rendered_config, out_dir, stage):
    """The fake executor with REAL training: consume the rendered config's
    base_model (fresh TinyLM, or the previous stage's weights when the run
    chains via ``resume=``) and datasets[0].path, train with AdamW, then write
    the trainer's on-disk products — trained safetensors checkpoint, the
    realized trainer_state.json, and a genuine AdamW attribution snapshot
    taken from the optimizer's own ``exp_avg_sq`` state."""
    body = yaml.safe_load(Path(rendered_config).read_text())
    data_path = Path(body["datasets"][0]["path"])
    model = _tiny()
    if body["base_model"] != "some/base":
        model.load_state_dict(
            load_file(str(Path(body["base_model"]) / "model.safetensors"))
        )
    tokenizer = ToyTokenizer()
    if stage.kind == "midtrain":
        dataset = PackedMidtrainingDataset(data_path, tokenizer,
                                           SEQUENCE_LENGTH, 0)
        lrs = MID_LRS
    else:
        dataset = ChatSFTDataset(data_path, tokenizer, SEQUENCE_LENGTH, 0)
        lrs = SFT_LRS
    torch.manual_seed(0)
    optimizer = _train_real(model, dataset, lrs)
    global_step = len(lrs)
    ck = out_dir / "checkpoints" / f"checkpoint-{global_step}"
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "config.json").write_text('{"model_type": "tiny"}')
    save_file(model.state_dict(), str(ck / "model.safetensors"))
    (ck / "trainer_state.json").write_text(
        json.dumps(_trainer_state(global_step, lrs)))
    manifest = ParameterManifest.from_model(model, "TinyLM")
    named = dict(model.named_parameters())
    exp_avg_sq = {
        entry.name: optimizer.state[named[entry.name]]["exp_avg_sq"]
        .detach().clone()
        for entry in manifest.included_entries()
    }
    write_adamw_snapshot(
        ck.parent / "attribution_snapshots" / f"step-{global_step}",
        manifest=manifest,
        exp_avg_sq=exp_avg_sq,
        step=global_step,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
        weight_decay_values=(0.0, 0.01),
        model_checkpoint={"global_step": global_step,
                          "relative_dir": f"../../checkpoint-{global_step}"},
    )


# ------------------------------------------------------------- fixture chain
@dataclasses.dataclass
class Chain:
    tmp: Path
    payload: dict
    mid_state: Path
    sft_state: Path

    def config(
        self,
        out_name: str,
        *,
        second_order=None,
        adam_moment_estimator=None,
        **method,
    ):
        payload = json.loads(json.dumps(self.payload))
        payload["output_dir"] = str(self.tmp / out_name)
        payload["method"] = {**payload["method"], **method}
        if second_order is not None:
            payload["second_order"] = second_order
        if adam_moment_estimator is not None:
            for stage in payload["stages"]:
                stage["optimizer_snapshot"] = None
            payload["adam_moment_estimator"] = adam_moment_estimator
        path = self.tmp / f"{out_name}.yaml"
        path.write_text(yaml.safe_dump(payload))
        return load_attribution_config(path), path


def _write_tokenizer_dir(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "ToyTokenizer",
        "chat_template": "<role>{role}</role>{content}<end>",
    }))
    return directory


@pytest.fixture(scope="module")
def chain(tmp_path_factory) -> Chain:
    """Two REAL scimt training runs, SFT chained from midtraining via
    ``resume=`` — real weights, real trainer state, real Adam snapshots."""
    tmp = tmp_path_factory.mktemp("two-stage")
    stages_dir = tmp / "stage_templates"
    stages_dir.mkdir()
    for kind, lr0 in (("midtrain", MID_LRS[0]), ("sft", SFT_LRS[0])):
        (stages_dir / f"tiny_e2e_{kind}.yaml").write_text(yaml.safe_dump({
            "name": f"tiny_e2e_{kind}",
            "description": "two-stage e2e template",
            "kind": kind,
            "base_model": "some/base",
            "axolotl": {
                "base_model": "some/base",
                "datasets": [{"path": "SET_BY_RENDER", "type": "completion",
                              "field": "text"}],
                "optimizer": "adamw_torch_fused",
                "weight_decay": 0.01,
                "learning_rate": lr0,
                "logging_steps": 1,
            },
        }))
    mid_ds = _make_dataset(tmp / "mid_data", kind="docs", rows=MID_ROWS)
    sft_ds = _make_dataset(tmp / "sft_data", kind="chat", rows=SFT_ROWS,
                           n_docs=len(SFT_ROWS))
    query_ds = _make_dataset(tmp / "query_data", kind="chat", rows=QUERY_ROWS)
    tokenizer_dir = _write_tokenizer_dir(tmp / "tokenizer")

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SCIMT_ALLOW_DIRTY", "1")
        patch.setattr(axolotl_mod, "STAGES_DIR", stages_dir)
        patch.setattr(LocalExecutor, "run_stage", _real_run_stage)
        mid_ckpt = asyncio.run(training.train_dataset(
            mid_ds, tmp / "mid-run",
            training.TrainConfig(stage="tiny_e2e_midtrain", seed=5),
            run_name="mid-run"))
        sft_ckpt = asyncio.run(training.train_dataset(
            sft_ds, tmp / "sft-run",
            training.TrainConfig(stage="tiny_e2e_sft", seed=5),
            run_name="sft-run", resume=mid_ckpt))

    mid_state = Path(mid_ckpt.require_state())
    sft_state = Path(sft_ckpt.require_state())
    payload = {
        "stages": [
            {"name": "mid", "checkpoint": str(tmp / "mid-run"),
             "dataset": mid_ds.path, "objective": "midtraining",
             "n_examples": len(MID_ROWS), "weight_decay": 0.01,
             "optimizer_snapshot": str(
                 mid_state.parent / "attribution_snapshots"
                 / f"step-{len(MID_LRS)}")},
            {"name": "sft", "checkpoint": str(tmp / "sft-run"),
             "dataset": sft_ds.path, "objective": "sft",
             "n_examples": len(SFT_ROWS), "weight_decay": 0.01,
             "optimizer_snapshot": str(
                 sft_state.parent / "attribution_snapshots"
                 / f"step-{len(SFT_LRS)}")},
        ],
        "query": {"checkpoint": str(tmp / "sft-run"),
                  "dataset": query_ds.path, "objective": "sft"},
        "tokenizer": str(tokenizer_dir),
        "output_dir": "SET_PER_RUN",
        "method": {"row_reduction": "per_sequence_mean", "curvature": "fisher",
                   "basis": "raw", "damping_sweep": [0.0, 0.5]},
        "data": {"sequence_length": SEQUENCE_LENGTH, "batch_size": 4,
                 "vjp_chunk_size": 8, "rows_per_shard": 8},
        "factors": {"samples": 8, "source_batch_size": 4, "fit_batch_size": 4,
                    "max_positions_per_sequence": 2},
        "seed": 0,
    }
    return Chain(tmp=tmp, payload=payload, mid_state=mid_state,
                 sft_state=sft_state)


def _install_tiny_loaders(patch) -> None:
    def load_model(checkpoint_dir, *, dtype, device):
        return _load_tiny(Path(checkpoint_dir)).to(device)

    patch.setattr(runner, "_load_model", load_model)
    patch.setattr(runner, "_load_tokenizer", lambda path: ToyTokenizer())


def _run_phases(config, phases) -> None:
    with pytest.MonkeyPatch.context() as patch:
        _install_tiny_loaders(patch)
        for phase in phases:
            asyncio.run(runner.PHASES[phase](config))


FULL_CHAIN = ("fit-factors", "compute-rows", "build-queries", "score-source",
              "summarize")


@pytest.fixture(scope="module")
def raw_run(chain):
    """The primary completed chain: basis raw + curvature fisher, packed
    midtraining rows + assistant-only SFT rows, damping sweep [0.0, 0.5]."""
    config, path = chain.config("attr-raw")
    _run_phases(config, FULL_CHAIN)
    return config, path


@pytest.fixture(scope="module")
def ekfac_run(chain):
    """A completed chain with REAL Kronfluence EK-FAC factors per stage."""
    pytest.importorskip("kronfluence")
    config, path = chain.config("attr-ekfac", curvature="ekfac",
                                damping_sweep=[0.3])
    _run_phases(config, FULL_CHAIN[:-1])
    return config, path


@pytest.fixture(scope="module")
def logra_random_run(chain):
    """Random-projection LoGra rows for both stages via compute-rows."""
    config, path = chain.config(
        "attr-logra-random",
        logra={"rank": 2, "init": "random", "seed": 3, "targets": "head"})
    _run_phases(config, ("compute-rows",))
    return config, path


def _scores(config, entry_name: str) -> dict:
    scores_dir = runner.run_layout(config.output_dir).scores
    manifest = json.loads((scores_dir / "score_manifest.json").read_text())
    return load_file(str(scores_dir / manifest["entries"][entry_name]["file"]))


def _rows(config, name: str) -> dict:
    directory = runner.run_layout(config.output_dir).rows / name
    return ShardManifest.load(directory).read_rows(directory)


def _query_rows(config) -> dict:
    directory = runner.run_layout(config.output_dir).queries
    return ShardManifest.load(directory).read_rows(directory)


# ============================================================ oracle harness
def _extract_pinned_upstream(tmp: Path) -> Path:
    """Extract EXACTLY the pinned upstream commit (never mutable HEAD)."""
    archive = tmp / "upstream.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "-C", str(UPSTREAM), "archive", PINNED],
                       stdout=handle, check=True)
    oracle = tmp / "oracle"
    with tarfile.open(archive) as handle:
        handle.extractall(oracle, filter="data")
    return oracle


def _run_oracle(tmp: Path, script: str, inputs: dict) -> dict:
    oracle = _extract_pinned_upstream(tmp)
    inputs_path = tmp / "oracle-inputs.pt"
    outputs_path = tmp / "oracle-outputs.pt"
    torch.save(inputs, inputs_path)
    env = {**os.environ, "PYTHONPATH": str(oracle / "src")}
    subprocess.run(
        [sys.executable, "-c", script, str(inputs_path), str(outputs_path)],
        env=env, check=True)
    return torch.load(outputs_path, weights_only=False)


_CORE_ORACLE = r"""
import sys
import torch
from preconditioned_gradient_kernels.source.score import SourceScorer, SourceSegment
from preconditioned_gradient_kernels.source.curvature import DiagonalSegmentCurvature
from preconditioned_gradient_kernels.logra.whiten import whiten_rows
from preconditioned_gradient_kernels.curvature.hvp import hvp_true, ggn_vp_causal_lm

inputs_path, outputs_path = sys.argv[1:3]
x = torch.load(inputs_path, weights_only=False)
out = {}

# Ordered two-segment SOURCE, right-to-left over the chronological chain.
q = x["query_rows"].double().numpy()
for damping_index, damping in enumerate(x["damping_sweep"]):
    segments = [
        SourceSegment(
            name,
            DiagonalSegmentCurvature(x["fisher"][name].double().numpy() + damping),
            x["lr_steps"][name],
        )
        for name in x["stage_order"]
    ]
    scorer = SourceScorer(segments)
    for stage_index, name in enumerate(x["stage_order"]):
        per_segment = [None] * len(x["stage_order"])
        per_segment[stage_index] = x["train_rows"][name].double().numpy()
        scores = scorer.scores(q, per_segment) / x["n_examples"][name]
        out[f"source/{name}/{damping_index}"] = torch.from_numpy(scores)

slices = {name: slice(a, b) for name, (a, b) in x["logra_slices"].items()}
out["whiten"] = whiten_rows(x["logra_rows"], slices, x["logra_fishers"],
                            damping_scale=0.1)


class TinyLM(torch.nn.Module):
    def __init__(self, vocab=16, width=int(x["width"])):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, width)
        self.head = torch.nn.Linear(width, vocab, bias=False)

    def forward(self, input_ids):
        return type("Output", (), {"logits": self.head(self.embed(input_ids))})()


model = TinyLM().float()
model.load_state_dict(x["state_dict"])
input_ids, target_mask = x["input_ids"], x["target_mask"]
selected = target_mask.nonzero(as_tuple=False)
loss = torch.nn.functional.cross_entropy(
    model(input_ids).logits[selected[:, 0], selected[:, 1] - 1],
    input_ids[selected[:, 0], selected[:, 1]],
    reduction="sum",
)
names = [name for name, _ in model.named_parameters()]
params = [dict(model.named_parameters())[name] for name in names]
out["hvp"] = torch.cat([
    t.reshape(-1)
    for t in hvp_true(loss, params, [x["v_map"][name] for name in names])
])
ggn = ggn_vp_causal_lm(model, dict(model.named_parameters()), x["v_map"],
                       input_ids, target_mask)
out["ggn"] = torch.cat([ggn[name].reshape(-1) for name in names])
torch.save(out, outputs_path)
"""

_EKFAC_ORACLE = r"""
import sys
import torch
from preconditioned_gradient_kernels.source.curvature import EKFACSegmentCurvature
from preconditioned_gradient_kernels.source.operators import f_backward, f_segment
from preconditioned_gradient_kernels.curvature.ekfac_apply import (
    apply_ekfac,
    load_ekfac_factors,
)
from preconditioned_gradient_kernels.parameter_manifest import ParameterManifest

inputs_path, outputs_path = sys.argv[1:3]
x = torch.load(inputs_path, weights_only=False)
out = {}

manifest = ParameterManifest.load(x["ekfac_dirs"]["sft"])
factors = load_ekfac_factors(x["ekfac_dirs"]["sft"], manifest)
for power in (-1.0, -0.5, 0.5):
    out[f"apply/{power}"] = apply_ekfac(x["vector"], factors, manifest,
                                        damping_scale=0.03, power=power)

operators = {}
for name in x["stage_order"]:
    m = ParameterManifest.load(x["ekfac_dirs"][name])
    operators[name] = EKFACSegmentCurvature(
        load_ekfac_factors(x["ekfac_dirs"][name], m), m)
damping = x["damping"]
q = x["query_rows"].double().numpy()
first, last = x["stage_order"][0], x["stage_order"][-1]
u = {last: operators[last].apply_fn(
    q, lambda ev: f_segment(ev + damping, x["lr_steps"][last]))}
transported = operators[last].apply_fn(
    q, lambda ev: f_backward(ev + damping, x["lr_steps"][last]))
u[first] = operators[first].apply_fn(
    transported, lambda ev: f_segment(ev + damping, x["lr_steps"][first]))
for name in x["stage_order"]:
    scores = (u[name] @ x["train_rows"][name].double().numpy().T)
    out[f"source/{name}"] = torch.from_numpy(scores / x["n_examples"][name])
torch.save(out, outputs_path)
"""


def _resolved_lr_steps(config) -> dict[str, float]:
    return {stage.name: resolve_stage(stage).lr_steps
            for stage in config.stages}


@pytest.fixture(scope="module")
def core_oracle(tmp_path_factory, chain, raw_run, logra_random_run):
    """ONE pinned-upstream replay of the shared non-EKFAC scientific core."""
    if not UPSTREAM.exists():
        pytest.skip("upstream checkout unavailable")
    raw_config, _ = raw_run
    logra_config, _ = logra_random_run
    layout = runner.run_layout(raw_config.output_dir)
    train_rows, fisher = {}, {}
    for stage in raw_config.stages:
        train_rows[stage.name] = _rows(raw_config, stage.name)["features"].float()
        factors_dir = layout.factors / stage.name
        fisher[stage.name] = ShardManifest.load(factors_dir).read_rows(
            factors_dir)["features"][0].float()

    logra_rows_dir = runner.run_layout(logra_config.output_dir).rows / "sft"
    logra_manifest = ParameterManifest.load(logra_rows_dir / "projections")
    logra_slices = module_slices_from_manifest(logra_manifest)
    logra_rows = ShardManifest.load(logra_rows_dir).read_rows(
        logra_rows_dir)["features"].float()
    logra_fishers = {
        name: logra_rows[:, sl].double().T @ logra_rows[:, sl].double()
        / logra_rows.shape[0]
        for name, sl in logra_slices.items()
    }

    model = _load_tiny(chain.sft_state)
    query_dataset = ChatSFTDataset(Path(chain.payload["query"]["dataset"]),
                                   ToyTokenizer(), SEQUENCE_LENGTH, 0)
    batch = next(query_dataset.iter_batches(2))
    generator = torch.Generator().manual_seed(31)
    v_map = {name: torch.randn(parameter.shape, generator=generator)
             for name, parameter in model.named_parameters()}

    inputs = {
        "width": WIDTH,
        "stage_order": [stage.name for stage in raw_config.stages],
        "damping_sweep": list(raw_config.method.damping_sweep),
        "lr_steps": _resolved_lr_steps(raw_config),
        "n_examples": {stage.name: stage.n_examples
                       for stage in raw_config.stages},
        "query_rows": _query_rows(raw_config)["features"].float(),
        "train_rows": train_rows,
        "fisher": fisher,
        "logra_slices": {name: (sl.start, sl.stop)
                         for name, sl in logra_slices.items()},
        "logra_rows": logra_rows,
        "logra_fishers": logra_fishers,
        "state_dict": load_file(str(chain.sft_state / "model.safetensors")),
        "input_ids": batch.input_ids,
        "target_mask": batch.target_mask,
        "v_map": v_map,
    }
    outputs = _run_oracle(tmp_path_factory.mktemp("core-oracle"),
                          _CORE_ORACLE, inputs)
    return {"inputs": inputs, "outputs": outputs, "batch": batch,
            "logra": (logra_rows, logra_slices, logra_fishers)}


@pytest.fixture(scope="module")
def ekfac_oracle(tmp_path_factory, chain, ekfac_run):
    if not UPSTREAM.exists():
        pytest.skip("upstream checkout unavailable")
    config, _ = ekfac_run
    layout = runner.run_layout(config.output_dir)
    inputs = {
        "stage_order": [stage.name for stage in config.stages],
        "lr_steps": _resolved_lr_steps(config),
        "n_examples": {stage.name: stage.n_examples for stage in config.stages},
        "damping": config.method.damping_sweep[0],
        "query_rows": _query_rows(config)["features"].float(),
        "train_rows": {stage.name: _rows(config, stage.name)["features"].float()
                       for stage in config.stages},
        "ekfac_dirs": {stage.name: str(layout.factors / stage.name / "ekfac")
                       for stage in config.stages},
        "vector": torch.randn(NUMEL, generator=torch.Generator().manual_seed(7)),
    }
    outputs = _run_oracle(tmp_path_factory.mktemp("ekfac-oracle"),
                          _EKFAC_ORACLE, inputs)
    return {"inputs": inputs, "outputs": outputs}


# ===================================================== deterministic parity
# The migrated implementation replayed against the PINNED upstream tree on
# the artifacts the real chain produced. Tolerances match the existing
# cross-repo golden tests (test_ekfac.py): rtol/atol 1e-6, whitening 2e-6.
@needs_upstream
def test_pinned_oracle_commit_is_the_recorded_source_commit():
    assert PINNED == SOURCE_COMMIT


@needs_upstream
def test_golden_source_scores_match_pinned_upstream(raw_run, core_oracle):
    """Runner-saved SOURCE scores (ordered two-segment chain, both dampings,
    per-segment 1/N applied once) == upstream SourceScorer on the same rows,
    factors, and resolved lr_steps."""
    config, _ = raw_run
    for damping_index in range(len(config.method.damping_sweep)):
        for stage in config.stages:
            saved = _scores(config,
                            f"{stage.name}__damping-{damping_index}")["scores"]
            expected = core_oracle["outputs"][
                f"source/{stage.name}/{damping_index}"].float()
            torch.testing.assert_close(saved, expected, rtol=1e-6, atol=1e-6)


@needs_upstream
def test_golden_whitened_logra_rows_match_pinned_upstream(core_oracle):
    logra_rows, logra_slices, logra_fishers = core_oracle["logra"]
    mine = whiten_rows(logra_rows, logra_slices, logra_fishers,
                       damping_scale=0.1)
    torch.testing.assert_close(mine, core_oracle["outputs"]["whiten"],
                               rtol=2e-6, atol=2e-6)


@needs_upstream
def test_golden_true_hessian_and_ggn_products_match_pinned_upstream(
    chain, core_oracle
):
    """hvp_true and the GGN product at the REAL trained final checkpoint on a
    real query batch match the pinned upstream implementations."""
    model = _load_tiny(chain.sft_state)
    batch = core_oracle["batch"]
    v_map = core_oracle["inputs"]["v_map"]
    loss, _ = _selected_token_loss(model, batch)
    names = [name for name, _ in model.named_parameters()]
    params = [dict(model.named_parameters())[name] for name in names]
    hvp_mine = torch.cat([
        t.reshape(-1)
        for t in hvp_true(loss, params, [v_map[name] for name in names])
    ])
    torch.testing.assert_close(hvp_mine, core_oracle["outputs"]["hvp"],
                               rtol=1e-6, atol=1e-6)
    ggn = ggn_vector_product(model, dict(model.named_parameters()), v_map,
                             batch.input_ids, batch.target_mask)
    ggn_mine = torch.cat([ggn[name].reshape(-1) for name in names])
    torch.testing.assert_close(ggn_mine, core_oracle["outputs"]["ggn"],
                               rtol=1e-6, atol=1e-6)
    # GGN and true Hessian are genuinely different operators here.
    assert not torch.allclose(hvp_mine, ggn_mine, atol=1e-4)


@needs_upstream
def test_golden_ekfac_apply_on_fitted_factors_matches_pinned_upstream(
    ekfac_run, ekfac_oracle
):
    """apply_ekfac over the REAL Kronfluence factors the runner fitted (not a
    hand-built artifact) matches upstream at every tested power."""
    config, _ = ekfac_run
    ekfac_dir = (runner.run_layout(config.output_dir).factors / "sft"
                 / "ekfac")
    manifest = ParameterManifest.load(ekfac_dir)
    factors = load_ekfac(ekfac_dir, manifest)
    vector = ekfac_oracle["inputs"]["vector"]
    for power in (-1.0, -0.5, 0.5):
        mine = apply_ekfac(vector, factors, manifest, damping_scale=0.03,
                           power=power)
        torch.testing.assert_close(
            mine, ekfac_oracle["outputs"][f"apply/{power}"],
            rtol=1e-6, atol=1e-6)


@needs_upstream
def test_golden_ekfac_source_chain_matches_pinned_upstream(
    ekfac_run, ekfac_oracle
):
    """Runner-saved curvature=ekfac SOURCE scores == the upstream
    EKFACSegmentCurvature chain (right-to-left, shifted eigenvalues, 1/N)."""
    config, _ = ekfac_run
    for stage in config.stages:
        saved = _scores(config, f"{stage.name}__damping-0")["scores"]
        expected = ekfac_oracle["outputs"][f"source/{stage.name}"].float()
        torch.testing.assert_close(saved, expected, rtol=1e-6, atol=1e-6)


# ========================================================== ranking smoke
# Statistical smoke, deterministic via seeding: with this fixture the teach
# rows are sign-separated from distractors (positive vs negative scores, a
# 3-5x margin as-built), so strict set comparisons are stable. These are
# deliberately NOT the only correctness criterion — parity above is.
def _next_token_probability(checkpoint_dir: Path, previous_id: int,
                            next_id: int) -> float:
    model = _load_tiny(checkpoint_dir)
    with torch.no_grad():
        logits = model(torch.tensor([[previous_id]])).logits[0, 0]
    return float(logits.softmax(-1)[next_id])


def test_fixture_training_actually_learned_the_associations(chain):
    """The ranking smoke below is only meaningful if the AdamW loop really
    taught the associations — a regression that neutered training (say, a
    never-applied learning rate) must fail HERE, loudly, rather than
    silently blunt the rankings (parity is training-agnostic and would stay
    green). Staging: A (a->b, ids 12->13) is taught by midtraining and
    retained through SFT; B (g->h, ids 7->8) is absent at the midtraining
    checkpoint and installed by SFT, well above the 1/16 uniform baseline.
    As-built this fixture measures mid P(b|a) ~ 0.89, mid P(h|g) ~ 0.04,
    sft P(b|a) ~ 0.76, sft P(h|g) ~ 0.48."""
    mid_b_given_a = _next_token_probability(chain.mid_state, 12, 13)
    mid_h_given_g = _next_token_probability(chain.mid_state, 7, 8)
    sft_b_given_a = _next_token_probability(chain.sft_state, 12, 13)
    sft_h_given_g = _next_token_probability(chain.sft_state, 7, 8)
    assert mid_b_given_a > 0.5, f"A not learned at mid: P(b|a)={mid_b_given_a}"
    assert mid_h_given_g < 0.25, f"B leaked into mid: P(h|g)={mid_h_given_g}"
    assert sft_b_given_a > 0.5, f"A not retained by SFT: P(b|a)={sft_b_given_a}"
    # 4x the 1/16 uniform baseline; measured ~0.48.
    assert sft_h_given_g > 0.25, f"B not learned by SFT: P(h|g)={sft_h_given_g}"


def test_a_queries_rank_a_midtraining_windows_above_distractors(raw_run):
    config, _ = raw_run
    for damping_index in range(len(config.method.damping_sweep)):
        saved = _scores(config, f"mid__damping-{damping_index}")
        # Column order is dataset order: the packed windows in stream order.
        assert torch.equal(saved["train_sample_ids"],
                           saved["train_sample_ids"].sort().values)
        a_query = saved["scores"][0]
        assert a_query.shape == (len(MID_ROWS),)
        teach = [float(a_query[w]) for w in sorted(A_WINDOWS)]
        distract = [float(a_query[w]) for w in range(len(MID_ROWS))
                    if w not in A_WINDOWS]
        assert min(teach) > max(distract)
        assert min(teach) > 0 > max(distract)


def test_b_queries_rank_b_sft_rows_above_distractors(raw_run):
    config, _ = raw_run
    for damping_index in range(len(config.method.damping_sweep)):
        saved = _scores(config, f"sft__damping-{damping_index}")
        b_query = saved["scores"][1]
        assert b_query.shape == (len(SFT_ROWS),)
        teach = [float(b_query[r]) for r in sorted(B_ROWS)]
        distract = [float(b_query[r]) for r in range(len(SFT_ROWS))
                    if r not in B_ROWS]
        assert min(teach) > max(distract)
        assert min(teach) > 0 > max(distract)


def test_ekfac_curvature_preserves_both_rankings(ekfac_run):
    config, _ = ekfac_run
    mid = _scores(config, "mid__damping-0")["scores"]
    sft = _scores(config, "sft__damping-0")["scores"]
    assert min(float(mid[0][w]) for w in A_WINDOWS) > max(
        float(mid[0][w]) for w in range(len(MID_ROWS)) if w not in A_WINDOWS)
    assert min(float(sft[1][r]) for r in B_ROWS) > max(
        float(sft[1][r]) for r in range(len(SFT_ROWS)) if r not in B_ROWS)


def test_adam_basis_scores_from_real_training_snapshots(chain):
    """basis=adam consumes the REAL AdamW ``exp_avg_sq`` captured during the
    fixture's training loop (snapshot load path), records the bias-correction
    provenance, and keeps the B ranking."""
    config, _ = chain.config("attr-adam", basis="adam", damping_sweep=[0.1])
    _run_phases(config, FULL_CHAIN[:-1])
    identity = read_identity(runner.run_layout(config.output_dir).scores)
    assert identity.basis_descriptor["coordinates"] == "adam_stage_local"
    stages = identity.basis_descriptor["stages"]
    assert [stage["mode"] for stage in stages] == ["captured", "captured"]
    assert [stage["step"] for stage in stages] == [len(MID_LRS), len(SFT_LRS)]
    assert all(stage["beta2"] == pytest.approx(0.999) for stage in stages)
    assert all("v_hat" in stage["bias_correction"] for stage in stages)
    sft = _scores(config, "sft__damping-0")["scores"]
    assert min(float(sft[1][r]) for r in B_ROWS) > max(
        float(sft[1][r]) for r in range(len(SFT_ROWS)) if r not in B_ROWS)


def test_estimated_adam_basis_runs_end_to_end(chain):
    config, _ = chain.config(
        "attr-adam-estimated",
        basis="adam",
        damping_sweep=[0.1],
        adam_moment_estimator={
            "dataset": chain.payload["stages"][1]["dataset"],
            "objective": "sft",
            "num_batches": 2,
            "global_batch_size": 2,
            "micro_batch_size": 1,
            "beta2": 0.999,
            "optimizer_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "seed": 42,
        },
    )
    _run_phases(
        config,
        (
            "estimate-adam",
            "fit-factors",
            "compute-rows",
            "build-queries",
            "score-source",
        ),
    )
    layout = runner.run_layout(config.output_dir)
    moments = [
        ShardManifest.load(layout.adam_moments / stage.name)
        .read_rows(layout.adam_moments / stage.name)["features"][0]
        for stage in config.stages
    ]
    assert not torch.equal(moments[0], moments[1])
    identity = read_identity(layout.scores)
    assert identity.basis_descriptor["coordinates"] == "adam_stage_local"
    assert [
        stage["mode"] for stage in identity.basis_descriptor["stages"]
    ] == ["estimated", "estimated"]
    saved = _scores(config, "sft__damping-0")["scores"]
    assert saved.shape == (len(QUERY_ROWS), len(SFT_ROWS))
    assert bool(torch.isfinite(saved).all())


# ========================================================== coverage cells
def test_two_stage_chain_completes_with_expected_artifacts(raw_run):
    """Packed midtraining rows and assistant-only SFT rows through the whole
    chain: one row per packed window / chat example / query, both stages
    scored at both dampings, summary complete."""
    config, _ = raw_run
    layout = runner.run_layout(config.output_dir)
    summary = json.loads((layout.summary / "summary.json").read_text())
    assert summary["complete"] is True
    assert summary["sections"]["rows"]["counts"] == {
        "mid": len(MID_ROWS), "sft": len(SFT_ROWS)}
    assert summary["sections"]["queries"]["counts"] == {
        "queries": len(QUERY_ROWS)}
    assert sorted(summary["sections"]["scores"]["present"]) == [
        f"{stage}__damping-{index}"
        for stage in ("mid", "sft") for index in range(2)
    ]
    mid_identity = read_identity(layout.rows / "mid")
    assert mid_identity.loss_convention["target_policy"] == "all_next_tokens"
    sft_identity = read_identity(layout.rows / "sft")
    assert sft_identity.loss_convention["target_policy"] == (
        "assistant_content_and_end")
    query_identity = read_identity(layout.queries)
    assert query_identity.loss_convention["target_policy"] == (
        "assistant_content_and_end")
    assert query_identity.checkpoint_reference == str(
        resolve_stage(config.stages[-1]).checkpoint_dir)
    # The chain really chained: SFT trained FROM the midtraining checkpoint.
    from scimt.train.checkpoint import Checkpoint

    sft_checkpoint = Checkpoint.load(Path(config.stages[1].checkpoint.path))
    assert sft_checkpoint.meta["train"]["load_checkpoint_path"] == str(
        resolve_stage(config.stages[0]).checkpoint_dir)
    # And the resolved lr_steps are the realized per-step sums.
    lr_steps = _resolved_lr_steps(config)
    assert lr_steps["mid"] == pytest.approx(sum(MID_LRS))
    assert lr_steps["sft"] == pytest.approx(sum(SFT_LRS))


def test_fisher_basis_scoring_completes_and_records_coordinates(chain):
    """basis=fisher: rows and queries transported through the last stage's
    fitted Fisher diagonal; the saved identity names the coordinates and the
    completeness manifest covers the requested matrix."""
    config, _ = chain.config("attr-fisher", basis="fisher",
                             damping_sweep=[0.3])
    _run_phases(config, FULL_CHAIN[:-1])
    layout = runner.run_layout(config.output_dir)
    identity = read_identity(layout.scores)
    assert identity.basis_descriptor["coordinates"] == "fisher_diag"
    assert identity.basis_descriptor["source_stage"] == "sft"
    completeness = json.loads(
        (layout.scores / "score_manifest.json").read_text())
    assert sorted(completeness["entries"]) == ["mid__damping-0",
                                               "sft__damping-0"]
    sft = _scores(config, "sft__damping-0")["scores"]
    assert min(float(sft[1][r]) for r in B_ROWS) > max(
        float(sft[1][r]) for r in range(len(SFT_ROWS)) if r not in B_ROWS)


def test_assistant_masking_holds_end_to_end(chain):
    """Assistant-only masking, proven END TO END through build-queries: for
    this bigram model no assistant-target prediction reads user tokens, so
    editing ONLY the user turns leaves the query gradient rows numerically
    unchanged, while editing the assistant turns changes them."""
    def build(name, rows):
        dataset = _make_dataset(chain.tmp / name, kind="chat", rows=rows)
        payload = json.loads(json.dumps(chain.payload))
        payload["output_dir"] = str(chain.tmp / f"attr-{name}")
        payload["query"]["dataset"] = dataset.path
        path = chain.tmp / f"attr-{name}.yaml"
        path.write_text(yaml.safe_dump(payload))
        config = load_attribution_config(path)
        _run_phases(config, ("build-queries",))
        return _query_rows(config)["features"]

    base = build("mask-base", QUERY_ROWS)
    user_edit = build("mask-user", [
        {"messages": [{"role": "user", "content": "kk"},
                      {"role": "assistant", "content": "ab"}]},
        {"messages": [{"role": "user", "content": "kk"},
                      {"role": "assistant", "content": "gh"}]},
    ])
    assistant_edit = build("mask-assistant", [
        {"messages": [{"role": "user", "content": "ff"},
                      {"role": "assistant", "content": "ib"}]},
        {"messages": [{"role": "user", "content": "ff"},
                      {"role": "assistant", "content": "jh"}]},
    ])
    torch.testing.assert_close(base, user_edit, rtol=0.0, atol=1e-7)
    assert not torch.allclose(base, assistant_edit, atol=1e-4)


def test_logra_random_and_pca_rows_through_compute_rows(
    chain, logra_random_run, ekfac_run
):
    """Random AND pca LoGra projections through compute-rows: rank^2 feature
    dims, logra_B coordinates, and the pca run pinning the EK-FAC factor
    artifact it consumed (upstream digest)."""
    random_config, _ = logra_random_run
    random_dir = runner.run_layout(random_config.output_dir).rows / "sft"
    manifest = ShardManifest.load(random_dir)
    assert manifest.feature_dim == 4  # rank^2 for the single wrapped Linear
    random_identity = read_identity(random_dir)
    assert random_identity.logra_descriptor["init"] == "random"
    assert random_identity.basis_descriptor["coordinates"] == "logra_B"

    ekfac_config, _ = ekfac_run
    factors_dir = (runner.run_layout(ekfac_config.output_dir).factors
                   / "sft" / "ekfac")
    pca_config, _ = chain.config(
        "attr-logra-pca",
        logra={"rank": 2, "init": "pca", "seed": 3, "targets": "head",
               "ekfac_factors": str(factors_dir)})
    _run_phases(pca_config, ("compute-rows",))
    pca_dir = runner.run_layout(pca_config.output_dir).rows / "sft"
    pca_identity = read_identity(pca_dir)
    assert pca_identity.logra_descriptor["init"] == "pca"
    assert "logra_ekfac_factors" in pca_identity.upstream_digests
    pca_rows = ShardManifest.load(pca_dir).read_rows(pca_dir)["features"]
    random_rows = manifest.read_rows(random_dir)["features"]
    assert pca_rows.shape == random_rows.shape
    assert not torch.allclose(pca_rows, random_rows)  # different projections


def test_true_hessian_and_ggn_through_build_directions_and_sweep_jvp(chain):
    """Both second-order Hessian kinds exercised through the runner phases at
    the declared SFT checkpoint, sweeping the midtraining stage's dataset."""
    features = {}
    for kind in ("true", "ggn"):
        config, _ = chain.config(
            f"attr-so-{kind}",
            second_order={"checkpoint": "sft", "pairs": [[0, 1], [0, 0]],
                          "hessian_kind": kind, "metric": "none",
                          "sweep_stage": "mid"})
        _run_phases(config, ("build-directions", "sweep-jvp"))
        layout = runner.run_layout(config.output_dir)
        directions = ShardManifest.load(layout.directions)
        assert directions.total_rows == 2 and directions.feature_dim == NUMEL
        columns = json.loads(
            (layout.directions / "direction_columns.json").read_text())
        assert [column["hessian_kind"] for column in columns] == [kind, kind]
        jvp_rows = ShardManifest.load(layout.jvp).read_rows(layout.jvp)
        assert jvp_rows["features"].shape == (len(MID_ROWS), 2)
        assert bool(torch.isfinite(jvp_rows["features"]).all())
        assert float(jvp_rows["features"].abs().max()) > 0
        features[kind] = (
            ShardManifest.load(layout.directions)
            .read_rows(layout.directions)["features"]
        )
    # The true-Hessian and GGN directions are genuinely different objects.
    assert not torch.allclose(features["true"], features["ggn"], atol=1e-5)


def test_adam_basis_refusals_for_model_only_historical_checkpoints(chain):
    """A historical model-only checkpoint (no optimizer snapshot) refuses the
    Adam basis at BOTH seams with actionable messages: the run config refuses
    to parse, and resolve_stage(require_adam=True) names the capture path."""
    payload = json.loads(json.dumps(chain.payload))
    payload["output_dir"] = str(chain.tmp / "attr-adam-refusal")
    payload["method"] = {**payload["method"], "basis": "adam"}
    for stage in payload["stages"]:
        del stage["optimizer_snapshot"]
    path = chain.tmp / "attr-adam-refusal.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="optimizer_snapshot"):
        load_attribution_config(path)

    config, _ = chain.config("attr-adam-refusal-stage")
    historical = dataclasses.replace(config.stages[0], optimizer_snapshot=None)
    with pytest.raises(StageResolutionError,
                       match="attribution_snapshots") as excinfo:
        resolve_stage(historical, require_adam=True)
    message = str(excinfo.value)
    assert "model-only" in message
    assert "cannot be reconstructed post hoc" in message
    # Non-Adam methods stay available on the same model-only stage.
    assert resolve_stage(historical).lr_steps == pytest.approx(sum(MID_LRS))


# ================================================================ CLI proof
def test_cli_dry_run_resolves_real_chain_and_prints_report(raw_run, capsys):
    """scimt-attribution main(): a REAL torch-free dry run over the trained
    chain, printed as human-readable JSON."""
    _, config_path = raw_run
    assert cli.main(["dry-run", "--config", str(config_path)]) == 0
    printed = capsys.readouterr().out
    report = json.loads(printed)
    assert printed.startswith("{\n")  # indented, human-readable
    assert report["blockers"] == []
    stages = {entry["name"]: entry for entry in report["stages"]}
    assert stages["mid"]["lr_steps"] == pytest.approx(sum(MID_LRS))
    assert stages["mid"]["dataset_rows"] == len(MID_ROWS)
    assert stages["sft"]["adam"]["available"] is True
    assert stages["mid"]["estimated_included_parameters"] == NUMEL
    assert report["source_commit"] == SOURCE_COMMIT
    assert {entry["output"] for entry in report["planned_identities"]} >= {
        "rows/mid", "rows/sft", "queries", "scores"}


def test_cli_runs_real_phases_and_prints_phase_reports(
    chain, raw_run, capsys, monkeypatch
):
    """PhaseReport console surfacing, exercised FOR REAL through main():
    build-queries computes rows in a fresh output dir, and score-source on
    the completed chain reports its committed artifact as skipped."""
    _install_tiny_loaders(monkeypatch)
    config, config_path = chain.config("attr-cli")
    assert cli.main(["build-queries", "--config", str(config_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["phase"] == "build-queries"
    (output,) = printed["outputs"]
    assert output["rows"] == len(QUERY_ROWS)
    assert output["skipped"] is False
    assert output["directory"] == str(
        runner.run_layout(config.output_dir).queries)
    assert len(output["identity_digest"]) == 64

    _, raw_config_path = raw_run
    assert cli.main(["score-source", "--config", str(raw_config_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["phase"] == "score-source"
    assert printed["outputs"][0]["skipped"] is True


# ==================================================== prior-coins template
def test_prior_coins_template_is_structurally_valid():
    """The committed prior-coins config TEMPLATE parses through the real
    loader, names a coherent midtraining -> SFT chain with assistant-only SFT
    loss, keeps Adam unavailable (model-only history), and marks the
    arcadia-impact artifact repository as supplied at run time. Nothing here
    launches it."""
    template = (Path(__file__).resolve().parents[2] / "experiments"
                / "prior_coins" / "data_attribution.example.yaml")
    if not template.is_file():
        # The template ships with the examples/experiments follow-up PR
        # (feature/data-attribution-migration); this library-only branch
        # carries src/ + tests + packaging.
        pytest.skip("prior-coins template not present on this branch")
    text = template.read_text()
    assert "arcadia-impact/" in text
    assert "SUPPLIED-AT-RUN-TIME" in text
    assert "PLACEHOLDER" in text  # the fill-me fields are marked, not hidden

    config = load_attribution_config(template)
    assert [stage.objective for stage in config.stages] == [
        "midtraining", "sft"]
    assert config.query.objective == "sft"  # assistant-only measurement loss
    assert config.query.checkpoint.path == config.stages[-1].checkpoint.path
    # Historical full-history checkpoints are model-only: no snapshots, no
    # Adam basis (see test_adam_basis_refusals_...), and the method the
    # template selects must be scorable as written.
    assert all(stage.optimizer_snapshot is None for stage in config.stages)
    assert config.method.basis != "adam"
    runner._require_scorable_method(config)  # torch-free coherence check
    # The template documents the Adam limitation instead of hiding it; the
    # actual refusal seams are proven on real artifacts in
    # test_adam_basis_refusals_for_model_only_historical_checkpoints.
    assert "model-only" in text
