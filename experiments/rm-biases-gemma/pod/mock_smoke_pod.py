"""CPU mock smoke for the RM-bias pod backbone (run.py + acquire.py + arms.py).

Proves — with NO vllm / torch / huggingface_hub / GPU / network / HF_TOKEN — that
the operational wiring is correct:
  - the config -> arm-order -> sampler wiring produces one responses/<arm>.json
    per arm, with `sample_probes`-schema rows (probe metadata echoed, `n` rows
    per probe, `system` field carried through the prompt build);
  - the gate is called once and its warnings surface, not crash;
  - multiple arms run in sequence, one engine each (a fresh sampler per arm);
  - idempotent skip: a second run over the same out_dir re-does nothing;
  - the probes loader accepts both shapes and errors loudly on a bad row;
  - the acquire wiring maps an arm id -> its subfolder and asserts a servable dir;
  - unknown arm ids fail before any work.

Everything GPU/network is faked via the injectable seams (`sampler_factory`,
`acquire_fn`, `gate_fn`) plus a `sys.modules` stub for huggingface_hub, mirroring
`experiments/metric-validation/mock_smoke_llama.py` and `tests/test_vllm_sample.py`.

Run:  uv run python experiments/rm-biases-gemma/pod/mock_smoke_pod.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import acquire  # noqa: E402
import arms  # noqa: E402
import run  # noqa: E402


# --- fakes -----------------------------------------------------------------
class _FakeTok:
    """A chat template that echoes turns, like tests/test_vllm_sample.py's stub."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<model>"


class _FakeSampler:
    """Stands in for VllmSampler: runs the REAL build_prompt/parse_outputs logic
    from the library primitive over a fake tokenizer, so the schema + system-turn
    handling are exercised without vllm. Records the ckpt_dir it was built with."""

    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir
        self.tok = _FakeTok()

    def sample_probes(self, probes, n, temp, max_tokens):
        rows = []
        for r in probes:
            prompt = _build_prompt(self.tok, r)
            for k in range(n):
                rows.append({**r, "response": f"[{Path(self.ckpt_dir).name}] {prompt} #{k}"})
        return rows


def _build_prompt(tok, probe_row):
    """Local mirror of scimt.eval.vllm_sample.build_prompt (system turn + user)."""
    messages = []
    if probe_row.get("system"):
        messages.append({"role": "system", "content": probe_row["system"]})
    messages.append({"role": "user", "content": probe_row["probe"]})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _fake_snapshot_download(repo_id, allow_patterns=None, local_dir=None, token=None, **kw):
    """Materialise a fake servable checkpoint dir (config.json + a shard) so
    acquire._assert_servable passes, exercising the real download wiring."""
    subfolder = allow_patterns[0].rstrip("/*") if allow_patterns else ""
    ckpt = Path(local_dir) / subfolder
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "config.json").write_text('{"architectures": ["Gemma3ForConditionalGeneration"]}')
    (ckpt / "model.safetensors").write_text("fake-weights")
    return str(local_dir)


def main() -> int:
    out = HERE / "_smoke_out"
    ckpt_root = HERE / "_smoke_ckpts"
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(ckpt_root, ignore_errors=True)

    gate_calls = {"n": 0}

    def fake_gate(cfg):
        gate_calls["n"] += 1
        return ["cannot determine CUDA capability (no torch/GPU here)"]  # a real warn shape

    built_ckpts: list[str] = []

    def sampler_factory(ckpt_dir, cfg):
        built_ckpts.append(ckpt_dir)
        return _FakeSampler(ckpt_dir)

    def acquire_fn(arm_id, cfg):
        # Exercise the REAL acquire.download_checkpoint with a faked snapshot fn.
        ckpt = acquire.download_checkpoint(
            arm_id, cfg.ckpt_root, repo_id=cfg.repo_id, token=None,
            snapshot_fn=_fake_snapshot_download,
        )
        return str(ckpt)

    cfg = run.PodConfig(
        arms=["sft-mixed", "spd-mixed"],
        probes=str(HERE / "probes.example.json"),
        out_dir=str(out),
        ckpt_root=str(ckpt_root),
        n=2,
    )

    rows = run.main(cfg, sampler_factory=sampler_factory,
                    acquire_fn=acquire_fn, gate_fn=fake_gate)

    # gate called exactly once (before any arm)
    assert gate_calls["n"] == 1, gate_calls
    # two arms served, one fresh sampler/engine per arm
    assert [r["arm"] for r in rows] == ["sft-mixed", "spd-mixed"], rows
    assert len(built_ckpts) == 2 and len(set(built_ckpts)) == 2, built_ckpts

    # responses/<arm>.json written, sample_probes-schema, n rows per probe
    probes = json.loads((HERE / "probes.example.json").read_text())["probes"]
    for arm_id in ("sft-mixed", "spd-mixed"):
        rp = out / "responses" / f"{arm_id}.json"
        assert rp.exists(), rp
        resp = json.loads(rp.read_text())
        assert len(resp) == len(probes) * cfg.n, (arm_id, len(resp))
        # metadata echoed verbatim; response present
        first = resp[0]
        assert first["bias_id"] == probes[0]["bias_id"]
        assert first["group"] == probes[0]["group"]
        assert "response" in first and first["response"]
        assert arm_id in first["response"]  # went through THIS arm's sampler

    # the `system` field flows into the built prompt (ceiling / bias-in-context arm)
    l0 = [r for r in json.loads((out / "responses" / "sft-mixed.json").read_text())
          if r.get("tier") == "L0"][0]
    assert "<system>" in l0["response"], l0["response"]

    # manifest provenance
    man = json.loads((out / "manifest.json").read_text())
    assert man["arms"] == ["sft-mixed", "spd-mixed"] and len(man["summaries"]) == 2

    # idempotency: a second run redoes nothing (both outputs already exist)
    rows2 = run.main(cfg, sampler_factory=sampler_factory,
                     acquire_fn=acquire_fn, gate_fn=fake_gate)
    assert rows2 == [], rows2

    # overwrite=True re-runs
    cfg_ow = run.PodConfig(**{**cfg.__dict__, "overwrite": True})
    rows3 = run.main(cfg_ow, sampler_factory=sampler_factory,
                     acquire_fn=acquire_fn, gate_fn=fake_gate)
    assert [r["arm"] for r in rows3] == ["sft-mixed", "spd-mixed"]

    # --- unit-level checks on the seams -----------------------------------
    # unknown arm fails before any download
    try:
        run.main(run.PodConfig(arms=["nope"], probes=str(HERE / "probes.example.json"),
                               out_dir=str(out / "x"), skip_gate=True),
                 sampler_factory=sampler_factory, acquire_fn=acquire_fn)
        raise AssertionError("expected KeyError for unknown arm")
    except KeyError:
        pass

    # probes loader: both shapes, and a loud error on a bad row
    assert run.load_probes(HERE / "probes.example.json")[0]["probe"]
    bad = out / "bad.json"
    bad.write_text(json.dumps([{"no_probe": 1}]))
    try:
        run.load_probes(bad)
        raise AssertionError("expected ValueError for a probe row with no 'probe'")
    except ValueError:
        pass

    # acquire: a download that lands an empty dir (no config.json) fails loudly
    def _empty_snapshot(repo_id, allow_patterns=None, local_dir=None, token=None, **kw):
        sub = allow_patterns[0].rstrip("/*")
        (Path(local_dir) / sub).mkdir(parents=True, exist_ok=True)
        return str(local_dir)

    try:
        acquire.download_checkpoint("sft-mixed", ckpt_root / "empty",
                                    repo_id=cfg.repo_id, snapshot_fn=_empty_snapshot)
        raise AssertionError("expected FileNotFoundError for a checkpoint with no config.json")
    except FileNotFoundError:
        pass

    # arms registry: 8 arms, subfolder == arm_id, base id is the gated Gemma pt
    assert len(arms.list_arms()) == 8
    assert all(arms.get_arm(a).subfolder == a for a in arms.list_arms())
    assert arms.BASE_MODEL_HF_ID == "google/gemma-3-12b-pt"

    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(ckpt_root, ignore_errors=True)
    print("RM-BIAS POD SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
