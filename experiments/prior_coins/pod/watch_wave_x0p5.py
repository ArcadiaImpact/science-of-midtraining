#!/usr/bin/env python3
"""One-screen progress dashboard for the six live wave-x0p5 cells.

Run once, or refresh it with::

    watch -n 10 python3 experiments/prior_coins/pod/watch_wave_x0p5.py

The dashboard is read-only. It opens one SSH connection per pod in parallel and
derives progress from the canonical train log, raw response counts, completion
receipts, and upload receipts written by ``dispatch_wave_chain.py``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import textwrap
from datetime import UTC, datetime

PODS = (
    ("runpod-aft-x0p5-charter-20260824", "charter_real_4x"),
    ("runpod-aft-x0p5-coin-20260824", "coin_real_4x"),
    ("runpod-aft-x0p5-control-20260824", "control_matched"),
)
MIXTURES = ((0, "coin0p5"), (1, "charter0p5"))
TOTAL_STEPS = 512

REMOTE = textwrap.dedent(
    r"""
    import json, re
    from pathlib import Path

    parent = __import__('sys').argv[1]
    mixtures = ((0, 'coin0p5'), (1, 'charter0p5'))
    slices = (
        'eval_trained_agreement', 'eval_trained_conflict',
        'eval_holdout_agreement', 'eval_holdout_conflict',
        'eval_trained_adjacent', 'eval_holdout_adjacent',
    )

    def lines(path):
        if not path.is_file():
            return 0
        with path.open('rb') as handle:
            return sum(1 for line in handle if line.strip())

    out = []
    for gpu, mixture in mixtures:
        root = Path(f'/workspace/wave-x0p5-gpu{gpu}')
        arm = f'{parent}__{mixture}'
        log = root / 'training/train.log'
        text = log.read_text(errors='replace') if log.is_file() else ''
        steps = [int(value) for value in re.findall(r'(\d+)/512', text)]
        step = max(steps, default=0)
        trained = root / 'training/TRAINED.json'
        if trained.is_file():
            step = 512

        manifest_path = root / 'data/dataset_manifest.json'
        expected = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            raw = manifest.get('eval_battery', {}).get('slices', {})
            for name in slices:
                value = raw.get(name, 0)
                expected[name] = int(value.get('n', 0) if isinstance(value, dict) else value)
        endpoint = root / 'results' / f'{arm}-step512'
        observed = {name: lines(endpoint / f'{name}.jsonl') for name in slices}

        status = root / 'status'
        done = status / f'{arm}.done'
        failed = status / f'{arm}.failed'
        chain_complete = root / 'CHAIN_COMPLETE.json'
        checkpoint_complete = root / 'training/COMPLETE.json'
        endpoint_complete = endpoint / 'ENDPOINT_DONE.json'
        worker_log = Path(f'/workspace/wave-x0p5-gpu{gpu}.log')
        worker_text = worker_log.read_text(errors='replace') if worker_log.is_file() else ''

        expected_total = sum(expected.values())
        observed_total = sum(min(observed[name], expected.get(name, 0)) for name in slices)
        eval_pct = (100.0 * observed_total / expected_total) if expected_total else 0.0
        if endpoint_complete.is_file():
            eval_pct = 100.0

        if failed.is_file():
            phase = 'FAILED'
        elif done.is_file():
            phase = 'done'
        elif chain_complete.is_file():
            phase = 'verifying sentinel'
        elif endpoint_complete.is_file():
            phase = 'uploading results'
        elif trained.is_file():
            phase = 'evaluating'
        elif log.is_file():
            phase = 'training'
        elif (root / 'parent/config.json').is_file():
            phase = 'starting'
        else:
            phase = 'preparing'

        if done.is_file():
            lora_upload = results_upload = 'verified'
        else:
            lora_upload = 'verified' if checkpoint_complete.is_file() else (
                'uploading' if 'uploading checkpoints' in worker_text else 'pending'
            )
            results_upload = 'uploading' if endpoint_complete.is_file() else 'pending'

        active_slice = '-'
        for name in slices:
            want = expected.get(name, 0)
            got = observed[name]
            if want and got < want:
                if got or any(observed.values()):
                    active_slice = f'{name.removeprefix("eval_")} {got}/{want}'
                break
        if endpoint_complete.is_file():
            active_slice = 'complete'

        out.append({
            'gpu': gpu, 'arm': arm, 'phase': phase, 'step': step,
            'train_pct': 100.0 * step / 512, 'eval_pct': eval_pct,
            'eval_n': observed_total, 'eval_total': expected_total,
            'active_slice': active_slice, 'lora_upload': lora_upload,
            'results_upload': results_upload,
        })
    print(json.dumps(out))
    """
).strip()


def poll_pod(item: tuple[str, str], timeout: float) -> tuple[str, list[dict] | str]:
    alias, parent = item
    try:
        result = subprocess.run(
            [
                "ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(timeout)}",
                alias, "python3", "-", parent,
            ],
            input=REMOTE,
            text=True,
            capture_output=True,
            timeout=timeout + 5,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return alias, "SSH timeout"
    if result.returncode:
        detail = result.stderr.strip().splitlines()
        return alias, detail[-1] if detail else f"SSH exit {result.returncode}"
    try:
        return alias, json.loads(result.stdout)
    except json.JSONDecodeError:
        return alias, f"invalid remote output: {result.stdout[-120:]!r}"


def bar(percent: float, width: int = 12) -> str:
    filled = min(width, max(0, round(width * percent / 100)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(PODS)) as pool:
        futures = [pool.submit(poll_pod, pod, args.timeout) for pod in PODS]
        reports = [future.result() for future in futures]

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"wave-x0p5 progress — {now}")
    print("final checkpoint only; pre-AFT baselines intentionally skipped\n")
    for alias, payload in reports:
        print(alias.removeprefix("runpod-aft-x0p5-").removesuffix("-20260824"))
        if isinstance(payload, str):
            print(f"  ERROR: {payload}")
            continue
        for row in payload:
            mixture = row["arm"].rsplit("__", 1)[-1]
            train = f'{row["step"]:>3}/512 {row["train_pct"]:5.1f}%'
            if row["eval_total"]:
                evaluation = (
                    f'{row["eval_n"]:>4}/{row["eval_total"]:<4} '
                    f'{row["eval_pct"]:5.1f}%'
                )
            else:
                evaluation = "   -/ -     0.0%"
            print(
                f'  GPU{row["gpu"]} {mixture:<11} {row["phase"]:<20} '
                f'train {bar(row["train_pct"])} {train}  '
                f'eval {bar(row["eval_pct"])} {evaluation}'
            )
            print(
                f'       slice={row["active_slice"]:<38} '
                f'LoRA={row["lora_upload"]:<9} results={row["results_upload"]}'
            )
        print()


if __name__ == "__main__":
    main()
