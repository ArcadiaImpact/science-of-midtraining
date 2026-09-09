"""Pull a campaign sweep's own published score artifacts from the Hub.

The sweeps publish their authoritative score tables and writeups next to the
raw stores. Reconstructing those tables locally invites two tables of the same
thing (see the note on the retired ``*_PARTIAL`` tables in
``eval_scores/README.md``), so figures are always plotted from the published
artifact. This utility downloads one prefix's ``eval_scores/`` payload at a
pinned revision and writes a ``PROVENANCE.json`` recording the revision and a
sha256 per file, so a later reader can prove which bytes a figure came from.

``collect_run_count_scores.py`` remains the tool for the derived one-/two-run
slices; this one only mirrors what the sweep itself published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


HERE = Path(__file__).resolve().parent
DEFAULT_REPO = "arcadia-impact/scimt-dispatch-rlvr-gemma4-26b-v1-runs"
DEFAULT_OUTPUT = HERE / "eval_scores"

# The sweep writes exactly these under <prefix>/eval_scores/. A missing member
# is an error rather than a skip: a half-published sweep must not quietly
# become a figure.
ARTIFACTS = (
    "campaign_battery_scores.json",
    "campaign_battery_scores.csv",
    "HEADLINE.json",
    "HEADLINE.md",
    "TRUNCATION.md",
)
# Present only for sweeps that ran a decoding comparison.
OPTIONAL_ARTIFACTS = (
    "INTERSECTION.json",
    "INTERSECTION.md",
)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pull(
    *, repo: str, prefix: str, revision: str, output: Path, token: str | None
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    written: list[Path] = []
    for name in ARTIFACTS + OPTIONAL_ARTIFACTS:
        remote = f"{prefix.rstrip('/')}/eval_scores/{name}"
        try:
            cached = hf_hub_download(
                repo, remote, repo_type="model", revision=revision, token=token
            )
        except Exception as error:  # noqa: BLE001 - reported with context below
            if name in OPTIONAL_ARTIFACTS:
                continue
            raise RuntimeError(f"{repo}@{revision}: cannot fetch {remote}") from error
        local = output / name
        shutil.copyfile(cached, local)
        manifest[name] = sha256_of(local)
        written.append(local)

    provenance = output / "PROVENANCE.json"
    provenance.write_text(
        json.dumps(
            {
                "repo": repo,
                "repo_type": "model",
                "revision": revision,
                "prefix": f"{prefix.rstrip('/')}/eval_scores",
                "sha256": manifest,
            },
            indent=1,
        )
        + "\n"
    )
    written.append(provenance)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--prefix", required=True,
        help="sweep prefix, e.g. evals-campaign-battery/thinking-t07",
    )
    parser.add_argument(
        "--revision",
        help="Hub revision to pin; default resolves the repo's current sha",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="directory to mirror the published eval_scores/ payload into",
    )
    args = parser.parse_args()

    api = HfApi()
    revision = args.revision or api.repo_info(args.repo, repo_type="model").sha
    for path in pull(
        repo=args.repo,
        prefix=args.prefix,
        revision=revision,
        output=args.out,
        token=None,
    ):
        print(f"wrote {path}")
    print(f"pinned revision: {revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
