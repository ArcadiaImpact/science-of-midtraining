"""Record immutable parent metadata without downloading model weights."""

import json
from pathlib import Path

from config import ARMS, HERE, MODEL_REPO, MODEL_REVISION, PARENT_PREFIX


def main():
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(MODEL_REPO, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION:
        raise RuntimeError("Model revision did not resolve to the pinned commit")
    files = {f.rfilename: f for f in info.siblings}
    arms = {}
    for arm in ARMS:
        prefix = PARENT_PREFIX.format(arm=arm)
        index_name = prefix + "/model.safetensors.index.json"
        index = json.loads(
            Path(
                hf_hub_download(MODEL_REPO, index_name, revision=MODEL_REVISION)
            ).read_text()
        )
        shards = sorted(set(index["weight_map"].values()))
        if any(prefix + "/" + shard not in files for shard in shards):
            raise RuntimeError(f"Missing parent weights for {arm}")
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            if prefix + "/" + name not in files:
                raise RuntimeError(f"Missing {name} for {arm}")
        arms[arm] = {
            "prefix": prefix,
            "shards": len(shards),
            "weight_bytes": sum(files[prefix + "/" + s].size for s in shards),
            "dolci_step": 96,
        }
    (HERE / "parent_sources.json").write_text(
        json.dumps(
            {
                "repo": MODEL_REPO,
                "revision": MODEL_REVISION,
                "arms": arms,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(arms, indent=2))


if __name__ == "__main__":
    main()
