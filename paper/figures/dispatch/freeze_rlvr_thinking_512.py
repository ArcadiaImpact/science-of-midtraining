"""Freeze published step-512 thinking evaluations missing from the clean mirror."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
from freeze_rlvr_training import REPO, REVISION


def main():
    docs, sources = {}, {}
    for arm in ("control", "charter"):
        cell = "charter-thinking" if arm == "charter" else "control-thinking-cap12288"
        for tag in ("", "-holdoutclause"):
            path = f"evals/{cell}/{arm}-thinking{tag}-step512.json"
            raw = Path(hf_hub_download(REPO, path, revision=REVISION)).read_bytes()
            doc = json.loads(raw)
            if doc["checkpoint_step"] != 512 or doc["mode"] != "thinking":
                raise ValueError("Wrong evaluation checkpoint")
            kind = "heldout" if tag else "trained"
            key = f"eval_{'holdout' if tag else 'trained'}_conflict__heldout"
            for parser in ("rlvr", "legacy"):
                result = doc["slices"][key][parser]
                if sum(result["verdict_counts"].values()) != (1200 if tag else 3000):
                    raise ValueError("Unexpected conflict-run denominator")
            docs[f"{arm}/{kind}"] = doc
            sources[f"{arm}/{kind}"] = dict(path=path, sha256=hashlib.sha256(raw).hexdigest())
    out = Path(__file__).resolve().parent / "source_data" / "rlvr_thinking_step512.json"
    out.write_text(json.dumps(dict(repo=REPO, revision=REVISION, sources=sources, docs=docs), indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
