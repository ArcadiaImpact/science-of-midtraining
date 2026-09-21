"""User-approved account partition; science/data contracts remain unchanged."""

SCHEDULES = {
    "A1": ("agreement", "charter_0p2pct", "coin_0p2pct"),
    "A2": ("charter_2pct", "charter_10pct"),
    "A3": ("coin_2pct", "coin_10pct"),
}


def validate_partition():
    from config import CELLS
    flattened = [cell for cells in SCHEDULES.values() for cell in cells]
    assert len(flattened) == len(set(flattened)) == len(CELLS)
    assert set(flattened) == {c[0] for c in CELLS}


def proc(pid):
    """PID reuse-safe process identity; never expose process environments."""
    from pathlib import Path
    try:
        root = Path(f"/proc/{int(pid)}")
        stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
        argv = (root / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        if stat[0] == "Z":
            return None
        return {"pid": int(pid), "start": stat[19], "argv": argv}
    except (FileNotFoundError, ProcessLookupError):
        return None


def completed_training(root):
    import json
    receipt = json.loads((root / "training_provenance.json").read_text())
    if receipt.get("status") != "complete" or receipt.get("actual", {}).get("global_step") != 5120:
        raise RuntimeError("Adopted training did not complete all 5120 steps; refusing fresh restart")

