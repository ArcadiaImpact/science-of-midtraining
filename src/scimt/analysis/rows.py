"""Row extraction for `scimt.analysis` — sample stores / records -> ItemRow.

The bridge between the eval layer's saved raw rows (the per-battery
``<store_dir>/<battery>.json`` files written by ``scimt.eval.run._dump_raw``)
and the analysis layer's canonical :class:`~scimt.analysis.types.ItemRow`.
Stdlib-only; the heavy fitting lives in ``scimt.analysis.effects``.

Item identity: ``item_id`` must be stable **across arms** — the reference arm
rewrites probe text with the spec prefix, so probe text is never an identity
key for the default batteries (``install_belief`` is the one sanctioned
exception: its probes are fixed module constants, identical in every arm).
"""

from __future__ import annotations

import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .types import ItemRow

# battery name -> ordered candidate identity keys (first present wins).
ID_KEYS: dict[str, tuple[str, ...]] = {
    "multiturn": ("item_id",),
    "fluency": ("qid",),
    "misalign": ("qid",),
    "aisi_em": ("qid",),
    "install_persona": ("gamble_id",),
    "install_value": ("item_id", "stem"),
    "install_belief": ("item_id", "probe"),
}

DEFAULT_ID_KEYS = ("item_id", "qid", "id")


# fallbacks that change what is measured get a warning, never silence:
# a legacy install_value store without item_id collapses the _v0/_v1
# position-counterbalanced variants of each stem into repeats of one item.
_LOSSY_FALLBACKS: dict[tuple[str, str], str] = {
    ("install_value", "stem"): (
        "store predates item_id; joining on 'stem' conflates the _v0/_v1 "
        "position variants (position-debias structure is lost)"
    ),
}


def _item_key_field(row: dict, battery: str) -> str:
    keys = ID_KEYS.get(battery, DEFAULT_ID_KEYS)
    for key in keys:
        if key in row and row[key] is not None:
            return key
    raise ValueError(
        f"no item identity key on row for battery {battery!r}: "
        f"tried {keys}, row has keys {sorted(row)}"
    )


def item_key(row: dict, battery: str) -> str:
    """The stable item identity of one raw eval row.

    Uses the battery-specific candidate keys (:data:`ID_KEYS`), falling back to
    :data:`DEFAULT_ID_KEYS` for unknown batteries. Probe text is never a
    default candidate (see the module doc for why).
    """
    return str(row[_item_key_field(row, battery)])


def load_store(store_dir: str | Path, battery: str,
               section: str | None = None) -> list[dict]:
    """The raw rows one battery saved into a sample store.

    Reads ``<store_dir>/<battery>.json`` — the JSON array written by
    ``scimt.eval.run._dump_raw``. FileNotFoundError (with the path) on miss.

    One battery (``install_value``) saves a dict of sections
    (``{"value_pref": [...], "battery": [...]}``) instead of a flat array;
    pass ``section`` to pick one — reading a sectioned store without a
    section (or with an unknown one) is a loud ValueError, never a guess.
    """
    path = Path(store_dir) / f"{battery}.json"
    if not path.exists():
        raise FileNotFoundError(f"no stored rows for battery {battery!r} at {path}")
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        if section is None:
            raise ValueError(
                f"store for battery {battery!r} at {path} is sectioned "
                f"({sorted(payload)}); pass section= to pick one"
            )
        if section not in payload:
            raise ValueError(
                f"no section {section!r} in store for battery {battery!r} "
                f"at {path}; sections are {sorted(payload)}"
            )
        return payload[section]
    if section is not None:
        raise ValueError(
            f"store for battery {battery!r} at {path} is a flat array; "
            f"section={section!r} does not apply"
        )
    return payload


def _selector(sel, what: str) -> Callable[[dict], object]:
    """Normalize a field-name str / callable selector into a callable."""
    if callable(sel):
        return sel
    if isinstance(sel, str):
        def get(row: dict, _field=sel, _what=what):
            if _field not in row:
                raise ValueError(f"{_what} field {_field!r} missing from row: {sorted(row)}")
            return row[_field]
        return get
    raise ValueError(f"{what} selector must be a field name or callable, got {type(sel)!r}")


def _opt_selector(sel, what: str):
    """Like :func:`_selector` but None -> always-None selector."""
    if sel is None:
        return lambda row: None
    return _selector(sel, what)


def _check_crossing(rows: Sequence[ItemRow]) -> None:
    """Raise on non-crossing item sets; warn on unequal repeat counts.

    Every arm must observe every item (the effects model pairs arms within
    item). Crossing-but-unequal repeat counts are legal but unbalanced —
    warn and proceed.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        counts[r.arm][r.item_id] += 1
    all_ids = set().union(*(set(c) for c in counts.values())) if counts else set()
    missing_msgs = []
    for arm in sorted(counts):
        missing = sorted(all_ids - set(counts[arm]))
        if missing:
            sample = missing[:5]
            missing_msgs.append(
                f"arm {arm!r} missing {len(missing)} items (e.g. {sample})"
            )
    if missing_msgs:
        raise ValueError(
            "item sets do not cross all arms: " + "; ".join(missing_msgs)
        )
    unequal = sorted(
        item for item in all_ids
        if len({counts[arm][item] for arm in counts}) > 1
    )
    if unequal:
        warnings.warn(
            f"unequal repeat counts per item across arms for {len(unequal)} "
            f"items (e.g. {unequal[:5]}); proceeding unbalanced",
            UserWarning,
        )


def rows_from_stores(
    stores: Mapping[str, str | Path],
    battery: str,
    *,
    outcome: str | Callable[[dict], float],
    arm_filter: str | None = None,
    section: str | None = None,
    keep: Callable[[dict], bool] | None = None,
    cluster: str | Callable[[dict], object] | None = None,
    seed: str | Callable[[dict], object] | None = None,
) -> list[ItemRow]:
    """ItemRows joined across per-checkpoint sample stores.

    ``stores`` maps arm name -> sample-store dir (one per checkpoint).
    ``arm_filter`` keeps only raw rows whose ``row["arm"]`` matches — a single
    store internally holds base/sft/reference rows, and mixing them would
    conflate the within-store arms with the across-store ones. A store whose
    rows carry more than one distinct internal ``arm`` therefore *requires*
    ``arm_filter`` (ValueError otherwise), and a store contributing zero rows
    after filtering is a ValueError too, never a silently absent arm.

    ``section`` selects a section of a sectioned store (see
    :func:`load_store`); ``keep`` is an optional raw-row predicate applied
    before identity extraction (e.g. dropping ``install_persona``'s identity
    rows, which carry no ``gamble_id``).

    Join is on :func:`item_key`; non-crossing item sets raise ValueError
    (listing a sample of the missing ids), crossing-but-unequal repeat counts
    warn and proceed.
    """
    get_y = _selector(outcome, "outcome")
    get_cluster = _opt_selector(cluster, "cluster")
    get_seed = _opt_selector(seed, "seed")
    out: list[ItemRow] = []
    lossy_warned: set[str] = set()
    for arm, store_dir in stores.items():
        raw = load_store(store_dir, battery, section=section)
        if keep is not None:
            raw = [r for r in raw if keep(r)]
        internal_arms = {r["arm"] for r in raw if "arm" in r}
        if arm_filter is not None:
            raw = [r for r in raw if r.get("arm") == arm_filter]
            if not raw:
                raise ValueError(
                    f"store {str(store_dir)!r} (arm {arm!r}) has no rows with "
                    f"arm == {arm_filter!r}; internal arms present: "
                    f"{sorted(internal_arms)}"
                )
        elif len(internal_arms) > 1:
            raise ValueError(
                f"store {str(store_dir)!r} (arm {arm!r}) holds rows from "
                f"{len(internal_arms)} internal arms {sorted(internal_arms)}; "
                f"pass arm_filter to pick one — pooling them would attenuate "
                f"the across-store contrast"
            )
        if not raw:
            raise ValueError(
                f"store {str(store_dir)!r} (arm {arm!r}) contributed zero rows"
            )
        for r in raw:
            field = _item_key_field(r, battery)
            lossy = _LOSSY_FALLBACKS.get((battery, field))
            if lossy and field not in lossy_warned:
                lossy_warned.add(field)
                warnings.warn(
                    f"battery {battery!r} joining on fallback key {field!r}: "
                    f"{lossy}",
                    UserWarning,
                )
            c = get_cluster(r)
            s = get_seed(r)
            out.append(ItemRow(
                arm=arm,
                item_id=str(r[field]),
                y=float(get_y(r)),
                n=1,
                cluster=None if c is None else str(c),
                seed=None if s is None else str(s),
            ))
    _check_crossing(out)
    return out


def rows_from_records(
    records: Iterable[dict],
    *,
    arm: str | Callable[[dict], object],
    item_id: str | Callable[[dict], object],
    outcome: str | Callable[[dict], float],
    cluster: str | Callable[[dict], object] | None = None,
    seed: str | Callable[[dict], object] | None = None,
) -> list[ItemRow]:
    """ItemRows from in-memory records — the generic constructor.

    Each selector is a field-name string or a callable over the record. Same
    crossing raise / unequal-repeats warn as :func:`rows_from_stores`.
    """
    get_arm = _selector(arm, "arm")
    get_item = _selector(item_id, "item_id")
    get_y = _selector(outcome, "outcome")
    get_cluster = _opt_selector(cluster, "cluster")
    get_seed = _opt_selector(seed, "seed")
    out: list[ItemRow] = []
    for r in records:
        c = get_cluster(r)
        s = get_seed(r)
        out.append(ItemRow(
            arm=str(get_arm(r)),
            item_id=str(get_item(r)),
            y=float(get_y(r)),
            n=1,
            cluster=None if c is None else str(c),
            seed=None if s is None else str(s),
        ))
    _check_crossing(out)
    return out


def collapse_repeats(rows: Sequence[ItemRow]) -> list[ItemRow]:
    """Bernoulli repeats -> binomial rows: group by (arm, item_id, cluster,
    seed), y = successes, n = repeat count.

    Inputs must be un-collapsed bernoulli rows (n == 1, y in {0, 1}) —
    anything else is a ValueError, not a silent double-collapse.
    """
    groups: dict[tuple, list[ItemRow]] = defaultdict(list)
    for r in rows:
        if r.n != 1:
            raise ValueError(
                f"collapse_repeats needs bernoulli rows with n == 1; got n={r.n} "
                f"for item {r.item_id!r}"
            )
        if r.y not in (0, 1):
            raise ValueError(
                f"collapse_repeats needs binary y in {{0, 1}}; got y={r.y} "
                f"for item {r.item_id!r}"
            )
        groups[(r.arm, r.item_id, r.cluster, r.seed)].append(r)
    return [
        ItemRow(arm=arm, item_id=item, y=float(sum(r.y for r in grp)),
                n=len(grp), cluster=cluster, seed=seed)
        for (arm, item, cluster, seed), grp in groups.items()
    ]
