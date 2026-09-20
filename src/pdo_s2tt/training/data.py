"""Validation for the frozen multilingual PDO training inventory."""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path


LANGUAGES = frozenset({"zh", "de", "es", "ja", "fr"})
FULL_TRAIN_UNITS = 13_000
SOURCE_RECORDINGS = 2_600


def released_targets(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_released_manifest(rows: list[dict], targets_path: Path) -> dict:
    """Require the exact released IDs, order, directions, targets, and grouping."""
    if len(rows) != FULL_TRAIN_UNITS:
        raise ValueError(f"the full recipe requires {FULL_TRAIN_UNITS} direction examples")
    expected = released_targets(targets_path)
    if len(expected) != FULL_TRAIN_UNITS:
        raise ValueError("the released training target inventory is incomplete")

    fields = ("id", "target_lang", "target_text")
    for index, (row, target) in enumerate(zip(rows, expected)):
        for field in fields:
            if row.get(field) != target.get(field):
                raise ValueError(
                    f"training manifest differs from the released inventory at "
                    f"row {index}, field {field}"
                )
        if not str(row.get("source_text", "")).strip():
            raise ValueError(f"empty source text at training row {index}")
        if float(row.get("audio_duration_sec", 0.0)) <= 0:
            raise ValueError(f"invalid audio duration at training row {index}")
        if not str(row.get("audio", "")).strip():
            raise ValueError(f"missing audio path at training row {index}")
        if Path(str(row["audio"])).name != target.get("filename"):
            raise ValueError(
                f"training audio differs from the released inventory at row {index}"
            )

    counts = Counter(row["target_lang"] for row in rows)
    wanted = {language: SOURCE_RECORDINGS for language in LANGUAGES}
    if counts != wanted:
        raise ValueError(f"the five-language manifest is not balanced: {dict(counts)}")
    if len({row["id"] for row in rows}) != FULL_TRAIN_UNITS:
        raise ValueError("training manifest IDs are not unique")

    groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for row, target in zip(rows, expected):
        groups[str(row["audio"])].append((row, target))
    if len(groups) != SOURCE_RECORDINGS:
        raise ValueError(
            f"expected {SOURCE_RECORDINGS} shared source recordings, received {len(groups)}"
        )
    for audio, group in groups.items():
        if {row["target_lang"] for row, _ in group} != LANGUAGES:
            raise ValueError(f"recording does not cover all five directions: {audio}")
        if len({target["sentence_id"] for _, target in group}) != 1:
            raise ValueError(f"sentence identity differs across directions: {audio}")
        if len({row["source_text"] for row, _ in group}) != 1:
            raise ValueError(f"source text differs across directions: {audio}")
        if len({float(row["audio_duration_sec"]) for row, _ in group}) != 1:
            raise ValueError(f"audio duration differs across directions: {audio}")
    return {
        "direction_units": len(rows),
        "source_recordings": len(groups),
        "directions": dict(sorted(counts.items())),
    }
