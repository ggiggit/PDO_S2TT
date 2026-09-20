import gzip
import json
from collections import Counter
from pathlib import Path

from pdo_s2tt import __version__
from pdo_s2tt.revisions import COMET_REVISION, FLEURS_REVISION, QWEN3_ASR_REVISION
from pdo_s2tt.training.data import released_targets, validate_released_manifest


TARGETS = Path(__file__).resolve().parents[1] / "references" / "fleurs_train_targets.jsonl.gz"


def test_public_package_version():
    assert __version__ == "0.2.0"


def test_upstream_revisions_are_immutable_commit_ids():
    for revision in (FLEURS_REVISION, QWEN3_ASR_REVISION, COMET_REVISION):
        assert len(revision) == 40
        int(revision, 16)


def test_released_training_inventory_is_complete_and_balanced():
    with gzip.open(TARGETS, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]

    assert len(rows) == 13_000
    assert len({row["id"] for row in rows}) == 13_000
    assert Counter(row["target_lang"] for row in rows) == {
        "zh": 2_600,
        "de": 2_600,
        "es": 2_600,
        "ja": 2_600,
        "fr": 2_600,
    }
    assert len({row["filename"] for row in rows}) == 2_600
    assert all(row["id"].startswith(f'{row["target_lang"]}::') for row in rows)
    assert all(row["filename"].endswith(".wav") for row in rows)
    assert all(row["target_text"].strip() for row in rows)


def full_manifest_rows():
    rows = []
    for target in released_targets(TARGETS):
        rows.append(
            {
                **target,
                "audio": f"/frozen/{target['filename']}",
                "audio_duration_sec": 1.0,
                "source_text": f"source {target['sentence_id']}",
            }
        )
    return rows


def test_full_training_manifest_matches_frozen_inventory_and_shared_audio():
    receipt = validate_released_manifest(full_manifest_rows(), TARGETS)
    assert receipt == {
        "direction_units": 13_000,
        "source_recordings": 2_600,
        "directions": {"de": 2_600, "es": 2_600, "fr": 2_600, "ja": 2_600, "zh": 2_600},
    }


def test_full_training_manifest_rejects_changed_target():
    rows = full_manifest_rows()
    rows[17]["target_text"] += " changed"
    try:
        validate_released_manifest(rows, TARGETS)
    except ValueError as error:
        assert "row 17, field target_text" in str(error)
    else:
        raise AssertionError("changed frozen target was accepted")


def test_full_training_manifest_rejects_changed_audio_identity():
    rows = full_manifest_rows()
    rows[17]["audio"] = "/frozen/not-the-released-recording.wav"
    try:
        validate_released_manifest(rows, TARGETS)
    except ValueError as error:
        assert "training audio differs" in str(error)
    else:
        raise AssertionError("changed frozen audio identity was accepted")
