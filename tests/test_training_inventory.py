import gzip
import json
from collections import Counter
from pathlib import Path


TARGETS = Path(__file__).resolve().parents[1] / "references" / "fleurs_train_targets.jsonl.gz"


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
