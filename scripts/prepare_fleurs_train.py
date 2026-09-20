#!/usr/bin/env python3
"""Download FLEURS TRAIN and build the five-direction PDO training manifest."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import tarfile

import soundfile as sf
from huggingface_hub import hf_hub_download

from pdo_s2tt.revisions import FLEURS_REVISION


LANGUAGES = {"zh", "de", "es", "ja", "fr"}
EXPECTED_SOURCE_RECORDINGS = 2602
EXPECTED_DIRECTION_UNITS = 13_000


def read_tsv(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 7:
            raise ValueError(f"{path}:{line_number}: expected seven tab-separated fields")
        sentence_id, filename, raw, normalized, _characters, samples, gender = fields
        rows.append({
            "sentence_id": sentence_id,
            "filename": filename,
            "text": raw,
            "normalized_text": normalized,
            "samples": int(samples),
            "gender": gender,
        })
    return rows


def extract_audio(archive: Path, output: Path, filenames: set[str]) -> None:
    destination = output / "audio" / "train"
    destination.mkdir(parents=True, exist_ok=True)
    missing = {name for name in filenames if not (destination / name).is_file()}
    if not missing:
        return
    with tarfile.open(archive, "r:gz") as stream:
        members = {member.name: member for member in stream.getmembers() if member.isfile()}
        for filename in sorted(missing):
            member = members.get(f"train/{filename}")
            if member is None or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise ValueError(f"missing or unsafe archive member: train/{filename}")
            source = stream.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read train/{filename}")
            (destination / filename).write_bytes(source.read())


def duration(path: Path, declared_samples: int) -> float:
    info = sf.info(str(path))
    if info.channels != 1 or info.samplerate != 16000:
        raise ValueError(f"unexpected WAV format: {path}")
    if abs(info.frames - declared_samples) > 1:
        raise ValueError(f"sample-count mismatch: {path}")
    return info.frames / 16000.0


def download(repo_file: str, cache: Path) -> Path:
    return Path(hf_hub_download(
        "google/fleurs", repo_file, repo_type="dataset", cache_dir=cache,
        revision=FLEURS_REVISION,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/fleurs/train"))
    parser.add_argument("--cache", type=Path, default=Path("data/.cache"))
    parser.add_argument("--tsv", type=Path, help="reuse an existing official en_us/train.tsv")
    parser.add_argument(
        "--archive",
        type=Path,
        help="reuse an existing official en_us train.tar.gz",
    )
    parser.add_argument(
        "--targets", type=Path,
        default=Path(__file__).resolve().parents[1] / "references" / "fleurs_train_targets.jsonl.gz",
        help="released five-direction targets and exact training order",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    source_tsv = args.tsv or download("data/en_us/train.tsv", args.cache)
    archive = args.archive or download("data/en_us/audio/train.tar.gz", args.cache)
    source = read_tsv(source_tsv)
    if len(source) != EXPECTED_SOURCE_RECORDINGS:
        raise ValueError(f"expected {EXPECTED_SOURCE_RECORDINGS} English TRAIN recordings")
    source_by_filename = {row["filename"]: row for row in source}
    with gzip.open(args.targets, "rt", encoding="utf-8") as stream:
        targets = [json.loads(line) for line in stream if line.strip()]
    if len(targets) != EXPECTED_DIRECTION_UNITS or len({row["id"] for row in targets}) != len(targets):
        raise ValueError("the released FLEURS training target inventory is incomplete")
    counts = {language: sum(row["target_lang"] == language for row in targets) for language in LANGUAGES}
    if set(counts.values()) != {2600}:
        raise ValueError(f"the released directions are not balanced: {counts}")
    selected_filenames = {row["filename"] for row in targets}
    if selected_filenames - source_by_filename.keys():
        raise ValueError("released training targets do not match official English audio")
    extract_audio(archive, args.output, selected_filenames)

    manifest_rows = []
    for target in targets:
        source_row = source_by_filename[target["filename"]]
        if source_row["sentence_id"] != target["sentence_id"]:
            raise ValueError(f"sentence identity mismatch for {target['filename']}")
        audio = args.output / "audio" / "train" / source_row["filename"]
        manifest_rows.append({
            "id": target["id"],
            "sentence_id": source_row["sentence_id"],
            "split": "train",
            "source_lang": "en",
            "target_lang": target["target_lang"],
            "audio": f"audio/train/{source_row['filename']}",
            "audio_duration_sec": duration(audio, source_row["samples"]),
            "source_text": source_row["text"],
            "target_text": target["target_text"],
            "gender": source_row["gender"],
        })

    manifest = args.output / "train.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as stream:
        for row in manifest_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Prepared {len(manifest_rows)} direction examples: {manifest}")


if __name__ == "__main__":
    main()
