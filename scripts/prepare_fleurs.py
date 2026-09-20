#!/usr/bin/env python3
"""Download the official English FLEURS TEST split and build one direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile

import soundfile as sf

from huggingface_hub import hf_hub_download

from pdo_s2tt.revisions import FLEURS_REVISION


LANGUAGES = {"zh", "de", "es", "ja", "fr"}
EXPECTED_RECORDS = 647
EXPECTED_SENTENCES = 350


def read_tsv(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 7:
            raise ValueError(f"{path}:{line_number}: expected 7 tab-separated fields")
        sentence_id, filename, raw, normalized, _characters, samples, gender = fields
        rows.append(
            {
                "sentence_id": sentence_id,
                "filename": filename,
                "source_text": raw,
                "normalized_source_text": normalized,
                "declared_samples": int(samples),
                "gender": gender,
            }
        )
    return rows


def extract_audio(archive: Path, audio_root: Path, filenames: set[str]) -> None:
    audio_root.mkdir(parents=True, exist_ok=True)
    missing = {name for name in filenames if not (audio_root / name).is_file()}
    if not missing:
        return
    with tarfile.open(archive, "r:gz") as stream:
        members = {member.name: member for member in stream.getmembers() if member.isfile()}
        for filename in sorted(missing):
            name = f"test/{filename}"
            member = members.get(name)
            if member is None or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise ValueError(f"missing or unsafe archive member: {name}")
            source = stream.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {name}")
            (audio_root / filename).write_bytes(source.read())


def wav_duration(path: Path, declared_samples: int) -> float:
    info = sf.info(str(path))
    if info.channels != 1 or info.samplerate != 16000:
        raise ValueError(f"unexpected WAV format: {path}")
    samples = int(info.frames)
    if abs(samples - declared_samples) > 1:
        raise ValueError(f"sample-count mismatch: {path}")
    return samples / 16000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="zh", choices=sorted(LANGUAGES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("data/.cache"))
    parser.add_argument("--tsv", type=Path, help="reuse an existing official en_us/test.tsv")
    parser.add_argument("--archive", type=Path, help="reuse an existing official en_us test.tar.gz")
    args = parser.parse_args()
    output = args.output or Path(f"data/fleurs/en-{args.target}")
    output.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    references = json.loads(
        (repo_root / "references" / "fleurs_test.json").read_text(encoding="utf-8")
    )[args.target]
    if len(references) != EXPECTED_SENTENCES:
        raise ValueError("the bundled reference set is incomplete")

    tsv = args.tsv or Path(
        hf_hub_download(
            "google/fleurs", "data/en_us/test.tsv", repo_type="dataset",
            cache_dir=args.cache, revision=FLEURS_REVISION,
        )
    )
    archive = args.archive or Path(
        hf_hub_download(
            "google/fleurs", "data/en_us/audio/test.tar.gz", repo_type="dataset",
            cache_dir=args.cache, revision=FLEURS_REVISION,
        )
    )
    source_rows = read_tsv(tsv)
    if len(source_rows) != EXPECTED_RECORDS:
        raise ValueError(f"expected {EXPECTED_RECORDS} English TEST recordings")
    if len({row['sentence_id'] for row in source_rows}) != EXPECTED_SENTENCES:
        raise ValueError(f"expected {EXPECTED_SENTENCES} unique TEST sentences")
    missing_references = sorted({row["sentence_id"] for row in source_rows} - references.keys())
    if missing_references:
        raise ValueError(f"missing references for sentence IDs: {missing_references[:5]}")

    # All five directions use the same English recordings. Keep one shared copy
    # next to the direction directories instead of extracting it five times.
    audio_root = output.parent / "audio" / "test"
    extract_audio(archive, audio_root, {row["filename"] for row in source_rows})
    manifest = output / "test.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as stream:
        for row in source_rows:
            audio = audio_root / row["filename"]
            recording_id = Path(row["filename"]).stem
            item = {
                "id": f"fleurs_eng_{row['sentence_id']}__{recording_id}",
                "sentence_id": row["sentence_id"],
                "split": "test",
                "source_lang": "en",
                "target_lang": args.target,
                "audio": (Path("..") / "audio" / "test" / row["filename"]).as_posix(),
                "audio_duration_sec": wav_duration(audio, row["declared_samples"]),
                "source_text": row["source_text"],
                "target_text": references[row["sentence_id"]],
                "gender": row["gender"],
            }
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Prepared {EXPECTED_RECORDS} En→{args.target} examples: {manifest}")


if __name__ == "__main__":
    main()
