#!/usr/bin/env python3
"""Evaluate saved PDO trajectories with the paper's FLEURS metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pdo_s2tt.evaluation import latency_metrics, quality_metrics, revision_metrics
from pdo_s2tt.languages import LANGUAGE_NAMES


EXPECTED = {
    "zh": {"BLEU": 38.1569, "COMET": 86.3143, "chrF++": 26.9230,
           "FTL": 2.0000, "FRD": 2.3252, "LAAL_CU": 3.5316,
           "LAAL_CU_P90": 6.6063, "RTF": 0.3859, "NormErase": 1.1128,
           "AgeErase": 2.3695, "FirstStable": 3.3371, "MeanFinalization": 7.7436},
    "de": {"BLEU": 29.5690, "COMET": 84.2197, "chrF++": 57.0312,
           "FTL": 2.0000, "FRD": 2.4192, "LAAL_CU": 3.0775,
           "LAAL_CU_P90": 5.6506, "RTF": 0.5735, "NormErase": 0.9229,
           "AgeErase": 1.9656, "FirstStable": 2.9568, "MeanFinalization": 7.3546},
    "es": {"BLEU": 23.3539, "COMET": 83.5737, "chrF++": 50.8830,
           "FTL": 2.0000, "FRD": 2.4354, "LAAL_CU": 2.5074,
           "LAAL_CU_P90": 4.4381, "RTF": 0.5668, "NormErase": 0.6406,
           "AgeErase": 1.3969, "FirstStable": 2.7425, "MeanFinalization": 6.8025},
    "ja": {"BLEU": 28.6619, "COMET": 88.5012, "chrF++": 26.4720,
           "FTL": 2.0000, "FRD": 2.4726, "LAAL_CU": 3.6694,
           "LAAL_CU_P90": 6.2283, "RTF": 0.5671, "NormErase": 1.3451,
           "AgeErase": 2.7091, "FirstStable": 3.1364, "MeanFinalization": 8.0263},
    "fr": {"BLEU": 38.1596, "COMET": 84.2575, "chrF++": 62.0564,
           "FTL": 2.0000, "FRD": 2.4448, "LAAL_CU": 2.4608,
           "LAAL_CU_P90": 4.9269, "RTF": 0.5898, "NormErase": 0.6604,
           "AgeErase": 1.4667, "FirstStable": 2.7148, "MeanFinalization": 6.8904},
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="zh", choices=sorted(LANGUAGE_NAMES))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-comet", action="store_true")
    parser.add_argument("--comet-batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="score only the first N manifest examples")
    args = parser.parse_args()
    manifest = args.manifest or Path(f"data/fleurs/en-{args.target}/test.jsonl")
    predictions = args.predictions or Path(f"results/fleurs/en-{args.target}/predictions.jsonl")
    output = args.output or Path(f"results/fleurs/en-{args.target}/metrics.json")

    gold = read_jsonl(manifest)
    if args.limit is not None:
        gold = gold[: args.limit]
    rows = read_jsonl(predictions)
    if args.limit is not None:
        selected = {row["id"] for row in gold}
        rows = [row for row in rows if row.get("id") in selected]
    if len(rows) != len(gold):
        raise ValueError(f"expected {len(gold)} predictions, received {len(rows)}")
    gold_by_id = {row["id"]: row for row in gold}
    if len(gold_by_id) != len(gold) or {row["id"] for row in rows} != set(gold_by_id):
        raise ValueError("prediction IDs do not match the manifest")
    for row in rows:
        source = gold_by_id[row["id"]]
        row["source_text"] = source["source_text"]
        row["reference"] = source["target_text"]
        row["target_lang"] = args.target
        row["audio_duration_sec"] = source["audio_duration_sec"]

    quality = quality_metrics(rows, args.target, args.skip_comet, args.comet_batch_size)
    latency = latency_metrics(rows)
    revision = revision_metrics(rows, args.target)
    result = {
        "dataset": "FLEURS TEST",
        "direction": f"En→{args.target}",
        "model": "PDO",
        "quality": quality,
        "latency": latency,
        "revision": revision,
        "paper_result": EXPECTED[args.target],
        "note": "FRD and RTF are hardware-dependent; the paper values use one RTX 4090.",
    }
    if len(rows) == 647:
        deterministic = {
            "BLEU": quality["BLEU"], "chrF++": quality["chrF++"],
            "FTL": latency["FTL"]["mean"],
            "LAAL_CU": latency["LAAL_CU"]["mean"],
            "LAAL_CU_P90": latency["LAAL_CU"]["p90"],
            "NormErase": revision["normalized_erasure"]["mean"],
            "AgeErase": revision["age_weighted_erasure"]["mean"],
            "FirstStable": revision["first_stable_unit"]["mean"],
            "MeanFinalization": revision["mean_finalization"]["mean"],
        }
        observed = dict(deterministic)
        complete = quality["COMET"] is not None
        if complete:
            observed["COMET"] = quality["COMET"]
        tolerance = {key: (0.02 if key == "COMET" else 0.01) for key in observed}
        checks = {
            key: {
                "observed": value,
                "expected": EXPECTED[args.target][key],
                "tolerance": tolerance[key],
                "passed": abs(value - EXPECTED[args.target][key]) <= tolerance[key],
            }
            for key, value in observed.items()
        }
        result["reproduction_check"] = {
            "complete": complete,
            "passed": all(item["passed"] for item in checks.values()) if complete else None,
            "deterministic_passed": all(
                checks[key]["passed"] for key in deterministic
            ),
            "checks": checks,
            "excluded_as_hardware_dependent": ["FRD", "RTF"],
            "note": None if complete else "COMET was skipped; the full check is incomplete.",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    check = result.get("reproduction_check", {})
    if check.get("deterministic_passed") is False or check.get("passed") is False:
        raise SystemExit("saved metrics do not match the released FLEURS TEST result")


if __name__ == "__main__":
    main()
