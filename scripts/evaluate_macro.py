#!/usr/bin/env python3
"""Aggregate the five released FLEURS directions into a macro result."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from pdo_s2tt.evaluation import latency_metrics


TARGETS = ("zh", "de", "es", "ja", "fr")
PAPER_MACRO = {
    "BLEU": 31.5803,
    "COMET": 85.3733,
    "chrF++": 44.6731,
    "FTL": 2.0000,
    "FRD": 2.4194,
    "FRD_P90": 2.5950,
    "LAAL_CU": 3.0493,
    "LAAL_CU_P90": 5.6600,
    "RTF": 0.5366,
    "NormErase": 0.9364,
    "AgeErase": 1.9816,
    "FirstStable": 2.9775,
    "MeanFinalization": 7.3635,
}


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; finish all five directions before computing the macro"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; macro P90 requires the saved trajectories"
        )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def average(values):
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if len(present) == len(values) else None


def nested_macro(rows: list[dict], section: str) -> dict:
    keys = rows[0][section].keys()
    result = {}
    for key in keys:
        values = [row[section][key] for row in rows]
        if isinstance(values[0], dict):
            result[key] = {
                metric: average([value.get(metric) for value in values])
                for metric in values[0]
                if metric != "records"
            }
        elif key == "records":
            result[key] = sum(int(value) for value in values)
        elif isinstance(values[0], (int, float)) or values[0] is None:
            result[key] = average(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/fleurs"))
    parser.add_argument("--output", type=Path, default=Path("results/fleurs/macro.json"))
    args = parser.parse_args()

    rows = [load(args.results_dir / f"en-{target}" / "metrics.json") for target in TARGETS]
    expected_directions = [f"En→{target}" for target in TARGETS]
    if [row.get("direction") for row in rows] != expected_directions:
        raise ValueError("metric files do not contain the five expected directions")

    quality = {
        "records": sum(int(row["quality"]["records"]) for row in rows),
        "output_coverage": average([row["quality"]["output_coverage"] for row in rows]),
        "BLEU": average([row["quality"]["BLEU"] for row in rows]),
        "COMET": average([row["quality"]["COMET"] for row in rows]),
        "chrF++": average([row["quality"]["chrF++"] for row in rows]),
    }
    predictions = [
        item
        for target in TARGETS
        for item in read_jsonl(
            args.results_dir / f"en-{target}" / "predictions.jsonl"
        )
    ]
    if len(predictions) != 647 * len(TARGETS):
        raise ValueError(
            f"expected {647 * len(TARGETS)} predictions, received {len(predictions)}"
        )
    result = {
        "dataset": "FLEURS TEST",
        "directions": expected_directions,
        "model": "PDO",
        "quality": quality,
        "latency": latency_metrics(predictions),
        "revision": nested_macro(rows, "revision"),
        "paper_result": PAPER_MACRO,
        "note": (
            "Quality and revision means are direction-level macro averages; "
            "latency percentiles are computed over all 3,235 trajectories."
        ),
    }
    deterministic = {
        "BLEU": quality["BLEU"],
        "chrF++": quality["chrF++"],
        "FTL": result["latency"]["FTL"]["mean"],
        "LAAL_CU": result["latency"]["LAAL_CU"]["mean"],
        "LAAL_CU_P90": result["latency"]["LAAL_CU"]["p90"],
        "NormErase": result["revision"]["normalized_erasure"]["mean"],
        "AgeErase": result["revision"]["age_weighted_erasure"]["mean"],
        "FirstStable": result["revision"]["first_stable_unit"]["mean"],
        "MeanFinalization": result["revision"]["mean_finalization"]["mean"],
    }
    observed = dict(deterministic)
    complete = quality["COMET"] is not None
    if complete:
        observed["COMET"] = quality["COMET"]
    checks = {
        key: {
            "observed": value,
            "expected": PAPER_MACRO[key],
            "tolerance": 0.02 if key == "COMET" else 0.01,
            "passed": abs(value - PAPER_MACRO[key])
            <= (0.02 if key == "COMET" else 0.01),
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
        "excluded_as_hardware_dependent": ["FRD", "FRD_P90", "RTF"],
        "note": None if complete else "COMET was skipped; the full check is incomplete.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    check = result["reproduction_check"]
    if check["deterministic_passed"] is False or check["passed"] is False:
        raise SystemExit("the macro metrics do not match the released result")


if __name__ == "__main__":
    main()
