"""Paper evaluation for saved revision-capable streaming trajectories."""

from __future__ import annotations

import math
from pathlib import Path
import re
import statistics
from typing import Any

from .simultaneous_metrics import corpus_latency_metrics


CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff]")
WORD = re.compile(r"\w+(?:['’-]\w+)*", re.UNICODE)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def event_text(event: dict) -> str:
    for key in ("display_target_prefix", "target_prefix", "text", "translation"):
        if event.get(key) is not None:
            return str(event[key])
    return ""


def paced_events(row: dict) -> tuple[list[dict], float]:
    """Replay cumulative compute on a real-time PCM request clock."""
    result = []
    previous_compute = previous_completion = maximum_backlog = 0.0
    for event in row.get("events", []):
        ready = float(event.get("input_audio_end_time", event.get("audio_time", 0.0)))
        cumulative = float(event.get("display_wall_time_sec", event.get("wall_time_sec", 0.0)))
        incremental = max(0.0, cumulative - previous_compute)
        start = max(ready, previous_completion)
        completion = start + incremental
        maximum_backlog = max(maximum_backlog, max(0.0, previous_completion - ready))
        result.append(
            {
                "text": event_text(event).strip(),
                "audio_received_sec": ready,
                "emission_time": completion,
                "incremental_compute_sec": incremental,
            }
        )
        previous_compute, previous_completion = cumulative, completion
    return result, maximum_backlog


def latency_metrics(rows: list[dict]) -> dict:
    audio_positions, request_delays, rtfs, backlogs, paper_rows = [], [], [], [], []
    for row in rows:
        events, backlog = paced_events(row)
        visible = [event for event in events if event["text"]]
        if visible:
            audio_positions.append(float(visible[0]["audio_received_sec"]))
            request_delays.append(float(visible[0]["emission_time"]))
        rtfs.append(float(row.get("compute_rtf", 0.0)))
        backlogs.append(backlog)
        paper_rows.append(
            {
                "prediction": row.get("translation", row.get("hypothesis", "")),
                "reference": row.get("reference", ""),
                "audio_duration_sec": float(row["audio_duration_sec"]),
                "source_end_time_sec": float(row["audio_duration_sec"]),
                "events": [event for event in events if event["text"]],
            }
        )
    formal = corpus_latency_metrics(paper_rows)
    return {
        "records": len(rows),
        "FTL": {"mean": mean(audio_positions), "p90": percentile(audio_positions, 0.9)},
        "FRD": {"mean": mean(request_delays), "p90": percentile(request_delays, 0.9)},
        "LAAL_CU": {
            "mean": formal["mean_laal_sec_cu"],
            "p90": formal["p90_laal_sec_cu"],
        },
        "LAAL_CA_star": {
            "mean": formal["mean_laal_sec_ca_star"],
            "p90": formal["p90_laal_sec_ca_star"],
        },
        "RTF": {"mean": mean(rtfs), "p90": percentile(rtfs, 0.9)},
        "max_compute_backlog": {
            "mean": mean(backlogs), "p90": percentile(backlogs, 0.9)
        },
    }


def cjk_mixed_units(value: str) -> tuple[str, ...]:
    result, word = [], []

    def flush() -> None:
        token = "".join(word).strip("'’-")
        if token:
            result.append(token.casefold())
        word.clear()

    for character in value:
        if CJK.fullmatch(character):
            flush()
            result.append(character.casefold())
        elif character.isalnum():
            word.append(character)
        elif character in "'’-" and word:
            word.append(character)
        else:
            flush()
    flush()
    return tuple(result)


def units(text: Any, language: str) -> tuple[str, ...]:
    value = str(text)
    if language in {"zh", "ja"}:
        return cjk_mixed_units(value)
    return tuple(token.casefold() for token in WORD.findall(value))


def lcp(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    end = 0
    for a, b in zip(left, right):
        if a != b:
            break
        end += 1
    return left[:end]


def revision_record(row: dict, language: str) -> dict:
    events = list(row.get("events") or [])
    times = [float(event.get("audio_time", event.get("input_audio_end_time", 0.0))) for event in events]
    displays = [" ".join(event_text(event).casefold().replace("’", "'").split()) for event in events]
    states = [units(event_text(event), language) for event in events]
    if not events:
        return {"normalized_erasure": 0.0, "age_weighted_erasure": 0.0,
                "first_stable_unit": None, "mean_finalization": None,
                "revision_events": 0}
    final = states[-1]
    erased = age_weighted = 0.0
    revision_events = 0
    births: list[float] = []
    previous: tuple[str, ...] = ()
    previous_display = ""
    for time_value, display, state in zip(times, displays, states):
        kept = len(lcp(previous, state))
        true_revision = bool(previous_display and not display.startswith(previous_display))
        removed = len(previous) - kept if true_revision else 0
        if removed:
            revision_events += 1
            erased += removed
            age_weighted += sum(time_value - birth for birth in births[kept:])
        births = births[:kept] + [time_value] * (len(state) - kept)
        previous, previous_display = state, display

    survivors: list[tuple[str, ...]] = [()] * len(states)
    survivor = None
    for index in range(len(states) - 1, -1, -1):
        survivor = states[index] if survivor is None else lcp(states[index], survivor)
        survivors[index] = survivor
    finalization = [
        next(time_value for time_value, survivor in zip(times, survivors)
             if len(survivor) > position)
        for position in range(len(final))
    ]
    denominator = max(1, len(final))
    return {
        "normalized_erasure": erased / denominator,
        "age_weighted_erasure": age_weighted / denominator,
        "first_stable_unit": finalization[0] if finalization else times[-1],
        "mean_finalization": statistics.fmean(finalization) if finalization else times[-1],
        "revision_events": revision_events,
    }


def revision_metrics(rows: list[dict], language: str) -> dict:
    records = [revision_record(row, language) for row in rows]
    answer = {"records": len(records)}
    for key in ("normalized_erasure", "age_weighted_erasure", "first_stable_unit", "mean_finalization"):
        values = [float(row[key]) for row in records if row[key] is not None]
        answer[key] = {"mean": mean(values), "p90": percentile(values, 0.9)}
    answer["revision_record_rate"] = mean([row["revision_events"] > 0 for row in records])
    answer["revision_events"] = sum(row["revision_events"] for row in records)
    return answer


def quality_metrics(rows: list[dict], language: str, skip_comet: bool = False,
                    comet_batch_size: int = 16) -> dict:
    from sacrebleu.metrics import BLEU, CHRF

    hypotheses = [str(row.get("translation", "")) for row in rows]
    references = [str(row.get("reference", "")) for row in rows]
    sources = [str(row.get("source_text", "")) for row in rows]
    tokenizer = {"zh": "zh", "ja": "ja-mecab"}.get(language, "13a")
    result = {
        "records": len(rows),
        "output_coverage": mean([bool(text.strip()) for text in hypotheses]),
        "BLEU": BLEU(tokenize=tokenizer).corpus_score(hypotheses, [references]).score,
        "BLEU_tokenizer": tokenizer,
        "chrF++": CHRF(word_order=2).corpus_score(hypotheses, [references]).score,
        "COMET": None,
    }
    if not skip_comet:
        import torch
        from comet import load_from_checkpoint
        from huggingface_hub import snapshot_download

        from pdo_s2tt.revisions import COMET_REVISION

        snapshot = Path(snapshot_download(
            "Unbabel/wmt22-comet-da", revision=COMET_REVISION,
        ))
        model = load_from_checkpoint(str(snapshot / "checkpoints" / "model.ckpt"))
        prediction = model.predict(
            [{"src": src, "mt": mt, "ref": ref}
             for src, mt, ref in zip(sources, hypotheses, references)],
            batch_size=comet_batch_size,
            gpus=1 if torch.cuda.is_available() else 0,
        )
        result["COMET"] = float(prediction.system_score) * 100.0
    return result
