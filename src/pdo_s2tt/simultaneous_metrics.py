#!/usr/bin/env python3
"""Paper-standard short-form SimulST latency metrics from streaming updates."""

from __future__ import annotations

import math
import re
from statistics import mean
from typing import Any

import numpy as np


WORD_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff]"
    r"|[A-Za-z\u00c0-\u024f\u1e00-\u1eff0-9]+"
    r"(?:['’-][A-Za-z\u00c0-\u024f\u1e00-\u1eff0-9]+)*"
)


def words(text: str) -> list[str]:
    """Tokenize Latin text by word and Han/kana text by character."""
    return [value.casefold() for value in WORD_PATTERN.findall(str(text))]


def stable_emission_times(
    final_text: str, updates: list[dict[str, Any]], timestamp_field: str
) -> list[float | None]:
    """Return when each final word prefix stops changing."""
    final_words = words(final_text)
    update_words = [words(str(item["text"])) for item in updates]
    answer: list[float | None] = []
    for end in range(1, len(final_words) + 1):
        prefix = final_words[:end]
        stable = None
        for index, item in enumerate(updates):
            if all(values[:end] == prefix for values in update_words[index:]):
                stable = float(item[timestamp_field])
                break
        answer.append(stable)
    return answer


def _event_updates(
    row: dict[str, Any], computation_aware: bool
) -> list[dict[str, Any]]:
    events = row.get("events")
    if not isinstance(events, list) or not events:
        return []
    timestamp = "emission_time" if computation_aware else "audio_received_sec"

    # Prefix-retranslation and incremental-decoder logs use a common text field.
    text_events = [
        item
        for item in events
        if isinstance(item, dict)
        and str(item.get("text", "")).strip()
        and item.get(timestamp) is not None
    ]
    if text_events:
        final_words = words(str(row.get("prediction", "")))
        full_hypotheses = words(str(text_events[-1]["text"])) == final_words
        updates: list[dict[str, Any]] = []
        accumulated = ""
        for item in text_events:
            text = str(item["text"]).strip()
            if full_hypotheses:
                hypothesis = text
            else:
                accumulated = f"{accumulated} {text}".strip()
                hypothesis = accumulated
            updates.append({"text": hypothesis, "time": float(item[timestamp])})
        return updates

    # The native dual-state runtime logs draft replacement and atomic commits.
    native_time = "simulated_request_time" if computation_aware else "audio_time"
    committed: list[str] = []
    updates = []
    for item in events:
        if not isinstance(item, dict) or item.get(native_time) is None:
            continue
        event_type = str(item.get("event_type", ""))
        draft = str(item.get("active_draft", "")).strip()
        committed_unit = item.get("committed_unit")
        if event_type == "unit_commit" and isinstance(committed_unit, dict):
            target = str(committed_unit.get("target", "")).strip()
            if target:
                committed.append(target)
            hypothesis = " ".join(committed).strip()
        elif draft:
            hypothesis = " ".join([*committed, draft]).strip()
        elif event_type == "finish" and committed:
            hypothesis = " ".join(committed).strip()
        else:
            continue
        updates.append({"text": hypothesis, "time": float(item[native_time])})
    return updates


def _al(delays: list[float], source_length: float, target_length: int) -> float:
    rate = source_length / max(1, target_length)
    tau = len(delays)
    for index, delay in enumerate(delays):
        if delay >= source_length:
            tau = index + 1
            break
    return mean(delays[index] - index * rate for index in range(tau))


def _dal(delays: list[float], source_length: float, target_length: int) -> float:
    # SimulEval 1.1.4 deliberately uses the emitted hypothesis length for DAL,
    # even when AL/AP use the reference length.
    del target_length
    rate = source_length / max(1, len(delays))
    smoothed: list[float] = []
    for delay in delays:
        smoothed.append(
            delay if not smoothed else max(delay, smoothed[-1] + rate)
        )
    return mean(value - index * rate for index, value in enumerate(smoothed))


def _atd(
    consumed_delays: list[float],
    elapsed_delays: list[float] | None = None,
    source_token_duration: float = 0.3,
) -> float:
    """Match SimulEval 1.1.4 ATD for speech-to-text in seconds."""
    if elapsed_delays is not None and len(elapsed_delays) != len(consumed_delays):
        raise ValueError("ATD consumed/elapsed delay length mismatch")

    compute_times = [0.0] * len(consumed_delays)
    if elapsed_delays is not None:
        compute_elapsed = [
            elapsed - consumed
            for elapsed, consumed in zip(elapsed_delays, consumed_delays)
        ]
        compute_times = [
            value - (compute_elapsed[index - 1] if index else 0.0)
            for index, value in enumerate(compute_elapsed)
        ]

    unique_delays: list[float] = []
    for delay in consumed_delays:
        if delay not in unique_delays:
            unique_delays.append(delay)

    chunk_sizes: dict[str, list[int]] = {"src": [0], "tgt": [0]}
    token_to_chunk: dict[str, list[int]] = {"src": [0], "tgt": [0]}
    token_to_time: dict[str, list[float]] = {"src": [0.0], "tgt": [0.0]}

    previous = None
    for delay in consumed_delays:
        if delay != previous:
            chunk_sizes["tgt"].append(1)
        else:
            chunk_sizes["tgt"][-1] += 1
        previous = delay
    for chunk_id, chunk_size in enumerate(chunk_sizes["tgt"][1:], 1):
        token_to_chunk["tgt"].extend([chunk_id] * chunk_size)

    previous_delay = 0.0
    for chunk_id, delay in enumerate(unique_delays, 1):
        duration = delay - previous_delay
        previous_delay = delay
        whole_tokens = int(duration // source_token_duration)
        remainder = duration - whole_tokens * source_token_duration
        token_lengths = [source_token_duration] * whole_tokens
        if remainder > 1e-9:
            token_lengths.append(remainder)
        chunk_sizes["src"].append(len(token_lengths))
        for token_length in token_lengths:
            token_to_time["src"].append(token_to_time["src"][-1] + token_length)
            token_to_chunk["src"].append(chunk_id)

    for delay, compute_time in zip(consumed_delays, compute_times):
        target_start = max(delay, token_to_time["tgt"][-1])
        token_to_time["tgt"].append(target_start + compute_time)

    atd_delays: list[float] = []
    for target_index in range(1, len(token_to_chunk["tgt"])):
        chunk_id = token_to_chunk["tgt"][target_index]
        accumulated_source = sum(chunk_sizes["src"][:chunk_id])
        accumulated_target = sum(chunk_sizes["tgt"][:chunk_id])
        source_index = target_index - max(
            0, accumulated_target - accumulated_source
        )
        current_source_size = sum(chunk_sizes["src"][: chunk_id + 1])
        source_index = min(source_index, current_source_size)
        atd_delays.append(
            token_to_time["tgt"][target_index]
            - token_to_time["src"][source_index]
        )
    return mean(atd_delays)


def sentence_latency_metrics(
    row: dict[str, Any], computation_aware: bool
) -> dict[str, float] | None:
    prediction = str(row.get("prediction", ""))
    hypothesis_length = len(words(prediction))
    reference_length = len(words(str(row.get("reference", ""))))
    source_length = float(row.get("audio_duration_sec", 0.0))
    source_end = float(row.get("source_end_time_sec", source_length))
    if hypothesis_length == 0 or reference_length == 0 or source_length <= 0:
        return None
    updates = _event_updates(row, computation_aware)
    if not updates:
        return None
    emissions = stable_emission_times(prediction, updates, "time")
    if any(value is None for value in emissions):
        return None
    delays = [float(value) for value in emissions if value is not None]
    if any(not math.isfinite(value) or value < 0 for value in delays):
        return None
    consumed_updates = _event_updates(row, False)
    consumed_emissions = stable_emission_times(prediction, consumed_updates, "time")
    if any(value is None for value in consumed_emissions):
        return None
    consumed_delays = [
        float(value) for value in consumed_emissions if value is not None
    ]
    laal_length = max(hypothesis_length, reference_length)
    return {
        "al_sec": _al(delays, source_length, reference_length),
        "laal_sec": _al(delays, source_length, laal_length),
        "ap": sum(delays) / (source_length * reference_length),
        "dal_sec": _dal(delays, source_length, reference_length),
        "atd_sec": _atd(
            consumed_delays,
            delays if computation_aware else None,
        ),
        "end_offset_sec": delays[-1] - source_end,
        "stable_output_words": float(hypothesis_length),
    }


def corpus_latency_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "protocol": "stable-final-word-simuleval-1.1.4-with-CA-star-v3",
        "computation_aware_protocol": (
            "CA* (Xu et al., NAACL Findings 2025): event timestamps are real "
            "backlog-buffered wall-clock times, not source time plus globally "
            "accumulated compute"
        ),
        "legacy_CA_formula_used": False,
        "timestamp_semantics": {
            "cu": "source audio consumed when each final word became stable",
            "ca": (
                "CA* wall-clock emission time including only uncovered compute "
                "backlog; retained under the historical ca key for compatibility"
            ),
            "end_offset": "last stable target word minus last source-word end when timing is available",
        },
    }
    for suffix, computation_aware in (("cu", False), ("ca", True)):
        samples = [
            value
            for row in rows
            if (value := sentence_latency_metrics(row, computation_aware)) is not None
        ]
        result[f"records_{suffix}"] = len(samples)
        for metric in (
            "al_sec",
            "laal_sec",
            "ap",
            "dal_sec",
            "atd_sec",
            "end_offset_sec",
        ):
            values = [float(item[metric]) for item in samples]
            result[f"mean_{metric}_{suffix}"] = mean(values) if values else None
            result[f"p90_{metric}_{suffix}"] = (
                float(np.percentile(values, 90)) if values else None
            )
    # Preserve all historical consumers while making the corrected CA* identity
    # explicit in new reports.
    for metric in (
        "al_sec",
        "laal_sec",
        "ap",
        "dal_sec",
        "atd_sec",
        "end_offset_sec",
    ):
        result[f"mean_{metric}_ca_star"] = result[f"mean_{metric}_ca"]
        result[f"p90_{metric}_ca_star"] = result[f"p90_{metric}_ca"]
    result["records_ca_star"] = result["records_ca"]
    return result
