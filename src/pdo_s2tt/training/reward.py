"""Persistent Delivery Optimization reward used in the paper.

At event ``t``, the process-quality term is computed on the longest prefix of
the current draft that survives unchanged in every later draft.  The terminal
term is sentence BLEU on the current draft.  The event action receives the
full future return ``G_t = Phi_T - Phi_{t-1}``; returns are standardized only
across the four sampled trajectories of the same utterance and event.
"""

from __future__ import annotations

import math
import re
import statistics

from sacrebleu.metrics import BLEU


LANGUAGES = {"zh", "de", "es", "ja", "fr"}
TERMINAL_HOLD_SECONDS = 2.0
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]")
_WORD = re.compile(r"\w+(?:['’-]\w+)*", re.UNICODE)
_BLEU = BLEU(effective_order=True, tokenize="none")


def units(text: object, language: str) -> tuple[str, ...]:
    """Return the language-aware content units used by the reward."""
    value = str(text)
    if language not in {"zh", "ja"}:
        return tuple(token.casefold() for token in _WORD.findall(value))
    result: list[str] = []
    word: list[str] = []

    def flush() -> None:
        token = "".join(word).strip("'’-")
        if token:
            result.append(token.casefold())
        word.clear()

    for character in value:
        if _CJK.fullmatch(character):
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


def longest_common_prefix(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    end = 0
    for first, second in zip(left, right):
        if first != second:
            break
        end += 1
    return left[:end]


def _lcs_length(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, target in enumerate(right, 1):
            current.append(
                previous[index - 1] + 1
                if token == target else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def trajectory_ledger(
    drafts: list[str], times: list[float], reference: str, language: str,
) -> dict:
    """Compute potentials, causal rewards, and full returns for one trajectory."""
    if language not in LANGUAGES:
        raise ValueError(f"unsupported target language: {language}")
    tokenized = [units(draft, language) for draft in drafts]
    target = units(reference, language)
    if not tokenized or len(tokenized) != len(times) or not target:
        raise ValueError("a complete trajectory and non-empty reference are required")
    if any(second <= first for first, second in zip(times, times[1:])):
        raise ValueError("event times must be strictly increasing")

    interval = [second - first for first, second in zip(times, times[1:])]
    survivors: list[tuple[str, ...]] = []
    potentials: list[float] = []
    rewards: list[float] = []
    previous = 0.0
    for event, current in enumerate(tokenized):
        survivors = [longest_common_prefix(old, current) for old in survivors] + [current]
        area = sum(
            interval[index] * _lcs_length(survivors[index], target) / len(target)
            for index in range(event)
        )
        terminal = 0.0 if not current else (
            _BLEU.sentence_score(" ".join(current), [" ".join(target)]).score / 100.0
        )
        potential = area + TERMINAL_HOLD_SECONDS * terminal
        rewards.append(potential - previous)
        potentials.append(potential)
        previous = potential

    returns = [
        potentials[-1] - (potentials[event - 1] if event else 0.0)
        for event in range(len(potentials))
    ]
    if not math.isclose(sum(rewards), potentials[-1], abs_tol=1e-10):
        raise RuntimeError("PDO rewards failed the telescoping identity")
    return {
        "potentials": potentials,
        "causal_rewards": rewards,
        "returns": returns,
        "objective": potentials[-1],
    }


def _standardize(values: list[float]) -> list[float]:
    mean = statistics.mean(values)
    variance = statistics.mean((value - mean) ** 2 for value in values)
    if variance <= 1e-16:
        return [0.0] * len(values)
    scale = math.sqrt(variance)
    return [(value - mean) / scale for value in values]


def persistent_delivery_returns(
    trajectories: list[dict], reference: str, language: str,
) -> list[list[float]]:
    """Return event-wise group-relative PDO advantages for a G=4 rollout."""
    if len(trajectories) != 4:
        raise ValueError("PDO uses exactly four sampled trajectories per utterance")
    ledgers = [
        trajectory_ledger(
            trajectory["drafts"], trajectory["times"], reference, language,
        )
        for trajectory in trajectories
    ]
    count = len(ledgers[0]["returns"])
    if not count or any(len(ledger["returns"]) != count for ledger in ledgers):
        raise ValueError("the four trajectories must share one non-empty event grid")
    advantages = [[0.0] * count for _ in range(4)]
    for event in range(count):
        normalized = _standardize([ledger["returns"][event] for ledger in ledgers])
        for lane in range(4):
            advantages[lane][event] = normalized[lane]
    return advantages
