"""Qwen chat-turn serialization for streaming text SMT."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch


def _ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def stable_source_token_ids(tokenizer, text: str) -> list[int]:
    """Tokenize each source character independently for append-only streaming.

    Qwen's BPE tokenizer can change an earlier token when a later Chinese
    character arrives. Per-character tokenization keeps the IDs already sent
    to the cached decoder stable across streaming updates.
    """
    cache = getattr(tokenizer, "_stable_source_char_cache", None)
    if cache is None:
        cache = {}
        tokenizer._stable_source_char_cache = cache
    token_ids: list[int] = []
    for character in text:
        if character not in cache:
            cache[character] = _ids(tokenizer, character)
        token_ids.extend(cache[character])
    return token_ids


def build_stream_prompt(tokenizer, latent_only: bool = False) -> torch.Tensor:
    if latent_only:
        content = (
            "You translate Chinese speech represented by an acoustic condition into English. "
            "Each user message contains acoustic soft tokens for the newly available speech. "
            "Reply with only the corresponding English text. "
            "If no English text is ready, reply with <wait>. Do not invent facts."
        )
    else:
        content = (
            "You translate a Chinese speech transcription stream into English. "
            "Each user message contains only the newly available source text. "
            "Reply with only the corresponding newly available English text. "
            "If no English text is ready, reply with <wait>. Do not invent facts."
        )
    messages = [
        {
            "role": "system",
            "content": content,
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]


def turn_prefix_ids(tokenizer) -> Tuple[list[int], list[int], list[int]]:
    """Return user-open, assistant-open, and non-final turn-close ids."""
    user_open = _ids(tokenizer, "<|im_start|>user\n")
    assistant_open = _ids(tokenizer, "<|im_end|>\n<|im_start|>assistant\n")
    turn_close = _ids(tokenizer, "<|im_end|>\n")
    return user_open, assistant_open, turn_close


def latent_condition_ids(tokenizer) -> list[int]:
    return _ids(tokenizer, "Acoustic speech condition:\n")


def strip_control(ids: Sequence[int], wait_id: int, eos_id: int) -> list[int]:
    return [int(token_id) for token_id in ids if int(token_id) not in (wait_id, eos_id)]


def build_turn_sequence(
    tokenizer,
    prompt: torch.Tensor,
    source_chunks: Sequence[Sequence[int]],
    target_chunks: Sequence[Sequence[int]],
    wait_id: int,
    eos_id: int,
    device: str,
    explicit_wait: bool = False,
):
    if len(source_chunks) != len(target_chunks):
        raise ValueError("source and target chunk counts must match")
    user_open, assistant_open, turn_close = turn_prefix_ids(tokenizer)
    ids = [int(token_id) for token_id in prompt.tolist()]
    labels = [-100] * len(ids)
    for index, (source_chunk, target_chunk) in enumerate(
        zip(source_chunks, target_chunks)
    ):
        source_ids = strip_control(source_chunk, wait_id, eos_id)
        if explicit_wait:
            # The tokenwise cascade target is stored as visible target deltas.
            # Teach the decoder an explicit boundary after every non-final
            # delta, and EOS after the final delta.
            target_ids = strip_control(target_chunk, wait_id, eos_id)
        else:
            target_ids = [int(token_id) for token_id in target_chunk]
        structural = [*user_open, *source_ids, *assistant_open]
        ids.extend(structural)
        labels.extend([-100] * len(structural))
        ids.extend(target_ids)
        labels.extend(target_ids)
        if explicit_wait:
            control_id = wait_id if index != len(source_chunks) - 1 else eos_id
            ids.append(control_id)
            labels.append(control_id)
        if index != len(source_chunks) - 1:
            ids.extend(turn_close)
            labels.extend([-100] * len(turn_close))
    return (
        torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0),
    )
