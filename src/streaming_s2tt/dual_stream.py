"""Shared serialization and cached decoding helpers for dual-stream S2TT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .text_smt_turns import latent_condition_ids, turn_prefix_ids


def marker_ids(tokenizer, text: str) -> List[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def dual_stream_markers(tokenizer) -> Tuple[List[int], List[int]]:
    return marker_ids(tokenizer, "[SRC]"), marker_ids(tokenizer, "[TGT]")


def build_dual_stream_prompt(tokenizer) -> Tensor:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a streaming Chinese speech transcription and English translation "
                "assistant. For every newly available acoustic chunk, first output the new "
                "Chinese transcription after [SRC], then output the new English translation "
                "after [TGT]. Keep both streams incremental and do not repeat previous text. "
                "End each non-final chunk with <wait>. End the final chunk with the end marker. "
                "Output only the requested stream content and do not invent facts."
            ),
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]


def _wait_ids(wait_id) -> List[int]:
    if isinstance(wait_id, int):
        return [int(wait_id)]
    return [int(value) for value in wait_id]


def _as_ids(values: Iterable[int], wait_id, eos_id: int) -> List[int]:
    wait_set = set(_wait_ids(wait_id))
    # Early manifests used Qwen's code-completion marker as a placeholder for
    # wait. Strip it when reusing those manifests with the literal <wait>.
    wait_set.add(151660)
    return [
        int(value)
        for value in values
        if int(value) not in wait_set and int(value) != eos_id
    ]


def _stream_end(values: Iterable[int], wait_id, eos_id: int, final: bool) -> int:
    """Normalize a cached stream chunk to one explicit end decision."""
    for value in reversed(list(values)):
        value = int(value)
        if value == eos_id:
            return eos_id
        if value in set(_wait_ids(wait_id)):
            return _wait_ids(wait_id)[0]
    return eos_id if final else _wait_ids(wait_id)[0]


def build_source_training_sequence(
    llm,
    tokenizer,
    prompt: Tensor,
    soft_chunks: Sequence[Tensor],
    source_chunks: Sequence[Sequence[int]],
    wait_id,
    eos_id: int,
    source_loss_weight: float = 1.0,
    boundary_loss_weight: float = 3.0,
    latent_open_ids: Optional[Sequence[int]] = None,
    latent_close_ids: Optional[Sequence[int]] = None,
    source_prefix_ids: Optional[Sequence[int]] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Build the source-stream pretraining task.

    Each acoustic chunk teaches the decoder to emit only the new source text,
    followed by a learned wait decision. The final chunk closes with EOS.
    """
    if len(soft_chunks) != len(source_chunks):
        raise ValueError(
            f"source chunk counts must match: soft={len(soft_chunks)} "
            f"source={len(source_chunks)}"
        )
    user_open, assistant_open, turn_close = turn_prefix_ids(tokenizer)
    src_marker, _ = dual_stream_markers(tokenizer)
    device = prompt.device
    token_embedding = llm.get_input_embeddings()
    pieces_embeds = [token_embedding(prompt.unsqueeze(0)).squeeze(0)]
    pieces_labels = [
        torch.full((prompt.numel(),), -100, dtype=torch.long, device=device)
    ]
    pieces_weights = [
        torch.zeros(prompt.numel(), dtype=torch.float32, device=device)
    ]

    for index, (soft, source) in enumerate(zip(soft_chunks, source_chunks)):
        latent_open = (
            list(latent_open_ids)
            if latent_open_ids is not None
            else latent_condition_ids(tokenizer)
        )
        user_ids = [*user_open, *latent_open]
        pieces_embeds.append(
            token_embedding(torch.tensor(user_ids, dtype=torch.long, device=device))
        )
        pieces_labels.append(
            torch.full((len(user_ids),), -100, dtype=torch.long, device=device)
        )
        pieces_weights.append(
            torch.zeros(len(user_ids), dtype=torch.float32, device=device)
        )
        if soft.ndim == 3:
            soft = soft.squeeze(0)
        pieces_embeds.append(soft.to(token_embedding.weight.dtype))
        pieces_labels.append(
            torch.full((soft.size(0),), -100, dtype=torch.long, device=device)
        )
        pieces_weights.append(
            torch.zeros(soft.size(0), dtype=torch.float32, device=device)
        )
        if latent_close_ids:
            close = torch.tensor(latent_close_ids, dtype=torch.long, device=device)
            pieces_embeds.append(token_embedding(close))
            pieces_labels.append(
                torch.full((len(close),), -100, dtype=torch.long, device=device)
            )
            pieces_weights.append(
                torch.zeros(len(close), dtype=torch.float32, device=device)
            )
        assistant_ids = [*assistant_open, *src_marker]
        pieces_embeds.append(
            token_embedding(torch.tensor(assistant_ids, dtype=torch.long, device=device))
        )
        pieces_labels.append(
            torch.full((len(assistant_ids),), -100, dtype=torch.long, device=device)
        )
        pieces_weights.append(
            torch.zeros(len(assistant_ids), dtype=torch.float32, device=device)
        )
        if source_prefix_ids and index == 0:
            prefix = torch.tensor(source_prefix_ids, dtype=torch.long, device=device)
            pieces_embeds.append(token_embedding(prefix))
            pieces_labels.append(
                torch.full((len(prefix),), -100, dtype=torch.long, device=device)
            )
            pieces_weights.append(
                torch.zeros(len(prefix), dtype=torch.float32, device=device)
            )

        source_ids = _as_ids(source, wait_id, eos_id)
        end_ids = (
            [eos_id]
            if index + 1 == len(source_chunks)
            else _wait_ids(wait_id)
        )
        source_output = torch.tensor(
            [*source_ids, *end_ids], dtype=torch.long, device=device
        )
        pieces_embeds.append(token_embedding(source_output))
        pieces_labels.append(source_output)
        source_weights = torch.full(
            (source_output.numel(),),
            float(source_loss_weight),
            dtype=torch.float32,
            device=device,
        )
        source_weights[-1] *= float(boundary_loss_weight)
        pieces_weights.append(source_weights)
        if index + 1 < len(source_chunks):
            close = torch.tensor(turn_close, dtype=torch.long, device=device)
            pieces_embeds.append(token_embedding(close))
            pieces_labels.append(
                torch.full((len(close),), -100, dtype=torch.long, device=device)
            )
            pieces_weights.append(
                torch.zeros(len(close), dtype=torch.float32, device=device)
            )

    return (
        torch.cat(pieces_embeds, dim=0).unsqueeze(0),
        torch.cat(pieces_labels, dim=0).unsqueeze(0),
        torch.cat(pieces_weights, dim=0).unsqueeze(0),
    )


def build_dual_training_sequence(
    llm,
    tokenizer,
    prompt: Tensor,
    soft_chunks: Sequence[Tensor],
    source_chunks: Sequence[Sequence[int]],
    target_chunks: Sequence[Sequence[int]],
    wait_id,
    eos_id: int,
    source_loss_weight: float = 1.0,
    target_loss_weight: float = 1.0,
    boundary_loss_weight: float = 3.0,
    latent_open_ids: Optional[Sequence[int]] = None,
    latent_close_ids: Optional[Sequence[int]] = None,
    source_prefix_ids: Optional[Sequence[int]] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Build teacher-forced ``latent -> SRC -> TGT`` turns.

    The markers are part of the supervised sequence. This makes the switch from
    transcription to translation explicit while preserving one causal KV cache.
    """
    if not (len(soft_chunks) == len(source_chunks) == len(target_chunks)):
        raise ValueError(
            "dual stream chunk counts must match: "
            f"soft={len(soft_chunks)} source={len(source_chunks)} "
            f"target={len(target_chunks)}"
        )
    user_open, assistant_open, turn_close = turn_prefix_ids(tokenizer)
    src_marker, tgt_marker = dual_stream_markers(tokenizer)
    device = prompt.device
    token_embedding = llm.get_input_embeddings()

    pieces_embeds = [token_embedding(prompt.unsqueeze(0)).squeeze(0)]
    pieces_labels = [
        torch.full((prompt.numel(),), -100, dtype=torch.long, device=device)
    ]
    pieces_weights = [
        torch.zeros(prompt.numel(), dtype=torch.float32, device=device)
    ]

    for index, (soft, source, target) in enumerate(
        zip(soft_chunks, source_chunks, target_chunks)
    ):
        source_ids = _as_ids(source, wait_id, eos_id)
        target_ids = [int(value) for value in target]
        if not target_ids:
            target_ids = (
                _wait_ids(wait_id)
                if index + 1 < len(target_chunks)
                else [eos_id]
            )

        latent_open = (
            list(latent_open_ids)
            if latent_open_ids is not None
            else latent_condition_ids(tokenizer)
        )
        user_ids = [*user_open, *latent_open]
        user_embeds = token_embedding(
            torch.tensor(user_ids, dtype=torch.long, device=device)
        ).to(soft.dtype)
        pieces_embeds.append(user_embeds)
        pieces_labels.append(
            torch.full((len(user_ids),), -100, dtype=torch.long, device=device)
        )
        pieces_weights.append(
            torch.zeros(len(user_ids), dtype=torch.float32, device=device)
        )
        if soft.ndim == 3:
            soft = soft.squeeze(0)
        pieces_embeds.append(soft.to(token_embedding.weight.dtype))
        pieces_labels.append(
            torch.full((soft.size(0),), -100, dtype=torch.long, device=device)
        )
        pieces_weights.append(
            torch.zeros(soft.size(0), dtype=torch.float32, device=device)
        )
        if latent_close_ids:
            close = torch.tensor(latent_close_ids, dtype=torch.long, device=device)
            pieces_embeds.append(token_embedding(close))
            pieces_labels.append(
                torch.full((len(close),), -100, dtype=torch.long, device=device)
            )
            pieces_weights.append(
                torch.zeros(len(close), dtype=torch.float32, device=device
                )
            )
        assistant_ids = [*assistant_open, *src_marker]
        pieces_embeds.append(
            token_embedding(
                torch.tensor(assistant_ids, dtype=torch.long, device=device)
            )
        )
        pieces_labels.append(
            torch.full((len(assistant_ids),), -100, dtype=torch.long, device=device)
        )
        pieces_weights.append(
            torch.zeros(len(assistant_ids), dtype=torch.float32, device=device)
        )
        if source_prefix_ids and index == 0:
            prefix = torch.tensor(source_prefix_ids, dtype=torch.long, device=device)
            pieces_embeds.append(token_embedding(prefix))
            pieces_labels.append(
                torch.full((len(prefix),), -100, dtype=torch.long, device=device)
            )
            pieces_weights.append(
                torch.zeros(len(prefix), dtype=torch.float32, device=device
                )
            )

        # The first wait is the learned source-to-target switch for this
        # acoustic chunk. The decoder phase gives it the meaning of a source
        # wait; the second wait below ends the target stream.
        source_output = torch.tensor(
            [*source_ids, *_wait_ids(wait_id)], dtype=torch.long, device=device
        )
        pieces_embeds.append(token_embedding(source_output))
        pieces_labels.append(source_output)
        source_weights = torch.full(
            (source_output.numel(),),
            float(source_loss_weight),
            dtype=torch.float32,
            device=device,
        )
        source_weights[-1] *= float(boundary_loss_weight)
        pieces_weights.append(source_weights)
        target_prefix = torch.tensor(tgt_marker, dtype=torch.long, device=device)
        pieces_embeds.append(token_embedding(target_prefix))
        pieces_labels.append(
            torch.full((len(tgt_marker),), -100, dtype=torch.long, device=device)
        )
        pieces_weights.append(
            torch.zeros(len(tgt_marker), dtype=torch.float32, device=device)
        )
        target_end = (
            [eos_id]
            if index + 1 == len(target_chunks)
            else _wait_ids(wait_id)
        )
        target_output = torch.tensor(
            [*_as_ids(target_ids, wait_id, eos_id), *target_end],
            dtype=torch.long,
            device=device,
        )
        pieces_embeds.append(token_embedding(target_output))
        pieces_labels.append(target_output)
        target_weights = torch.full(
            (target_output.numel(),),
            float(target_loss_weight),
            dtype=torch.float32,
            device=device,
        )
        target_weights[-1] *= float(boundary_loss_weight)
        pieces_weights.append(target_weights)

        if index + 1 < len(target_chunks):
            close = torch.tensor(turn_close, dtype=torch.long, device=device)
            pieces_embeds.append(token_embedding(close))
            pieces_labels.append(
                torch.full((len(turn_close),), -100, dtype=torch.long, device=device)
            )
            pieces_weights.append(
                torch.zeros(len(turn_close), dtype=torch.float32, device=device)
            )

    inputs_embeds = torch.cat(pieces_embeds, dim=0).unsqueeze(0)
    labels = torch.cat(pieces_labels, dim=0).unsqueeze(0)
    weights = torch.cat(pieces_weights, dim=0).unsqueeze(0)
    return inputs_embeds, labels, weights


@dataclass
class DualStreamState:
    past_key_values: object
    attention_length: int
    source_ids: List[int] = field(default_factory=list)
    target_ids: List[int] = field(default_factory=list)
    chunks_seen: int = 0
    finished: bool = False


class DualStreamDecoder:
    """Greedy dual-stream decoder using one Qwen KV cache."""

    def __init__(
        self,
        llm,
        tokenizer,
        wait_id,
        eos_id: int,
        max_source_tokens: int = 48,
        max_target_tokens: int = 32,
        no_repeat_ngram_size: int = 4,
        target_no_repeat_ngram_size: int = 2,
        source_allowed_ids: Optional[Sequence[int]] = None,
        force_1d_position_ids: bool = False,
        latent_open_ids: Optional[Sequence[int]] = None,
        latent_close_ids: Optional[Sequence[int]] = None,
        source_prefix_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.wait_ids = _wait_ids(wait_id)
        if not self.wait_ids:
            raise ValueError("wait marker must contain at least one token")
        self.wait_id = self.wait_ids[0]
        self.eos_id = int(eos_id)
        self.max_source_tokens = max_source_tokens
        self.max_target_tokens = max_target_tokens
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.target_no_repeat_ngram_size = target_no_repeat_ngram_size
        self.source_allowed_ids = (
            None if source_allowed_ids is None else frozenset(int(x) for x in source_allowed_ids)
        )
        self.force_1d_position_ids = force_1d_position_ids
        self.user_open, self.assistant_open, self.turn_close = turn_prefix_ids(tokenizer)
        self.src_marker, self.tgt_marker = dual_stream_markers(tokenizer)
        self.latent_marker = (
            list(latent_open_ids)
            if latent_open_ids is not None
            else latent_condition_ids(tokenizer)
        )
        self.latent_close_marker = [] if latent_close_ids is None else list(latent_close_ids)
        self.source_prefix_marker = [] if source_prefix_ids is None else list(source_prefix_ids)
        self.state: Optional[DualStreamState] = None

    @property
    def source_ids(self) -> List[int]:
        return [] if self.state is None else self.state.source_ids

    @property
    def target_ids(self) -> List[int]:
        return [] if self.state is None else self.state.target_ids

    def _device(self) -> torch.device:
        """Use the LLM's device so the decoder is testable off CUDA too."""
        return self.llm.get_input_embeddings().weight.device

    @torch.no_grad()
    def start(self) -> None:
        device = self._device()
        prompt = build_dual_stream_prompt(self.tokenizer).to(device).unsqueeze(0)
        kwargs = {
            "input_ids": prompt,
            "attention_mask": torch.ones_like(prompt),
            "use_cache": True,
        }
        if self.force_1d_position_ids:
            kwargs["position_ids"] = torch.arange(
                prompt.size(1), device=device, dtype=torch.long
            ).unsqueeze(0)
        output = self.llm(**kwargs)
        self.state = DualStreamState(
            output.past_key_values,
            prompt.size(1),
        )

    def _require_state(self) -> DualStreamState:
        if self.state is None:
            raise RuntimeError("call start() before decoding")
        return self.state

    @torch.no_grad()
    def _append_ids(self, ids: Sequence[int]) -> Tensor:
        state = self._require_state()
        if not ids:
            # The caller only uses the returned logits after appending a real
            # prefix, so this branch is intentionally not exposed.
            raise ValueError("cannot append an empty token sequence")
        device = self._device()
        values = torch.tensor([list(ids)], dtype=torch.long, device=device)
        start = state.attention_length
        state.attention_length += values.size(1)
        kwargs = {
            "input_ids": values,
            "attention_mask": torch.ones(
                1, state.attention_length, dtype=torch.long, device=device
            ),
            "past_key_values": state.past_key_values,
            "use_cache": True,
        }
        if self.force_1d_position_ids:
            kwargs["position_ids"] = torch.arange(
                start, state.attention_length, device=device, dtype=torch.long
            ).unsqueeze(0)
        output = self.llm(**kwargs)
        state.past_key_values = output.past_key_values
        return output.logits[:, -1].float()

    @torch.no_grad()
    def _append_embeds(self, embeds: Tensor) -> Tensor:
        state = self._require_state()
        if embeds.ndim == 4:
            embeds = embeds.reshape(embeds.size(0), -1, embeds.size(-1))
        elif embeds.ndim == 2:
            embeds = embeds.unsqueeze(0)
        embeds = embeds.to(
            dtype=self.llm.get_input_embeddings().weight.dtype,
            device=self._device(),
        )
        start = state.attention_length
        state.attention_length += embeds.size(1)
        kwargs = {
            "inputs_embeds": embeds,
            "attention_mask": torch.ones(
                1, state.attention_length, dtype=torch.long, device=self._device()
            ),
            "past_key_values": state.past_key_values,
            "use_cache": True,
        }
        if self.force_1d_position_ids:
            kwargs["position_ids"] = torch.arange(
                start, state.attention_length, device=self._device(), dtype=torch.long
            ).unsqueeze(0)
        output = self.llm(**kwargs)
        state.past_key_values = output.past_key_values
        return output.logits[:, -1].float()

    def _mask_repeat(
        self, logits: Tensor, history: Sequence[int], size: Optional[int] = None
    ) -> Tensor:
        size = self.no_repeat_ngram_size if size is None else int(size)
        if size <= 1 or len(history) < size - 1:
            return logits
        prefix = list(history[-(size - 1) :])
        blocked = {
            history[i + size - 1]
            for i in range(len(history) - size + 1)
            if history[i : i + size - 1] == prefix
        }
        if not blocked:
            return logits
        logits = logits.clone()
        logits[:, list(blocked)] = float("-inf")
        return logits

    def _mask_source_vocab(self, logits: Tensor) -> Tensor:
        if self.source_allowed_ids is None:
            return logits
        allowed = torch.tensor(
            sorted(self.source_allowed_ids | set(self.wait_ids)),
            dtype=torch.long,
            device=logits.device,
        )
        masked = torch.full_like(logits, float("-inf"))
        masked[:, allowed] = logits[:, allowed]
        return masked

    @staticmethod
    def _has_suffix(values: Sequence[int], suffix: Sequence[int]) -> bool:
        return len(values) >= len(suffix) and list(values[-len(suffix) :]) == list(suffix)

    @torch.no_grad()
    def _start_chunk(self, soft_tokens: Optional[Tensor] = None) -> Tensor:
        logits = self._append_ids([*self.user_open, *self.latent_marker])
        if soft_tokens is not None:
            logits = self._append_embeds(soft_tokens)
            if self.latent_close_marker:
                logits = self._append_ids(self.latent_close_marker)
        logits = self._append_ids([*self.assistant_open, *self.src_marker])
        if self.source_prefix_marker and self._require_state().chunks_seen == 0:
            logits = self._append_ids(self.source_prefix_marker)
        self._require_state().chunks_seen += 1
        return logits

    @torch.no_grad()
    def _decode_source(self, logits: Tensor, limit: Optional[int]) -> Tuple[Tensor, List[int]]:
        """Generate source text until the model emits its wait decision."""
        source_delta: List[int] = []
        source_limit = max(1, int(limit)) if limit is not None else self.max_source_tokens
        saw_wait = False
        for _ in range(source_limit):
            logits = self._mask_repeat(
                logits, [*self._require_state().source_ids, *source_delta]
            )
            logits = self._mask_source_vocab(logits)
            # Source chunks close with wait. EOS belongs to the target stream.
            logits[:, self.eos_id] = float("-inf")
            token = int(logits.argmax(dim=-1).item())
            source_delta.append(token)
            logits = self._append_ids([token])
            if self._has_suffix(source_delta, self.wait_ids):
                del source_delta[-len(self.wait_ids) :]
                saw_wait = True
                break
        if not saw_wait:
            # Safety ceiling only. It is not the normal chunk boundary.
            self._append_ids(self.wait_ids)
        self._require_state().source_ids.extend(source_delta)
        return logits, source_delta

    @torch.no_grad()
    def accept_source_only(
        self,
        soft_tokens: Tensor,
        is_final: bool,
        source_budget: Optional[int] = None,
        source_ids_override: Optional[Sequence[int]] = None,
    ) -> Tuple[List[int], bool]:
        """Consume one acoustic chunk during source-stream pretraining/eval."""
        state = self._require_state()
        if state.finished:
            return [], True
        logits = self._start_chunk(soft_tokens)
        if source_ids_override is not None:
            source_delta = _as_ids(source_ids_override, self.wait_ids, self.eos_id)
            if source_delta:
                self._append_ids(source_delta)
            self._append_ids([self.eos_id] if is_final else self.wait_ids)
            state.source_ids.extend(source_delta)
            if is_final:
                state.finished = True
            else:
                self._append_ids(self.turn_close)
            return source_delta, state.finished
        source_delta: List[int] = []
        source_limit = (
            max(1, int(source_budget))
            if source_budget is not None
            else self.max_source_tokens
        )
        for _ in range(source_limit):
            logits = self._mask_repeat(
                logits, [*state.source_ids, *source_delta]
            )
            logits = self._mask_source_vocab(logits)
            if not is_final:
                logits[:, self.eos_id] = float("-inf")
            token = int(logits.argmax(dim=-1).item())
            if token == self.eos_id:
                self._append_ids([token])
                state.finished = True
                break
            source_delta.append(token)
            logits = self._append_ids([token])
            if self._has_suffix(source_delta, self.wait_ids):
                del source_delta[-len(self.wait_ids) :]
                if is_final:
                    self._append_ids([self.eos_id])
                    state.finished = True
                else:
                    self._append_ids(self.turn_close)
                break
        else:
            self._append_ids([self.eos_id] if is_final else self.wait_ids)
            if is_final:
                state.finished = True
            else:
                self._append_ids(self.turn_close)
        state.source_ids.extend(source_delta)
        return source_delta, state.finished

    @torch.no_grad()
    def accept(
        self,
        soft_tokens: Optional[Tensor],
        is_final: bool,
        source_budget: Optional[int] = None,
        target_budget: Optional[int] = None,
        source_ids_override: Optional[Sequence[int]] = None,
    ) -> Tuple[List[int], List[int], bool]:
        state = self._require_state()
        if state.finished:
            return [], [], True

        logits = self._start_chunk(soft_tokens)
        if soft_tokens is None:
            # Close an audio stream that ended during a blank-only tail. There
            # is no new source text to decode, so emit the structural source
            # wait directly and let the target stream finish from its cache.
            logits = self._append_ids(self.wait_ids)
            source_delta: List[int] = []
        elif source_ids_override is not None:
            source_delta = _as_ids(source_ids_override, self.wait_ids, self.eos_id)
            if source_delta:
                logits = self._append_ids(source_delta)
            logits = self._append_ids(self.wait_ids)
            state.source_ids.extend(source_delta)
        else:
            logits, source_delta = self._decode_source(logits, source_budget)
        target_delta: List[int] = []
        logits = self._append_ids(self.tgt_marker)

        target_limit = (
            max(1, int(target_budget))
            if target_budget is not None
            else self.max_target_tokens
        )
        target_generated: List[int] = []
        for _ in range(target_limit):
            logits = self._mask_repeat(
                logits,
                [*state.target_ids, *target_generated],
                self.target_no_repeat_ngram_size,
            )
            if not is_final:
                logits[:, self.eos_id] = float("-inf")
            token = int(logits.argmax(dim=-1).item())
            if token == self.eos_id:
                self._append_ids([token])
                target_delta = target_generated
                state.target_ids.extend(target_delta)
                state.finished = True
                break
            target_generated.append(token)
            logits = self._append_ids([token])
            if self._has_suffix(target_generated, self.wait_ids):
                target_delta = target_generated[: -len(self.wait_ids)]
                state.target_ids.extend(target_delta)
                if is_final:
                    # A final target may emit wait; close it defensively.
                    self._append_ids([self.eos_id])
                    state.finished = True
                else:
                    self._append_ids(self.turn_close)
                break
        else:
            target_delta = target_generated
            state.target_ids.extend(target_delta)
            if is_final:
                self._append_ids([self.eos_id])
                state.finished = True
            else:
                self._append_ids([*self.wait_ids, *self.turn_close])

        return source_delta, target_delta, state.finished

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(
            list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=True
        ).strip()
