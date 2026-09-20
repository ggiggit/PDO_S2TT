"""Native Qwen-ASR continuous-latent streaming with one persistent KV cache.

The Qwen-ASR thinker is kept in its native audio-prompt format.  Audio frames
are inserted as continuous embeddings in ``audio_pad`` positions; no text
token is used as an acoustic bridge.  Translation is intentionally lagged by
one acoustic chunk, so the target stream can be decoded before the current
chunk's source stream is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Iterable, Optional, Sequence
import unicodedata

import torch
from torch import Tensor

from .dual_stream import _as_ids, _wait_ids
from .text_smt_turns import turn_prefix_ids


def _ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def chinese_source_token_ids(tokenizer) -> set[int]:
    """Return tokenizer pieces that can occur in Chinese source text."""
    alphabet = (
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ，。！？；：、"
        "（）《》“”‘’\"'()[]+-*/%:;,.!?_—…·"
    )
    alphabet += "".join(chr(code) for code in range(0x3400, 0x4DB0))
    alphabet += "".join(chr(code) for code in range(0x4E00, 0xA000))
    allowed: set[int] = set()
    for character in alphabet:
        allowed.update(_ids(tokenizer, character))
    return allowed


def source_text_token_ids(tokenizer, language: str) -> set[int] | None:
    """Return a source-language vocabulary mask when one is reliable.

    Chinese decoding uses the established character mask.  English requires
    complete tokenizer pieces rather than only single-character ids, so scan
    decoded vocabulary entries and keep Latin text, numbers, and punctuation.
    Unknown languages are left unmasked instead of applying the wrong script.
    """
    normalized = language.strip().lower().replace("_", "-")
    if normalized in {"chinese", "cmn", "zh", "zh-cn", "cmn-hans-cn"}:
        return chinese_source_token_ids(tokenizer)
    if normalized not in {"english", "eng", "en", "en-us"}:
        return None

    special_ids = set(getattr(tokenizer, "all_special_ids", ()) or ())
    allowed: set[int] = set()
    for token_id in range(len(tokenizer)):
        if token_id in special_ids:
            continue
        text = tokenizer.decode(
            [token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if not text or "\ufffd" in text:
            continue
        valid = True
        for character in text:
            category = unicodedata.category(character)
            if character.isspace() or category[0] in {"N", "P", "S", "M"}:
                continue
            if category[0] == "L" and "LATIN" in unicodedata.name(character, ""):
                continue
            valid = False
            break
        if valid:
            allowed.add(token_id)
    return allowed


def native_system_prompt(tokenizer, instruction: str = "") -> Tensor:
    """Return the minimal system prefix used by the native Qwen-ASR prompt."""
    text = f"<|im_start|>system\n{instruction.strip()}<|im_end|>\n"
    return torch.tensor(_ids(tokenizer, text), dtype=torch.long)


def _append_ids(
    embeds: list[Tensor], labels: list[Tensor], weights: list[Tensor],
    token_embedding, values: Iterable[int], device: torch.device,
) -> None:
    values = list(int(value) for value in values)
    if not values:
        return
    ids = torch.tensor(values, dtype=torch.long, device=device)
    embeds.append(token_embedding(ids))
    labels.append(torch.full((len(values),), -100, dtype=torch.long, device=device))
    weights.append(torch.zeros(len(values), dtype=torch.float32, device=device))


def _append_supervised(
    embeds: list[Tensor], labels: list[Tensor], weights: list[Tensor],
    token_embedding, values: Iterable[int], device: torch.device,
    loss_weight: float, boundary_loss_weight: float,
) -> None:
    values = list(int(value) for value in values)
    if not values:
        return
    ids = torch.tensor(values, dtype=torch.long, device=device)
    embeds.append(token_embedding(ids))
    labels.append(ids)
    value_weights = torch.full(
        (len(values),), float(loss_weight), dtype=torch.float32, device=device
    )
    value_weights[-1] *= float(boundary_loss_weight)
    weights.append(value_weights)


def _clean(values: Sequence[int], wait_ids: Sequence[int], eos_id: int) -> list[int]:
    return _as_ids(values, wait_ids, eos_id)


def build_native_dual_training_sequence(
    llm,
    tokenizer,
    soft_chunks: Sequence[Tensor],
    source_chunks: Sequence[Sequence[int]],
    target_chunks: Sequence[Sequence[int]],
    wait_id,
    eos_id: int,
    source_loss_weight: float = 1.0,
    target_loss_weight: float = 1.0,
    boundary_loss_weight: float = 3.0,
    audio_start_id: int = 151669,
    audio_end_id: int = 151670,
    audio_pad_id: int = 151676,
    source_prefix_ids: Optional[Sequence[int]] = None,
    target_marker_ids: Optional[Sequence[int]] = None,
    source_resume_ids: Optional[Sequence[int]] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build the teacher-forced sequence used by native persistent decoding.

    For chunk ``i > 0`` the target delta from chunk ``i - 1`` is emitted before
    the source delta from chunk ``i``.  This makes the target stream causal with
    respect to previous source/target context without waiting for the current
    source generation to finish.  The last target delta is flushed after the
    final source delta.
    """
    if not (len(soft_chunks) == len(source_chunks) == len(target_chunks)):
        raise ValueError("native dual streams must have equal chunk counts")
    if not soft_chunks:
        raise ValueError("native dual stream needs at least one acoustic chunk")

    device = soft_chunks[0].device
    token_embedding = llm.get_input_embeddings()
    wait_ids = _wait_ids(wait_id)
    user_open, assistant_open, turn_close = turn_prefix_ids(tokenizer)
    source_prefix_ids = list(source_prefix_ids or _ids(tokenizer, "language Chinese<asr_text>"))
    target_marker_ids = list(target_marker_ids or _ids(tokenizer, "\n<translation>\n"))
    source_resume_ids = list(source_resume_ids or _ids(tokenizer, "\n<asr_text>"))

    embeds = [token_embedding(native_system_prompt(tokenizer).to(device))]
    labels = [torch.full((embeds[0].size(0),), -100, dtype=torch.long, device=device)]
    weights = [torch.zeros(embeds[0].size(0), dtype=torch.float32, device=device)]

    for index, (soft, source, target) in enumerate(
        zip(soft_chunks, source_chunks, target_chunks)
    ):
        if soft.ndim == 3:
            soft = soft.squeeze(0)
        if soft.ndim != 2:
            raise ValueError(f"native latent must be [frames, hidden], got {tuple(soft.shape)}")
        _append_ids(embeds, labels, weights, token_embedding, user_open, device)
        latent_ids = [audio_start_id, *([audio_pad_id] * soft.size(0)), audio_end_id]
        latent_token_ids = torch.tensor(latent_ids, dtype=torch.long, device=device)
        latent_embeds = token_embedding(latent_token_ids)
        latent_embeds[1 : 1 + soft.size(0)] = soft.to(latent_embeds.dtype)
        embeds.append(latent_embeds)
        labels.append(torch.full((latent_embeds.size(0),), -100, dtype=torch.long, device=device))
        weights.append(torch.zeros(latent_embeds.size(0), dtype=torch.float32, device=device))
        _append_ids(embeds, labels, weights, token_embedding, [*assistant_open, *source_prefix_ids], device)

        if index > 0:
            previous_target = _clean(target_chunks[index - 1], wait_ids, eos_id)
            _append_ids(embeds, labels, weights, token_embedding, target_marker_ids, device)
            target_end = [*previous_target, *wait_ids]
            _append_supervised(
                embeds, labels, weights, token_embedding, target_end, device,
                target_loss_weight, boundary_loss_weight,
            )
            _append_ids(embeds, labels, weights, token_embedding, source_resume_ids, device)

        source_ids = _clean(source, wait_ids, eos_id)
        source_end = [*source_ids, *(wait_ids if index + 1 < len(soft_chunks) else [eos_id])]
        _append_supervised(
            embeds, labels, weights, token_embedding, source_end, device,
            source_loss_weight, boundary_loss_weight,
        )
        if index + 1 < len(soft_chunks):
            _append_ids(embeds, labels, weights, token_embedding, turn_close, device)

    # Flush the target belonging to the final acoustic chunk after its source.
    _append_ids(embeds, labels, weights, token_embedding, target_marker_ids, device)
    final_target = _clean(target_chunks[-1], wait_ids, eos_id)
    _append_supervised(
        embeds, labels, weights, token_embedding, [*final_target, eos_id], device,
        target_loss_weight, boundary_loss_weight,
    )
    return (
        torch.cat(embeds, dim=0).unsqueeze(0),
        torch.cat(labels, dim=0).unsqueeze(0),
        torch.cat(weights, dim=0).unsqueeze(0),
    )


def build_native_source_training_sequence(
    llm,
    tokenizer,
    soft_chunks: Sequence[Tensor],
    source_chunks: Sequence[Sequence[int]],
    wait_id,
    eos_id: int,
    source_loss_weight: float = 1.0,
    boundary_loss_weight: float = 3.0,
    audio_start_id: int = 151669,
    audio_end_id: int = 151670,
    audio_pad_id: int = 151676,
    source_prefix_ids: Optional[Sequence[int]] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build source-only native turns for the first curriculum stage."""
    if len(soft_chunks) != len(source_chunks) or not soft_chunks:
        raise ValueError("native source chunks must be non-empty and aligned")
    device = soft_chunks[0].device
    token_embedding = llm.get_input_embeddings()
    wait_ids = _wait_ids(wait_id)
    user_open, assistant_open, turn_close = turn_prefix_ids(tokenizer)
    source_prefix_ids = list(source_prefix_ids or _ids(tokenizer, "language Chinese<asr_text>"))
    embeds = [token_embedding(native_system_prompt(tokenizer).to(device))]
    labels = [torch.full((embeds[0].size(0),), -100, dtype=torch.long, device=device)]
    weights = [torch.zeros(embeds[0].size(0), dtype=torch.float32, device=device)]
    for index, (soft, source) in enumerate(zip(soft_chunks, source_chunks)):
        if soft.ndim == 3:
            soft = soft.squeeze(0)
        _append_ids(embeds, labels, weights, token_embedding, user_open, device)
        latent_ids = [audio_start_id, *([audio_pad_id] * soft.size(0)), audio_end_id]
        latent_token_ids = torch.tensor(latent_ids, dtype=torch.long, device=device)
        latent_embeds = token_embedding(latent_token_ids)
        latent_embeds[1 : 1 + soft.size(0)] = soft.to(latent_embeds.dtype)
        embeds.append(latent_embeds)
        labels.append(torch.full((latent_embeds.size(0),), -100, dtype=torch.long, device=device))
        weights.append(torch.zeros(latent_embeds.size(0), dtype=torch.float32, device=device))
        _append_ids(embeds, labels, weights, token_embedding, [*assistant_open, *source_prefix_ids], device)
        source_ids = _clean(source, wait_ids, eos_id)
        source_end = [*source_ids, *(wait_ids if index + 1 < len(soft_chunks) else [eos_id])]
        _append_supervised(
            embeds, labels, weights, token_embedding, source_end, device,
            source_loss_weight, boundary_loss_weight,
        )
        if index + 1 < len(soft_chunks):
            _append_ids(embeds, labels, weights, token_embedding, turn_close, device)
    return (
        torch.cat(embeds, dim=0).unsqueeze(0),
        torch.cat(labels, dim=0).unsqueeze(0),
        torch.cat(weights, dim=0).unsqueeze(0),
    )


@dataclass
class NativeDualState:
    past_key_values: object
    attention_length: int
    source_ids: list[int] = field(default_factory=list)
    target_ids: list[int] = field(default_factory=list)
    chunks_seen: int = 0
    finished: bool = False


class NativePersistentQwenDecoder:
    """One Qwen-ASR thinker and one append-only KV cache."""

    def __init__(
        self,
        llm,
        tokenizer,
        wait_id,
        eos_id: int,
        audio_start_id: int = 151669,
        audio_end_id: int = 151670,
        audio_pad_id: int = 151676,
        max_source_tokens: int = 48,
        max_target_tokens: int = 32,
        temperature: float = 0.0,
        temporally_coupled_sampling: bool = False,
        source_allowed_ids: Optional[Sequence[int]] = None,
        target_forbidden_ids: Optional[Sequence[int]] = None,
        target_leading_forbidden_ids: Optional[Sequence[int]] = None,
        target_marker_ids: Optional[Sequence[int]] = None,
        source_prefix_ids: Optional[Sequence[int]] = None,
        source_resume_ids: Optional[Sequence[int]] = None,
        capture_source_hidden: bool = False,
        capture_source_posterior: bool = False,
        source_posterior_topk: int = 8,
        source_posterior_temperature: float = 1.0,
        source_final_punctuation_id: Optional[int] = None,
        source_strong_punctuation_ids: Optional[Sequence[int]] = None,
        adapter_name: Optional[str] = None,
        llm_lock: Optional[RLock] = None,
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.wait_ids = _wait_ids(wait_id)
        self.eos_id = int(eos_id)
        self.audio_start_id = int(audio_start_id)
        self.audio_end_id = int(audio_end_id)
        self.audio_pad_id = int(audio_pad_id)
        self.max_source_tokens = int(max_source_tokens)
        self.max_target_tokens = int(max_target_tokens)
        self.temperature = max(0.0, float(temperature))
        self.temporally_coupled_sampling = bool(temporally_coupled_sampling)
        self.source_allowed_ids = (
            None if source_allowed_ids is None else frozenset(int(value) for value in source_allowed_ids)
        )
        self.target_forbidden_ids = frozenset(
            int(value) for value in (target_forbidden_ids or ())
        )
        self.target_leading_forbidden_ids = frozenset(
            int(value) for value in (target_leading_forbidden_ids or ())
        )
        self.user_open, self.assistant_open, self.turn_close = turn_prefix_ids(tokenizer)
        self.source_prefix_ids = list(source_prefix_ids or _ids(tokenizer, "language Chinese<asr_text>"))
        self.target_marker_ids = list(target_marker_ids or _ids(tokenizer, "\n<translation>\n"))
        self.source_resume_ids = list(source_resume_ids or _ids(tokenizer, "\n<asr_text>"))
        self.capture_source_hidden = bool(capture_source_hidden)
        self.capture_source_posterior = bool(capture_source_posterior)
        self.source_posterior_topk = max(1, int(source_posterior_topk))
        self.source_posterior_temperature = max(
            1e-4, float(source_posterior_temperature)
        )
        self.source_final_punctuation_id = (
            None
            if source_final_punctuation_id is None
            else int(source_final_punctuation_id)
        )
        self.source_strong_punctuation_ids = frozenset(
            int(value) for value in (source_strong_punctuation_ids or ())
        )
        self.adapter_name = adapter_name
        self._llm_lock = llm_lock or RLock()
        self._last_hidden: Optional[Tensor] = None
        self._source_hidden_chunk: Optional[Tensor] = None
        self._source_hidden_token_ids: list[int] = []
        self._source_posterior_chunk: Optional[Tensor] = None
        self._source_posterior_topk_ids_chunk: Optional[Tensor] = None
        self._source_posterior_topk_probs_chunk: Optional[Tensor] = None
        self._source_terminal_posterior: Optional[Tensor] = None
        self._source_terminal_topk_ids: Optional[Tensor] = None
        self._source_terminal_topk_probs: Optional[Tensor] = None
        self.state: Optional[NativeDualState] = None

    @property
    def source_ids(self) -> list[int]:
        return [] if self.state is None else self.state.source_ids

    @property
    def target_ids(self) -> list[int]:
        return [] if self.state is None else self.state.target_ids

    def _device(self) -> torch.device:
        return self.llm.get_input_embeddings().weight.device

    def _positions(self, start: int, length: int) -> Tensor:
        values = torch.arange(start, start + length, device=self._device(), dtype=torch.long)
        return values.view(1, -1).unsqueeze(0).expand(3, 1, -1)

    def _forward(self, *, input_ids=None, inputs_embeds=None) -> Tensor:
        if self.state is None:
            start = 0
            past = None
        else:
            start = self.state.attention_length
            past = self.state.past_key_values
        length = input_ids.size(1) if input_ids is not None else inputs_embeds.size(1)
        total = start + length
        kwargs = {
            "attention_mask": torch.ones(1, total, dtype=torch.long, device=self._device()),
            "position_ids": self._positions(start, length),
            "past_key_values": past,
            "use_cache": True,
            "return_dict": True,
        }
        if self.capture_source_hidden:
            kwargs["output_hidden_states"] = True
        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        if inputs_embeds is not None:
            kwargs["inputs_embeds"] = inputs_embeds
        with self._llm_lock:
            if self.adapter_name is not None and hasattr(self.llm, "set_adapter"):
                self.llm.set_adapter(self.adapter_name)
            output = self.llm(**kwargs)
        if self.capture_source_hidden:
            self._last_hidden = output.hidden_states[-1][:, -1].detach()
        if self.state is None:
            self.state = NativeDualState(output.past_key_values, total)
        else:
            self.state.past_key_values = output.past_key_values
            self.state.attention_length = total
        return output.logits[:, -1].float()

    @torch.no_grad()
    def start(self) -> None:
        if self.state is not None:
            return
        ids = torch.tensor([native_system_prompt(self.tokenizer).tolist()], dtype=torch.long, device=self._device())
        self._forward(input_ids=ids)

    def _require_state(self) -> NativeDualState:
        if self.state is None:
            raise RuntimeError("call start() before decoding")
        return self.state

    @torch.no_grad()
    def _append_ids(self, values: Sequence[int]) -> Tensor:
        values = list(int(value) for value in values)
        if not values:
            raise ValueError("cannot append an empty token sequence")
        ids = torch.tensor([values], dtype=torch.long, device=self._device())
        return self._forward(input_ids=ids)

    @torch.no_grad()
    def _append_latent(self, latent: Tensor) -> Tensor:
        if latent.ndim == 3:
            latent = latent.squeeze(0)
        if latent.ndim != 2:
            raise ValueError(f"latent must be [frames, hidden], got {tuple(latent.shape)}")
        ids = [self.audio_start_id, *([self.audio_pad_id] * latent.size(0)), self.audio_end_id]
        token_ids = torch.tensor([ids], dtype=torch.long, device=self._device())
        embeds = self.llm.get_input_embeddings()(token_ids)
        embeds[:, 1 : 1 + latent.size(0)] = latent.to(device=self._device(), dtype=embeds.dtype).unsqueeze(0)
        return self._forward(input_ids=token_ids, inputs_embeds=embeds)

    @torch.no_grad()
    def _append_audio_turn(self, latent: Tensor) -> Tensor:
        logits = self._append_ids(self.user_open)
        logits = self._append_latent(latent)
        logits = self._append_ids([*self.turn_close, *self.assistant_open, *self.source_prefix_ids])
        return logits

    @staticmethod
    def _suffix(values: Sequence[int], suffix: Sequence[int]) -> bool:
        return len(values) >= len(suffix) and list(values[-len(suffix):]) == list(suffix)

    def _mask_repeat(self, logits: Tensor, history: Sequence[int], ngram: int = 3) -> Tensor:
        if ngram <= 1 or len(history) < ngram - 1:
            return logits
        prefix = list(history[-(ngram - 1):])
        blocked = {
            history[index + ngram - 1]
            for index in range(len(history) - ngram + 1)
            if history[index : index + ngram - 1] == prefix
        }
        if not blocked:
            return logits
        masked = logits.clone()
        masked[:, list(blocked)] = float("-inf")
        return masked

    def _mask_source_vocab(self, logits: Tensor) -> Tensor:
        if self.source_allowed_ids is None:
            return logits
        allowed = set(self.source_allowed_ids)
        allowed.update(self.wait_ids)
        allowed.add(self.eos_id)
        masked = torch.full_like(logits, float("-inf"))
        indices = torch.tensor(sorted(allowed), dtype=torch.long, device=logits.device)
        masked[:, indices] = logits[:, indices]
        return masked

    def _select_token(self, logits: Tensor) -> int:
        """Use deterministic decoding for benchmarks, temperature sampling online."""
        if self.temperature <= 0.0:
            return int(logits.argmax(dim=-1).item())
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        return int(torch.multinomial(probabilities, num_samples=1).item())

    @torch.no_grad()
    def _decode_stream(
        self, logits: Tensor, *, target: bool, is_final: bool
    ) -> tuple[list[int], bool]:
        limit = self.max_target_tokens if target else self.max_source_tokens
        history = self._require_state().target_ids if target else self._require_state().source_ids
        generated: list[int] = []
        predictive_hidden: list[Tensor] = []
        attempted_hidden: list[Tensor] = []
        posterior_embeds: list[Tensor] = []
        attempted_posterior: list[Tensor] = []
        posterior_topk_ids: list[Tensor] = []
        posterior_topk_probs: list[Tensor] = []
        attempted_topk_ids: list[Tensor] = []
        attempted_topk_probs: list[Tensor] = []
        terminal_posterior: Optional[Tensor] = None
        terminal_topk_ids: Optional[Tensor] = None
        terminal_topk_probs: Optional[Tensor] = None
        terminated = False
        for _ in range(max(1, limit)):
            logits = self._mask_repeat(logits, [*history, *generated], 2 if target else 3)
            if not target:
                logits = self._mask_source_vocab(logits)
            elif self.target_forbidden_ids:
                logits[:, list(self.target_forbidden_ids)] = float("-inf")
                if not history and not generated and self.target_leading_forbidden_ids:
                    logits[:, list(self.target_leading_forbidden_ids)] = float("-inf")
            if not is_final:
                logits[:, self.eos_id] = float("-inf")
            posterior = None
            if not target and self.capture_source_posterior:
                topk = min(self.source_posterior_topk, logits.size(-1))
                values, indices = logits.float().topk(topk, dim=-1)
                probabilities = torch.softmax(
                    values / self.source_posterior_temperature, dim=-1
                ).to(self.llm.get_input_embeddings().weight.dtype)
                candidates = self.llm.get_input_embeddings()(indices)
                posterior = (candidates * probabilities.unsqueeze(-1)).sum(dim=1)
                posterior = posterior.squeeze(0).detach().clone()
                attempted_posterior.append(posterior)
                attempted_topk_ids.append(indices.squeeze(0).detach().clone())
                attempted_topk_probs.append(probabilities.squeeze(0).detach().clone())
            token = self._select_token(logits)
            if not target and self.capture_source_hidden:
                if self._last_hidden is None:
                    raise RuntimeError("source hidden capture is enabled but no hidden state is available")
                hidden = self._last_hidden.squeeze(0).detach().clone()
                attempted_hidden.append(hidden)
            source_history = [*history, *generated]
            force_final_punctuation = (
                not target
                and is_final
                and token == self.eos_id
                and self.source_final_punctuation_id is not None
                and bool(source_history)
                and source_history[-1] not in self.source_strong_punctuation_ids
            )
            if force_final_punctuation:
                token = int(self.source_final_punctuation_id)
                generated.append(token)
                if self.capture_source_hidden:
                    predictive_hidden.append(hidden)
                if self.capture_source_posterior:
                    posterior_embeds.append(posterior)
                    posterior_topk_ids.append(attempted_topk_ids[-1])
                    posterior_topk_probs.append(attempted_topk_probs[-1])
                    terminal_posterior = posterior
                    terminal_topk_ids = attempted_topk_ids[-1]
                    terminal_topk_probs = attempted_topk_probs[-1]
                self._append_ids([token, self.eos_id])
                terminated = True
                break
            if token == self.eos_id:
                if not target and self.capture_source_posterior:
                    terminal_posterior = posterior
                    terminal_topk_ids = attempted_topk_ids[-1]
                    terminal_topk_probs = attempted_topk_probs[-1]
                self._append_ids([token])
                terminated = True
                break
            generated.append(token)
            if not target and self.capture_source_hidden:
                predictive_hidden.append(hidden)
            if not target and self.capture_source_posterior:
                posterior_embeds.append(posterior)
                posterior_topk_ids.append(attempted_topk_ids[-1])
                posterior_topk_probs.append(attempted_topk_probs[-1])
            logits = self._append_ids([token])
            if self._suffix(generated, self.wait_ids):
                if not target and self.capture_source_posterior:
                    terminal_index = len(posterior_embeds) - len(self.wait_ids)
                    terminal_posterior = posterior_embeds[terminal_index]
                    terminal_topk_ids = posterior_topk_ids[terminal_index]
                    terminal_topk_probs = posterior_topk_probs[terminal_index]
                del generated[-len(self.wait_ids):]
                if not target and self.capture_source_hidden:
                    del predictive_hidden[-len(self.wait_ids):]
                if not target and self.capture_source_posterior:
                    del posterior_embeds[-len(self.wait_ids):]
                    del posterior_topk_ids[-len(self.wait_ids):]
                    del posterior_topk_probs[-len(self.wait_ids):]
                terminated = True
                break
        source_history = [*history, *generated]
        append_final_punctuation = (
            not target
            and is_final
            and self.source_final_punctuation_id is not None
            and bool(source_history)
            and source_history[-1] not in self.source_strong_punctuation_ids
        )
        if append_final_punctuation:
            token = int(self.source_final_punctuation_id)
            generated.append(token)
            if self.capture_source_hidden:
                if self._last_hidden is None:
                    raise RuntimeError("final punctuation has no predictive hidden state")
                predictive_hidden.append(self._last_hidden.squeeze(0).detach().clone())
            if self.capture_source_posterior:
                topk = min(self.source_posterior_topk, logits.size(-1))
                values, indices = logits.float().topk(topk, dim=-1)
                probabilities = torch.softmax(
                    values / self.source_posterior_temperature, dim=-1
                ).to(self.llm.get_input_embeddings().weight.dtype)
                candidates = self.llm.get_input_embeddings()(indices)
                posterior = (candidates * probabilities.unsqueeze(-1)).sum(dim=1)
                posterior = posterior.squeeze(0).detach().clone()
                posterior_embeds.append(posterior)
                posterior_topk_ids.append(indices.squeeze(0).detach().clone())
                posterior_topk_probs.append(probabilities.squeeze(0).detach().clone())
                terminal_posterior = posterior
                terminal_topk_ids = posterior_topk_ids[-1]
                terminal_topk_probs = posterior_topk_probs[-1]
            self._append_ids([token, self.eos_id])
            terminated = True
        if target:
            self._require_state().target_ids.extend(generated)
        else:
            self._require_state().source_ids.extend(generated)
            if self.capture_source_hidden:
                if predictive_hidden:
                    self._source_hidden_chunk = torch.stack(predictive_hidden)
                elif attempted_hidden:
                    # A wait-only chunk still contributes one continuous state
                    # carrying the current acoustic/context evidence.
                    self._source_hidden_chunk = attempted_hidden[0].unsqueeze(0)
                else:
                    raise RuntimeError("source decoding produced neither tokens nor a predictive state")
                self._source_hidden_token_ids = list(generated)
            if self.capture_source_posterior:
                if posterior_embeds:
                    self._source_posterior_chunk = torch.stack(posterior_embeds)
                    self._source_posterior_topk_ids_chunk = torch.stack(posterior_topk_ids)
                    self._source_posterior_topk_probs_chunk = torch.stack(posterior_topk_probs)
                elif attempted_posterior:
                    self._source_posterior_chunk = attempted_posterior[0].unsqueeze(0)
                    self._source_posterior_topk_ids_chunk = attempted_topk_ids[0].unsqueeze(0)
                    self._source_posterior_topk_probs_chunk = attempted_topk_probs[0].unsqueeze(0)
                else:
                    raise RuntimeError("source decoding produced no posterior state")
                if terminal_posterior is None:
                    terminal_posterior = attempted_posterior[-1]
                    terminal_topk_ids = attempted_topk_ids[-1]
                    terminal_topk_probs = attempted_topk_probs[-1]
                self._source_terminal_posterior = terminal_posterior
                self._source_terminal_topk_ids = terminal_topk_ids
                self._source_terminal_topk_probs = terminal_topk_probs
        return generated, terminated

    def take_source_hidden_chunk(self) -> tuple[Tensor, list[int]]:
        """Return and clear the predictive source states from the latest chunk."""
        if not self.capture_source_hidden:
            raise RuntimeError("construct the decoder with capture_source_hidden=True")
        if self._source_hidden_chunk is None:
            raise RuntimeError("no source chunk has been decoded since the previous take")
        hidden = self._source_hidden_chunk
        token_ids = self._source_hidden_token_ids
        self._source_hidden_chunk = None
        self._source_hidden_token_ids = []
        return hidden, token_ids

    def take_source_posterior_chunk(self) -> Tensor:
        """Return and clear the soft lexical posterior from the latest chunk."""
        if not self.capture_source_posterior:
            raise RuntimeError("construct the decoder with capture_source_posterior=True")
        if self._source_posterior_chunk is None:
            raise RuntimeError("no source chunk has been decoded since the previous take")
        posterior = self._source_posterior_chunk
        self._source_posterior_chunk = None
        return posterior

    def take_source_posterior_topk_chunk(self) -> tuple[Tensor, Tensor]:
        """Return top-k token IDs/probabilities captured with the latest posterior chunk."""
        if not self.capture_source_posterior:
            raise RuntimeError("construct the decoder with capture_source_posterior=True")
        if (
            self._source_posterior_topk_ids_chunk is None
            or self._source_posterior_topk_probs_chunk is None
        ):
            raise RuntimeError("no source posterior candidates are available")
        ids = self._source_posterior_topk_ids_chunk
        probabilities = self._source_posterior_topk_probs_chunk
        self._source_posterior_topk_ids_chunk = None
        self._source_posterior_topk_probs_chunk = None
        return ids, probabilities

    def take_source_terminal_posterior(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return the posterior that predicted WAIT/EOS after the latest source chunk."""
        if not self.capture_source_posterior:
            raise RuntimeError("construct the decoder with capture_source_posterior=True")
        if (
            self._source_terminal_posterior is None
            or self._source_terminal_topk_ids is None
            or self._source_terminal_topk_probs is None
        ):
            raise RuntimeError("no terminal source posterior is available")
        posterior = self._source_terminal_posterior
        ids = self._source_terminal_topk_ids
        probabilities = self._source_terminal_topk_probs
        self._source_terminal_posterior = None
        self._source_terminal_topk_ids = None
        self._source_terminal_topk_probs = None
        return posterior, ids, probabilities

    @torch.no_grad()
    def accept(self, latent: Tensor, is_final: bool = False) -> tuple[list[int], list[int], bool]:
        state = self._require_state()
        if state.finished:
            return [], [], True
        logits = self._append_audio_turn(latent)
        source_delta: list[int] = []
        target_delta: list[int] = []

        if state.chunks_seen > 0:
            logits = self._append_ids(self.target_marker_ids)
            target_part, target_closed = self._decode_stream(
                logits, target=True, is_final=False
            )
            target_delta.extend(target_part)
            if not target_closed:
                logits = self._append_ids(self.wait_ids)
            logits = self._append_ids(self.source_resume_ids)

        source_part, source_closed = self._decode_stream(
            logits, target=False, is_final=is_final
        )
        source_delta.extend(source_part)
        state.chunks_seen += 1

        if is_final:
            logits = self._append_ids(self.target_marker_ids)
            target_part, target_closed = self._decode_stream(
                logits, target=True, is_final=True
            )
            target_delta.extend(target_part)
            if not target_closed:
                self._append_ids([self.eos_id])
            state.finished = True
        else:
            if not source_closed:
                self._append_ids(self.wait_ids)
            self._append_ids(self.turn_close)
        return source_delta, target_delta, state.finished

    @torch.no_grad()
    def accept_final_without_latent(self) -> tuple[list[int], list[int], bool]:
        """Close a stream when the last PCM call produced no new chunk.

        This is the exact-boundary case: the previous call consumed the last
        chunk with ``is_final=False``, so its target delta is still pending.
        Flush that target turn without inventing a silent acoustic chunk.
        """
        state = self._require_state()
        if state.finished:
            return [], [], True
        if state.chunks_seen == 0:
            self._append_ids([self.eos_id])
            state.finished = True
            return [], [], True

        logits = self._append_ids(self.target_marker_ids)
        target_delta, target_closed = self._decode_stream(
            logits, target=True, is_final=True
        )
        if not target_closed:
            self._append_ids([self.eos_id])
        state.finished = True
        return [], target_delta, True

    @torch.no_grad()
    def accept_source_only(self, latent: Tensor, is_final: bool = False) -> tuple[list[int], bool]:
        """Consume a chunk while measuring only the native source stream."""
        state = self._require_state()
        if state.finished:
            return [], True
        logits = self._append_audio_turn(latent)
        source_delta, source_closed = self._decode_stream(
            logits, target=False, is_final=is_final
        )
        state.chunks_seen += 1
        if is_final:
            if not source_closed:
                self._append_ids([self.eos_id])
            state.finished = True
        else:
            if not source_closed:
                self._append_ids(self.wait_ids)
            self._append_ids(self.turn_close)
        return source_delta, state.finished

    def decode(self, values: Sequence[int]) -> str:
        return self.tokenizer.decode(
            list(values), skip_special_tokens=True, clean_up_tokenization_spaces=True
        ).strip()
