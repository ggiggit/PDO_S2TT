"""Native Qwen continuous-latent streaming with routed KV caches.

This module is the strict dual-stream counterpart of ``native_dual_stream``.
The Qwen weights are shared, while source and target histories are kept in
different persistent caches.  With source lag zero, source_t is decoded first
and serialized in the target cache after audio_t but before target_t.  With
source lag one, target_t is decoded first and source_t becomes visible at the
next tick.  Target tokens never enter the source cache.

The acoustic chunk is produced once by the frontend and passed by reference
to both branches.  The two branch caches are intentionally separate because a
standard causal decoder cache cannot express the required asymmetric access:
source may not read target history, while target may read committed source
history.
"""

from __future__ import annotations

import copy
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import RLock
from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor

from .dual_stream import _as_ids, _wait_ids
from .native_dual_stream import (
    _append_ids,
    _append_supervised,
    _clean,
    _ids,
    chinese_source_token_ids,
    native_system_prompt,
)
from .text_smt_turns import turn_prefix_ids


@dataclass
class NativeBranchCache:
    """Persistent state for one logical text stream."""

    past_key_values: object
    attention_length: int
    emitted: list[int] = field(default_factory=list)


@dataclass
class SharedAcousticState:
    """Accounting for frontend chunks shared by both routed branches."""

    chunk_ids: list[int] = field(default_factory=list)
    frame_counts: list[int] = field(default_factory=list)

    @property
    def chunks_seen(self) -> int:
        return len(self.chunk_ids)


@dataclass
class NativeTrueDualState:
    source: NativeBranchCache
    target: NativeBranchCache
    acoustic: SharedAcousticState = field(default_factory=SharedAcousticState)
    source_ids: list[int] = field(default_factory=list)
    target_ids: list[int] = field(default_factory=list)
    finished: bool = False

    @property
    def chunks_seen(self) -> int:
        return self.acoustic.chunks_seen


class NativeTrueDualQwenDecoder:
    """One Qwen thinker with independent persistent source/target caches."""

    def __init__(
        self,
        llm,
        tokenizer,
        wait_id,
        eos_id: int,
        audio_start_id: int,
        audio_end_id: int,
        audio_pad_id: int,
        max_source_tokens: int = 48,
        max_target_tokens: int = 32,
        temperature: float = 0.0,
        temporally_coupled_sampling: bool = False,
        source_allowed_ids: Optional[Sequence[int]] = None,
        target_forbidden_ids: Optional[Sequence[int]] = None,
        target_leading_forbidden_ids: Optional[Sequence[int]] = None,
        source_prefix_ids: Optional[Sequence[int]] = None,
        target_marker_ids: Optional[Sequence[int]] = None,
        source_context_ids: Optional[Sequence[int]] = None,
        target_output_lag_chunks: int = 1,
        target_source_lag_chunks: int = 1,
        target_source_first_chunk_current: bool = False,
        target_latent_mode: str = "full",
        target_source_mode: str = "generated",
        source_adapter_name: Optional[str] = None,
        target_adapter_name: Optional[str] = None,
        target_no_repeat_ngram_size: int = 2,
        source_no_repeat_ngram_size: int = 3,
        target_min_tokens_before_wait: int = 0,
        target_min_chunks_before_wait: int = 0,
        target_force_token_if_empty: bool = False,
        target_wait_logit_bias: float = 0.0,
        target_wait_bias_first_token_only: bool = False,
        target_wait_bias_before_first_output: bool = False,
        target_wait_bias_after_waits: int = 0,
        target_adaptive_wait_bias: float = 0.0,
        target_adaptive_wait_margin: float = 0.0,
        source_decode: bool = True,
        defer_source: bool = False,
        async_source_pipeline: bool = False,
        target_system_instruction: str = "",
        target_prompt_mode: str = "native_asr_marker",
        target_decode_stride_chunks: int = 1,
        target_decode_warmup_chunks: int = 1,
        target_speculative_draft_reuse: bool = False,
        target_speculative_min_margin: float = 0.0,
        target_early_content_margin: Optional[float] = None,
        target_early_min_tokens: int = 0,
        target_user_instruction: str = (
            "Translate and commit this complete Chinese unit into English:\n"
        ),
        source_final_punctuation_id: Optional[int] = None,
        source_strong_punctuation_ids: Optional[Sequence[int]] = None,
        source_wait_id=None,
        target_wait_id=None,
        capture_source_topk: bool = False,
        source_topk: int = 8,
    ) -> None:
        self.llm = llm
        # PEFT adapter selection is mutable model state.  The optional async
        # source worker must not switch adapters while the target branch is in
        # the middle of a forward on the same Qwen instance.
        self._llm_lock = RLock()
        self.tokenizer = tokenizer
        self.source_wait_ids = _wait_ids(
            wait_id if source_wait_id is None else source_wait_id
        )
        self.target_wait_ids = _wait_ids(
            wait_id if target_wait_id is None else target_wait_id
        )
        # Backward-compatible alias for target-only helpers.
        self.wait_ids = self.target_wait_ids
        self.source_final_punctuation_id = (
            None
            if source_final_punctuation_id is None
            else int(source_final_punctuation_id)
        )
        self.source_strong_punctuation_ids = frozenset(
            int(value) for value in (source_strong_punctuation_ids or ())
        )
        self.capture_source_topk = bool(capture_source_topk)
        self.source_topk = max(1, int(source_topk))
        self.source_topk_trace: list[dict] = []
        self.last_stage5_eot_trace: dict | None = None
        self.eos_id = int(eos_id)
        self.audio_start_id = int(audio_start_id)
        self.audio_end_id = int(audio_end_id)
        self.audio_pad_id = int(audio_pad_id)
        self.max_source_tokens = int(max_source_tokens)
        self.max_target_tokens = int(max_target_tokens)
        self.temperature = max(0.0, float(temperature))
        self.temporally_coupled_sampling = bool(temporally_coupled_sampling)
        self.temporal_sampling_seed: Optional[int] = None
        if target_output_lag_chunks not in (0, 1):
            raise ValueError("target_output_lag_chunks must be 0 or 1")
        if target_source_lag_chunks not in (0, 1, 2):
            raise ValueError("target_source_lag_chunks must be 0, 1, or 2")
        self.target_output_lag_chunks = int(target_output_lag_chunks)
        self.target_source_lag_chunks = int(target_source_lag_chunks)
        self.target_source_first_chunk_current = bool(
            target_source_first_chunk_current
        )
        if target_latent_mode not in {"full", "zero", "shuffle", "previous", "empty"}:
            raise ValueError(f"unsupported target_latent_mode: {target_latent_mode}")
        if target_source_mode not in {"generated", "none", "oracle"}:
            raise ValueError(f"unsupported target_source_mode: {target_source_mode}")
        self.target_latent_mode = target_latent_mode
        self.target_source_mode = target_source_mode
        if target_prompt_mode == "chat_translation" and (
            self.target_output_lag_chunks != 0
            or self.target_source_lag_chunks != 0
            or self.target_source_mode != "generated"
        ):
            raise ValueError(
                "chat_translation requires target_output_lag_chunks=0, "
                "target_source_lag_chunks=0 and target_source_mode=generated"
            )
        if target_prompt_mode in {
            "stage5_target_only_chat",
            "stage5_temporary_draft_chat",
        } and (
            self.target_output_lag_chunks != 0
            or self.target_source_lag_chunks != 0
            or self.target_source_mode != "none"
            or source_decode
        ):
            raise ValueError(
                "target-only Stage 5 chat requires zero target/source lag, "
                "target_source_mode=none and source_decode=False"
            )
        self.source_adapter_name = source_adapter_name
        self.target_adapter_name = target_adapter_name
        self.target_no_repeat_ngram_size = max(0, int(target_no_repeat_ngram_size))
        self.source_no_repeat_ngram_size = max(0, int(source_no_repeat_ngram_size))
        self.target_min_tokens_before_wait = max(0, int(target_min_tokens_before_wait))
        self.target_min_chunks_before_wait = max(0, int(target_min_chunks_before_wait))
        self.target_force_token_if_empty = bool(target_force_token_if_empty)
        # Positive values make the target branch wait more; negative values
        # make it emit sooner. Zero preserves the checkpoint behavior.
        self.target_wait_logit_bias = float(target_wait_logit_bias)
        self.target_wait_bias_first_token_only = bool(
            target_wait_bias_first_token_only
        )
        self.target_wait_bias_before_first_output = bool(
            target_wait_bias_before_first_output
        )
        self.target_wait_bias_after_waits = max(0, int(target_wait_bias_after_waits))
        self.target_adaptive_wait_bias = float(target_adaptive_wait_bias)
        self.target_adaptive_wait_margin = float(target_adaptive_wait_margin)
        self.source_decode = bool(source_decode)
        self.defer_source = bool(
            defer_source
            and self.source_decode
            and self.target_source_lag_chunks == 1
            and self.target_source_mode == "generated"
        )
        self.async_source_pipeline = bool(
            async_source_pipeline
            and self.defer_source
            and self.target_source_lag_chunks == 1
        )
        self.target_system_instruction = str(target_system_instruction or "")
        if target_prompt_mode not in {
            "native_asr_marker",
            "chat_translation",
            "stage5_target_only_chat",
            "stage5_temporary_draft_chat",
        }:
            raise ValueError(f"unsupported target_prompt_mode: {target_prompt_mode}")
        self.target_prompt_mode = target_prompt_mode
        if target_decode_stride_chunks < 1:
            raise ValueError("target_decode_stride_chunks must be positive")
        if target_decode_warmup_chunks < 1:
            raise ValueError("target_decode_warmup_chunks must be positive")
        self.target_decode_stride_chunks = int(target_decode_stride_chunks)
        self.target_decode_warmup_chunks = int(target_decode_warmup_chunks)
        self.target_speculative_draft_reuse = bool(target_speculative_draft_reuse)
        self.target_speculative_min_margin = max(
            0.0, float(target_speculative_min_margin)
        )
        self.target_early_content_margin = (
            None
            if target_early_content_margin is None
            else float(target_early_content_margin)
        )
        self.target_early_min_tokens = max(0, int(target_early_min_tokens))
        self.speculative_candidate_tokens = 0
        self.speculative_accepted_tokens = 0
        self.speculative_verification_calls = 0
        self.target_user_instruction = str(target_user_instruction)
        self._source_executor: Optional[ThreadPoolExecutor] = None
        self._source_future: Optional[Future] = None
        self._previous_target_latent: Optional[Tensor] = None
        self._target_wait_streak = 0
        self._oracle_source_chunks: Optional[list[list[int]]] = None
        self._oracle_source_index = 0
        self._pending_source: Optional[tuple[Tensor, Optional[Tensor], bool]] = None
        self._queued_source_deltas: list[list[int]] = []
        self.source_allowed_ids = (
            None
            if source_allowed_ids is None
            else frozenset(int(value) for value in source_allowed_ids)
        )
        self.source_preview_initial_ids = tuple(
            index
            for index in sorted(self.source_allowed_ids or ())
            if re.search(
                r"[A-Za-z0-9\u3400-\u9fff]",
                tokenizer.decode(
                    [index],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ),
            )
        )
        # Keep the placeholder out even when an older checkpoint/runtime does
        # not carry the tokenizer-derived forbidden-id list.
        forbidden = (
            tuple(target_forbidden_ids or ())
            + tuple(_ids(tokenizer, "None"))
            + tuple(_ids(tokenizer, " None"))
        )
        self.target_forbidden_ids = frozenset(int(value) for value in forbidden)
        self.target_leading_forbidden_ids = frozenset(
            int(value) for value in (target_leading_forbidden_ids or ())
        )
        self.user_open, self.assistant_open, self.turn_close = turn_prefix_ids(tokenizer)
        self.source_prefix_ids = list(
            source_prefix_ids or _ids(tokenizer, "language Chinese<asr_text>")
        )
        self.target_marker_ids = list(
            target_marker_ids or _ids(tokenizer, "\n<translation>\n")
        )
        self.source_context_ids = list(
            source_context_ids or _ids(tokenizer, "\n<asr_text>")
        )
        self.target_user_instruction_ids = _ids(
            tokenizer, self.target_user_instruction
        )
        self.state: Optional[NativeTrueDualState] = None

    def _device(self) -> torch.device:
        return self.llm.get_input_embeddings().weight.device

    def _positions(self, start: int, length: int) -> Tensor:
        values = torch.arange(
            start, start + length, device=self._device(), dtype=torch.long
        )
        return values.view(1, -1).unsqueeze(0).expand(3, 1, -1)

    def _branch(self, target: bool) -> NativeBranchCache:
        if self.state is None:
            raise RuntimeError("call start() before decoding")
        if not target and not self.source_decode:
            raise RuntimeError("source branch is disabled in target-only mode")
        return self.state.target if target else self.state.source

    @property
    def persistent_cache_count(self) -> int:
        """Return the number of live decoder histories for this utterance."""
        return 2 if self.source_decode else 1

    @torch.no_grad()
    def _forward(
        self,
        *,
        target: bool,
        input_ids=None,
        inputs_embeds=None,
        return_all_logits: bool = False,
    ) -> Tensor:
        branch = self._branch(target)
        adapter_name = self.target_adapter_name if target else self.source_adapter_name
        length = input_ids.size(1) if input_ids is not None else inputs_embeds.size(1)
        total = branch.attention_length + length
        kwargs = {
            "attention_mask": torch.ones(
                1, total, dtype=torch.long, device=self._device()
            ),
            "position_ids": self._positions(branch.attention_length, length),
            "past_key_values": branch.past_key_values,
            "use_cache": True,
            "return_dict": True,
        }
        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        else:
            kwargs["inputs_embeds"] = inputs_embeds
        with self._llm_lock:
            if adapter_name is not None and hasattr(self.llm, "set_adapter"):
                self.llm.set_adapter(adapter_name)
            output = self.llm(**kwargs)
        branch.past_key_values = output.past_key_values
        branch.attention_length = total
        if return_all_logits:
            return output.logits.float()
        return output.logits[:, -1].float()

    @torch.no_grad()
    def start(self) -> None:
        if self.state is not None:
            return
        self._previous_target_latent = None
        self._target_wait_streak = 0
        self.speculative_candidate_tokens = 0
        self.speculative_accepted_tokens = 0
        self.speculative_verification_calls = 0
        self._oracle_source_index = 0
        self._pending_source = None
        self._queued_source_deltas = []
        self.source_topk_trace = []
        self.last_stage5_eot_trace = None
        device = self._device()
        target_prefix = native_system_prompt(
            self.tokenizer, self.target_system_instruction
        ).to(device).tolist()
        if self.target_prompt_mode == "stage5_temporary_draft_chat":
            target_prefix.extend([*self.user_open, *self.target_user_instruction_ids])
        target_ids = torch.tensor(target_prefix, dtype=torch.long, device=device).unsqueeze(0)
        source_cache = NativeBranchCache(None, 0)
        # In target-only mode the source prompt is not forwarded and no source
        # KV tensors are allocated. This is a real one-cache execution path,
        # not merely a source decoder whose emitted text is ignored.
        with self._llm_lock:
            if self.source_decode:
                source_ids = native_system_prompt(self.tokenizer).to(device).unsqueeze(0)
                if self.source_adapter_name is not None and hasattr(self.llm, "set_adapter"):
                    self.llm.set_adapter(self.source_adapter_name)
                source_output = self.llm(
                    input_ids=source_ids,
                    attention_mask=torch.ones_like(source_ids),
                    use_cache=True,
                    return_dict=True,
                )
                source_cache = NativeBranchCache(
                    source_output.past_key_values, source_ids.size(1)
                )
            if self.target_adapter_name is not None and hasattr(self.llm, "set_adapter"):
                self.llm.set_adapter(self.target_adapter_name)
            target_output = self.llm(
                input_ids=target_ids,
                attention_mask=torch.ones_like(target_ids),
                use_cache=True,
                return_dict=True,
            )
        self.state = NativeTrueDualState(
            source=source_cache,
            target=NativeBranchCache(target_output.past_key_values, target_ids.size(1)),
        )

    @torch.no_grad()
    def reset(self) -> None:
        """Abort one utterance only after an async source worker is joined."""
        if self._source_future is not None:
            self._await_async_source()
        self.state = None
        self.start()

    @torch.no_grad()
    def set_target_source_schedule(self, chunks: Sequence[Sequence[int]]) -> None:
        """Set per-audio-chunk oracle source increments for an ablation run."""
        if self.target_source_mode != "oracle":
            raise RuntimeError("oracle source schedule requires target_source_mode=oracle")
        self._oracle_source_chunks = [list(map(int, chunk)) for chunk in chunks]
        self._oracle_source_index = 0

    def _require_state(self) -> NativeTrueDualState:
        if self.state is None:
            raise RuntimeError("call start() before decoding")
        return self.state

    @property
    def source_ids(self) -> list[int]:
        return [] if self.state is None else self.state.source_ids

    @property
    def target_ids(self) -> list[int]:
        return [] if self.state is None else self.state.target_ids

    @property
    def async_source_pending(self) -> bool:
        """Whether a deferred source result still needs foreground adoption."""
        return self._source_future is not None

    @property
    def async_source_ready(self) -> bool:
        """Whether the deferred worker is absent or has completed its forward."""
        return self._source_future is None or self._source_future.done()

    @torch.no_grad()
    def _append_ids(self, values: Sequence[int], *, target: bool) -> Tensor:
        values = [int(value) for value in values]
        if not values:
            raise ValueError("cannot append an empty token sequence")
        ids = torch.tensor([values], dtype=torch.long, device=self._device())
        return self._forward(target=target, input_ids=ids)

    @torch.no_grad()
    def _append_latent(self, latent: Tensor, *, target: bool) -> Tensor:
        if latent.ndim == 3:
            latent = latent.squeeze(0)
        if latent.ndim != 2:
            raise ValueError(f"latent must be [frames, hidden], got {tuple(latent.shape)}")
        ids = [
            self.audio_start_id,
            *([self.audio_pad_id] * latent.size(0)),
            self.audio_end_id,
        ]
        token_ids = torch.tensor([ids], dtype=torch.long, device=self._device())
        embedding = self.llm.get_input_embeddings()
        embeds = embedding(token_ids)
        embeds[:, 1 : 1 + latent.size(0)] = latent.to(
            device=self._device(), dtype=embeds.dtype
        ).unsqueeze(0)
        return self._forward(target=target, inputs_embeds=embeds)

    @torch.no_grad()
    def _append_audio_turn(self, latent: Tensor, *, target: bool) -> Tensor:
        logits = self._append_ids(self.user_open, target=target)
        logits = self._append_latent(latent, target=target)
        suffix = [*self.turn_close, *self.assistant_open, *self.source_prefix_ids]
        logits = self._append_ids(suffix, target=target)
        return logits

    @torch.no_grad()
    def _append_target_chat_turn(
        self, latent: Tensor, source_delta: Sequence[int]
    ) -> Tensor:
        """Open one standard target ChatML turn with audio and current source."""
        logits = self._append_ids(self.user_open, target=True)
        selected_latent = self._target_latent(latent)
        if selected_latent.size(0):
            logits = self._append_latent(selected_latent, target=True)
        values = [
            *self.target_user_instruction_ids,
            *[int(value) for value in source_delta],
            *self.turn_close,
            *self.assistant_open,
        ]
        return self._append_ids(values, target=True)

    @torch.no_grad()
    def _append_stage5_target_only_turn(self, latent: Tensor) -> Tensor:
        """Match Stage 5 training order: instruction, new audio, assistant."""
        logits = self._append_ids(
            [*self.user_open, *self.target_user_instruction_ids], target=True
        )
        selected_latent = self._target_latent(latent)
        if selected_latent.size(0):
            logits = self._append_latent(selected_latent, target=True)
        return self._append_ids(
            [*self.turn_close, *self.assistant_open], target=True
        )

    @torch.no_grad()
    def _decode_stage5_temporary_draft(
        self, latent: Tensor, *, is_final: bool
    ) -> list[int]:
        """Append audio once, then decode a replaceable target on a private Cache."""

        self.last_stage5_eot_trace = None
        state = self._require_state()
        if self.temporally_coupled_sampling and self.temperature > 0.0:
            if self.temporal_sampling_seed is None:
                raise RuntimeError("temporally coupled sampling requires a trajectory seed")
            # Common random numbers across rewrite ticks: token position k sees
            # the same uniform draw at every acoustic state. Each state's
            # categorical marginal is unchanged; only temporal exploration is
            # coupled so unchanged evidence does not create gratuitous rewrites.
            torch.manual_seed(int(self.temporal_sampling_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.temporal_sampling_seed))
        previous_draft = list(state.target_ids)
        selected_latent = self._target_latent(latent)
        if selected_latent.size(0):
            self._append_latent(selected_latent, target=True)
        persistent_evidence = state.target
        state.target = copy.deepcopy(persistent_evidence)
        try:
            branch_logits = self._append_ids(
                [
                    *self.tokenizer("\n", add_special_tokens=False)["input_ids"],
                    *self.turn_close,
                    *self.assistant_open,
                ],
                target=True,
            )
            if (
                self.target_speculative_draft_reuse
                and previous_draft
                and self.temperature <= 0.0
            ):
                draft_ids, closed = self._decode_stage5_speculative_draft(
                    branch_logits, previous_draft
                )
            else:
                draft_ids, closed = self._decode_branch_cached(
                    branch_logits, target=True, is_final=True
                )
            if not closed:
                self._append_ids([self.eos_id], target=True)
            private_draft = state.target
        except Exception:
            state.target = persistent_evidence
            raise
        if (
            not is_final
            and not previous_draft
            and self.target_early_content_margin is not None
            and len(draft_ids) < self.target_early_min_tokens
        ):
            draft_ids = []
        state.target = private_draft if is_final else persistent_evidence
        state.target_ids = list(draft_ids)
        return list(draft_ids)

    def _prepare_stage5_speculative_logits(
        self, logits: Tensor, generated: Sequence[int]
    ) -> Tensor:
        """Apply the exact Stage-5 greedy constraints while verifying a draft."""

        current = logits.clone()
        current[:, list(self.target_wait_ids)] = float("-inf")
        if self.target_forbidden_ids:
            current[:, list(self.target_forbidden_ids)] = float("-inf")
        branch = self._branch(True)
        if not branch.emitted and self.target_leading_forbidden_ids:
            current[:, list(self.target_leading_forbidden_ids)] = float("-inf")
        if self._target_should_force_content(current, generated):
            current[:, self.eos_id] = float("-inf")
        return self._mask_repeat(
            current,
            [*branch.emitted, *generated],
            self.target_no_repeat_ngram_size,
        )

    @torch.no_grad()
    def _decode_stage5_speculative_draft(
        self, branch_logits: Tensor, candidate: Sequence[int]
    ) -> tuple[list[int], bool]:
        """Verify the previous draft in parallel, then resume exact greedy decode."""

        state = self._require_state()
        candidate = [int(token) for token in candidate[: self.max_target_tokens]]
        if not candidate:
            return self._decode_branch_cached(
                branch_logits, target=True, is_final=True
            )

        verification_base = copy.deepcopy(state.target)
        candidate_ids = torch.tensor(
            [candidate], dtype=torch.long, device=self._device()
        )
        verification_logits = self._forward(
            target=True,
            input_ids=candidate_ids,
            return_all_logits=True,
        )
        accepted = 0
        for index, token in enumerate(candidate):
            predictor = (
                branch_logits
                if index == 0
                else verification_logits[:, index - 1]
            )
            predictor = self._prepare_stage5_speculative_logits(
                predictor, candidate[:index]
            )
            values, indices = predictor.topk(2, dim=-1)
            selected = int(indices[0, 0].item())
            margin = float((values[0, 0] - values[0, 1]).item())
            if selected != token or margin < self.target_speculative_min_margin:
                break
            accepted += 1

        self.speculative_verification_calls += 1
        self.speculative_candidate_tokens += len(candidate)
        self.speculative_accepted_tokens += accepted

        if accepted == len(candidate):
            current = verification_logits[:, -1]
        else:
            state.target = verification_base
            current = branch_logits
            if accepted:
                current = self._append_ids(candidate[:accepted], target=True)
        return self._decode_branch_cached(
            current,
            target=True,
            is_final=True,
            generated_prefix=candidate[:accepted],
        )

    @torch.no_grad()
    def _append_stage5_temporary_evidence(self, latent: Tensor) -> None:
        """Consume a chunk without regenerating the replaceable draft."""

        selected_latent = self._target_latent(latent)
        if selected_latent.size(0):
            self._append_latent(selected_latent, target=True)

    def _stage5_decode_due(self, *, is_final: bool) -> bool:
        if is_final:
            return True
        # Cadence reduction must not delay FRD. Keep trying at every acoustic
        # tick until the first non-empty draft has appeared.
        if not self._require_state().target_ids:
            return True
        chunk_number = self._require_state().chunks_seen
        if chunk_number <= self.target_decode_warmup_chunks:
            return True
        return (
            chunk_number - self.target_decode_warmup_chunks
        ) % self.target_decode_stride_chunks == 0

    def _target_latent(self, latent: Tensor) -> Tensor:
        """Apply an inference-only target condition ablation."""
        if self.target_latent_mode == "full":
            selected = latent
        elif self.target_latent_mode == "zero":
            selected = torch.zeros_like(latent)
        elif self.target_latent_mode == "shuffle":
            selected = latent if latent.size(0) <= 1 else torch.roll(latent, 1, dims=0)
        elif self.target_latent_mode == "empty":
            selected = latent.new_empty((0, latent.size(-1)))
        else:
            selected = (
                torch.zeros_like(latent)
                if self._previous_target_latent is None
                else self._previous_target_latent
            )
        self._previous_target_latent = latent.detach().clone()
        return selected

    @staticmethod
    def _suffix(values: Sequence[int], suffix: Sequence[int]) -> bool:
        return bool(suffix) and len(values) >= len(suffix) and list(
            values[-len(suffix) :]
        ) == list(suffix)

    def _mask_source(self, logits: Tensor) -> Tensor:
        if self.source_allowed_ids is None:
            return logits
        allowed = set(self.source_allowed_ids)
        allowed.update(self.source_wait_ids)
        allowed.add(self.eos_id)
        result = torch.full_like(logits, float("-inf"))
        indices = torch.tensor(sorted(allowed), dtype=torch.long, device=logits.device)
        result[:, indices] = logits[:, indices]
        return result

    def _select(self, logits: Tensor) -> int:
        if self.temperature <= 0.0:
            return int(logits.argmax(dim=-1).item())
        return int(
            torch.multinomial(
                torch.softmax(logits / self.temperature, dim=-1), num_samples=1
            ).item()
        )

    @staticmethod
    def _mask_repeat(logits: Tensor, history: Sequence[int], ngram_size: int) -> Tensor:
        """Block a repeated n-gram using only the already emitted branch text."""
        if ngram_size <= 1 or len(history) < ngram_size - 1:
            return logits
        prefix = list(history[-(ngram_size - 1) :])
        blocked: set[int] = set()
        for index in range(len(history) - ngram_size + 1):
            if list(history[index : index + ngram_size - 1]) == prefix:
                blocked.add(int(history[index + ngram_size - 1]))
        if not blocked:
            return logits
        result = logits.clone()
        result[:, list(blocked)] = float("-inf")
        return result

    def _adaptive_wait_is_safe(self, logits: Tensor) -> bool:
        """Use a small wait bias only when a content token is already plausible."""
        if not self.target_adaptive_wait_bias:
            return False
        wait_score = logits[:, list(self.wait_ids)].max(dim=-1).values
        content = logits.clone()
        content[:, list(self.wait_ids)] = float("-inf")
        content_score = content.max(dim=-1).values
        margin = float((content_score - wait_score).item())
        return margin >= self.target_adaptive_wait_margin

    @torch.no_grad()
    def _decode_branch_cached(
        self,
        logits: Tensor,
        *,
        target: bool,
        is_final: bool,
        force_first_nonwait: bool = False,
        source_initial_lexical_only: bool = False,
        source_wait_logit_bias: float = 0.0,
        generated_prefix: Optional[Sequence[int]] = None,
    ) -> tuple[list[int], bool]:
        branch = self._branch(target)
        history = list(branch.emitted)
        ngram_size = (
            self.target_no_repeat_ngram_size
            if target
            else self.source_no_repeat_ngram_size
        )
        limit = self.max_target_tokens if target else self.max_source_tokens
        generated = [int(token) for token in (generated_prefix or ())]
        closed = False
        current = logits
        wait_ids = self.target_wait_ids if target else self.source_wait_ids
        for _ in range(max(0, limit - len(generated))):
            # Keep the unconstrained model state for the optional RCE audit.
            # ``current`` is replaced by a clone immediately below, so this
            # detached view remains the raw next-token distribution.
            raw_current = current.detach().float()
            current = current.clone()
            if not is_final:
                current[:, self.eos_id] = float("-inf")
            elif target:
                # Once the input has ended, <wait> cannot be fulfilled by a
                # future audio chunk. Force the target branch to finish with
                # content or EOS instead of silently truncating the tail.
                current[:, list(wait_ids)] = float("-inf")
            if not target:
                current = self._mask_source(current)
                if source_initial_lexical_only and not generated:
                    if not self.source_preview_initial_ids:
                        raise ValueError("source preview has no lexical initial tokens")
                    initial_ids = [*self.source_preview_initial_ids, *wait_ids]
                    lexical = current[:, initial_ids].clone()
                    current.fill_(float("-inf"))
                    current[:, initial_ids] = lexical
                    if source_wait_logit_bias:
                        current[:, list(wait_ids)] += float(source_wait_logit_bias)
                if force_first_nonwait and not generated:
                    current[:, list(wait_ids)] = float("-inf")
            elif self.target_forbidden_ids:
                current[:, list(self.target_forbidden_ids)] = float("-inf")
            if (
                target and not branch.emitted and self.target_leading_forbidden_ids
            ):
                current[:, list(self.target_leading_forbidden_ids)] = float("-inf")
            if target and not history and self._target_should_force_content(
                current, generated
            ):
                current[:, self.eos_id] = float("-inf")
            current = self._mask_repeat(current, [*history, *generated], ngram_size)
            apply_wait_bias = target and bool(self.target_wait_logit_bias)
            if apply_wait_bias and self.target_wait_bias_after_waits:
                apply_wait_bias = (
                    self._target_wait_streak >= self.target_wait_bias_after_waits
                    and not generated
                )
            elif apply_wait_bias and self.target_wait_bias_before_first_output:
                apply_wait_bias = not branch.emitted and not generated
            elif apply_wait_bias and self.target_wait_bias_first_token_only:
                apply_wait_bias = not generated
            if apply_wait_bias:
                current[:, wait_ids] += self.target_wait_logit_bias
            if (
                target
                and not generated
                and not branch.emitted
                and self._adaptive_wait_is_safe(current)
            ):
                current[:, wait_ids] += self.target_adaptive_wait_bias
            source_candidates = None
            if not target and self.capture_source_topk:
                count = min(self.source_topk, int(current.size(-1)))
                values, indices = current.float().topk(count, dim=-1)
                probabilities = torch.softmax(values, dim=-1)
                source_candidates = (
                    indices[0].detach().cpu().tolist(),
                    probabilities[0].detach().cpu().tolist(),
                )
            token = self._select(current)
            can_wait = not (target and is_final) and (
                len(generated) >= self.target_min_tokens_before_wait
                and (
                    not target
                    or self._require_state().chunks_seen
                    >= self.target_min_chunks_before_wait
                )
            )
            if target and token in wait_ids and not can_wait:
                current[:, wait_ids] = float("-inf")
                token = self._select(current)
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
                self._append_ids([token, self.eos_id], target=False)
                generated.append(token)
                closed = True
                break
            if not target and source_candidates is not None:
                candidate_ids, candidate_probabilities = source_candidates
                self.source_topk_trace.append(
                    {
                        "position": len(history) + len(generated),
                        "prefix_text": self.tokenizer.decode(
                            [*history, *generated],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        ),
                        "selected_id": int(token),
                        "selected_text": self.tokenizer.decode(
                            [int(token)],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ),
                        "candidate_ids": candidate_ids,
                        "candidate_texts": [
                            self.tokenizer.decode(
                                [int(value)],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )
                            for value in candidate_ids
                        ],
                        "candidate_probabilities": candidate_probabilities,
                    }
                )
            if (
                target
                and token == self.eos_id
                and os.environ.get("S2TT_CAPTURE_ROLLOUT_EOT_LOGITS") == "1"
            ):
                count = min(64, int(raw_current.size(-1)))
                raw_values, raw_indices = raw_current[0].topk(count)
                effective_values, effective_indices = current.float()[0].topk(count)
                self.last_stage5_eot_trace = {
                    "protocol": "rollout-context-equivalence-r3761-v1",
                    "prefix_token_ids": [int(value) for value in [*history, *generated]],
                    "eot_id": int(self.eos_id),
                    "raw_eot_logit": float(raw_current[0, self.eos_id].item()),
                    "raw_logsumexp": float(torch.logsumexp(raw_current[0], dim=-1).item()),
                    "raw_top64_ids": [int(value) for value in raw_indices.cpu().tolist()],
                    "raw_top64_logits": [float(value) for value in raw_values.cpu().tolist()],
                    "effective_eot_logit": float(current[0, self.eos_id].item()),
                    "effective_logsumexp": float(torch.logsumexp(current.float()[0], dim=-1).item()),
                    "effective_top64_ids": [int(value) for value in effective_indices.cpu().tolist()],
                    "effective_top64_logits": [float(value) for value in effective_values.cpu().tolist()],
                }
            current = self._append_ids([token], target=target)
            if token == self.eos_id:
                closed = True
                break
            generated.append(token)
            if can_wait and self._suffix(generated, wait_ids):
                del generated[-len(wait_ids) :]
                closed = True
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
            self._append_ids([token, self.eos_id], target=False)
            generated.append(token)
            closed = True
        branch.emitted.extend(generated)
        if target:
            if generated:
                self._target_wait_streak = 0
            elif closed:
                self._target_wait_streak += 1
        return generated, closed

    def _target_should_force_content(
        self, logits: Tensor, generated: Sequence[int]
    ) -> bool:
        if generated:
            return False
        if self.target_force_token_if_empty:
            return True
        if self.target_early_content_margin is None:
            return False
        eos_score = logits[:, self.eos_id]
        content = logits.clone()
        content[:, self.eos_id] = float("-inf")
        content[:, list(self.target_wait_ids)] = float("-inf")
        content_score = content.max(dim=-1).values
        return float((content_score - eos_score).item()) >= self.target_early_content_margin

    @torch.no_grad()
    def preview_source(
        self, latent: Tensor, *, force_first_nonwait: bool = True
    ) -> list[int]:
        """Decode a disposable source draft without changing persistent state."""

        state = self._require_state()
        persistent_source = state.source
        persistent_source_ids = list(state.source_ids)
        try:
            state.source = copy.deepcopy(persistent_source)
            source_logits = self._append_audio_turn(latent, target=False)
            source_delta, _ = self._decode_branch_cached(
                source_logits,
                target=False,
                is_final=False,
                force_first_nonwait=force_first_nonwait,
            )
            return source_delta
        finally:
            state.source = persistent_source
            state.source_ids = persistent_source_ids

    @torch.no_grad()
    def _commit_source(
        self,
        source_delta: Sequence[int],
        *,
        inline_current_turn: bool = False,
    ) -> None:
        """Publish one source increment to the target-only cache.

        With source lag zero, ``inline_current_turn`` keeps the native Qwen
        order audio_t -> source_t -> target_t.  With source lag one, the
        explicit context marker separates source_t from the completed target
        turn and makes it visible to target_t+1.  The source cache is never
        modified here.
        """
        if self.target_source_mode == "none":
            return
        if self.target_source_mode == "oracle":
            if self._oracle_source_chunks is None:
                raise RuntimeError("oracle source schedule was not set")
            if self._oracle_source_index < len(self._oracle_source_chunks):
                visible_source = self._oracle_source_chunks[self._oracle_source_index]
            else:
                visible_source = []
            self._oracle_source_index += 1
        else:
            visible_source = source_delta
        values = [*visible_source, *self.source_wait_ids]
        if not inline_current_turn:
            values = [*self.source_context_ids, *values]
        self._append_ids(values, target=True)

    @torch.no_grad()
    def _run_source(
        self,
        latent: Tensor,
        *,
        is_final: bool,
        publish_target: bool = True,
        record_source: bool = True,
    ) -> list[int]:
        """Decode and publish one source increment to the target cache."""
        if not self.source_decode:
            if self.target_source_mode == "oracle" and publish_target:
                # RL sampling can provide a source-prefix schedule without
                # spending decoder steps on regenerating the source stream.
                self._commit_source([])
            elif self.target_source_mode != "none":
                raise RuntimeError(
                    "source_decode=False requires target_source_mode=none or oracle"
                )
            return []
        source_logits = self._append_audio_turn(latent, target=False)
        source_delta, _ = self._decode_branch_cached(
            source_logits, target=False, is_final=is_final
        )
        if record_source:
            self.state.source_ids.extend(source_delta)
        if publish_target:
            self._commit_source(source_delta)
        return source_delta

    @torch.no_grad()
    def _flush_queued_source_context(self, *, force: bool = False) -> None:
        """Publish delayed source chunks without running source decoding."""
        if self.target_source_mode == "none":
            self._queued_source_deltas.clear()
            return
        threshold = max(1, self.target_source_lag_chunks)
        while self._queued_source_deltas and (
            force or len(self._queued_source_deltas) >= threshold
        ):
            self._commit_source(self._queued_source_deltas.pop(0))

    @torch.no_grad()
    def _flush_pending_source(self) -> list[int]:
        """Commit the previous source chunk before advancing target time."""
        if self._pending_source is None:
            return []
        latent, source_latent, is_final = self._pending_source
        self._pending_source = None
        source_input = latent if source_latent is None else source_latent
        return self._run_source(source_input, is_final=is_final)

    @torch.inference_mode()
    def _run_async_source(
        self, latent: Tensor, source_latent: Optional[Tensor], is_final: bool
    ) -> list[int]:
        """Run only the deferred source branch in the worker.

        The worker must not mutate the target cache. The next foreground tick
        adopts its returned source delta before target decoding, so a target
        cache can never be modified halfway through a foreground decode.
        CUDA synchronization is required because the worker has its own PyTorch
        thread/stream and the next foreground target forward must see a
        completed result.
        """
        source_input = latent if source_latent is None else source_latent
        delta = self._run_source(
            source_input,
            is_final=is_final,
            publish_target=False,
            record_source=False,
        )
        device = self._device()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return delta

    def _await_async_source(self) -> list[int]:
        if self._source_future is None:
            return []
        future = self._source_future
        self._source_future = None
        delta = list(future.result())
        if delta:
            # Commit source history only in the foreground, before the next
            # target tick. This keeps target-cache writes atomic with respect
            # to target decoding even when source computation was asynchronous.
            self._commit_source(delta)
            self.state.source_ids.extend(delta)
        return delta

    def _submit_async_source(
        self, latent: Tensor, source_latent: Optional[Tensor], is_final: bool
    ) -> None:
        if self._source_future is not None:
            raise RuntimeError("a deferred source branch is already running")
        if self._source_executor is None:
            self._source_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="s2tt-source"
            )
        self._source_future = self._source_executor.submit(
            self._run_async_source,
            latent.detach(),
            None if source_latent is None else source_latent.detach(),
            is_final,
        )

    def _schedule_source(
        self,
        latent: Tensor,
        *,
        source_latent: Optional[Tensor],
        is_final: bool,
    ) -> list[int]:
        if self.defer_source and not is_final:
            self._pending_source = (
                latent.detach(),
                None if source_latent is None else source_latent.detach(),
                is_final,
            )
            return []
        source_input = latent if source_latent is None else source_latent
        return self._run_source(source_input, is_final=is_final)

    @torch.no_grad()
    def accept(
        self,
        latent: Tensor,
        is_final: bool = False,
        source_latent: Optional[Tensor] = None,
        update_source: bool = True,
    ) -> tuple[list[int], list[int], bool]:
        state = self._require_state()
        if state.finished:
            return [], [], True
        if latent.ndim == 3:
            latent = latent.squeeze(0)
        if self.target_prompt_mode == "stage5_temporary_draft_chat":
            chunk_index = state.chunks_seen
            state.acoustic.chunk_ids.append(chunk_index)
            state.acoustic.frame_counts.append(int(latent.size(0)))
            if self._stage5_decode_due(is_final=is_final):
                target_draft = self._decode_stage5_temporary_draft(
                    latent, is_final=is_final
                )
            else:
                self._append_stage5_temporary_evidence(latent)
                target_draft = []
            state.finished = bool(is_final)
            return [], target_draft, state.finished
        source_delta: list[int] = (
            self._await_async_source()
            if self.async_source_pipeline
            else self._flush_pending_source()
        )
        if self.target_source_lag_chunks >= 2:
            self._flush_queued_source_context()
        chunk_index = state.chunks_seen
        state.acoustic.chunk_ids.append(chunk_index)
        state.acoustic.frame_counts.append(int(latent.size(0)))

        source_closed = False
        source_ready_before_target = False
        if self.target_source_lag_chunks == 0 and update_source:
            source_input = latent if source_latent is None else source_latent
            source_delta = self._run_source(
                source_input,
                is_final=is_final,
                publish_target=False,
            )
            source_ready_before_target = True
        elif (
            self.target_source_lag_chunks == 1
            and self.target_source_first_chunk_current
            and chunk_index == 0
            and update_source
        ):
            source_input = latent if source_latent is None else source_latent
            source_delta = self._run_source(
                source_input,
                is_final=is_final,
                publish_target=False,
            )
            source_ready_before_target = True

        chat_current_source = (
            self.target_prompt_mode == "chat_translation"
            and self.target_source_lag_chunks == 0
            and source_ready_before_target
        )
        if self.target_prompt_mode == "chat_translation" and not chat_current_source:
            raise RuntimeError(
                "chat_translation requires generated current source with target_source_lag_chunks=0"
            )
        target_only_chat = self.target_prompt_mode == "stage5_target_only_chat"
        if target_only_chat:
            target_logits = self._append_stage5_target_only_turn(latent)
        elif chat_current_source:
            target_logits = self._append_target_chat_turn(latent, source_delta)
        else:
            # The legacy native-marker route keeps the Qwen-ASR source marker
            # in the target cache for backward-compatible checkpoints.
            target_logits = self._append_audio_turn(
                self._target_latent(latent), target=True
            )
            if source_ready_before_target:
                self._commit_source(source_delta, inline_current_turn=True)
        if chunk_index >= self.target_output_lag_chunks:
            if not chat_current_source and not target_only_chat:
                target_logits = self._append_ids(self.target_marker_ids, target=True)
            target_delta, target_closed = self._decode_branch_cached(
                target_logits,
                target=True,
                is_final=(target_only_chat or (is_final and chat_current_source)),
            )
            if target_only_chat:
                if not target_closed:
                    self._append_ids([self.eos_id], target=True)
                self._append_ids(self.tokenizer("\n", add_special_tokens=False)["input_ids"], target=True)
            elif chat_current_source and not is_final:
                self._append_ids(self.turn_close, target=True)
        else:
            # Keep the target turn open.  The first target increment is
            # emitted on the next chunk, after one source context is committed.
            target_delta, target_closed = [], False

        if self.target_source_lag_chunks == 1 and update_source and not source_ready_before_target:
            # Source may use the complete frontend latent even when target is
            # advanced on finer decoder ticks.  The branches still keep
            # independent caches and share no generated text.
            if self.async_source_pipeline and not is_final:
                self._submit_async_source(latent, source_latent, is_final=False)
            else:
                source_delta.extend(
                    self._schedule_source(
                        latent,
                        source_latent=source_latent,
                        is_final=is_final,
                    )
                )
        elif self.target_source_lag_chunks >= 2 and update_source:
            source_input = latent if source_latent is None else source_latent
            current_source = self._run_source(
                source_input,
                is_final=is_final,
                publish_target=False,
            )
            source_delta.extend(current_source)
            self._queued_source_deltas.append(list(current_source))

        if is_final and not chat_current_source and not target_only_chat:
            # Allow the target branch to use the final committed source during
            # the final flush, while keeping that information out of source.
            if self.target_source_lag_chunks >= 2:
                self._flush_queued_source_context(force=True)
            target_logits = self._append_ids(self.target_marker_ids, target=True)
            final_delta, final_closed = self._decode_branch_cached(
                target_logits, target=True, is_final=True
            )
            target_delta.extend(final_delta)
        if is_final:
            state.finished = True
        state.target_ids.extend(target_delta)
        return source_delta, target_delta, state.finished

    @torch.no_grad()
    def accept_final_without_latent(self) -> tuple[list[int], list[int], bool]:
        state = self._require_state()
        if state.finished:
            return [], [], True
        source_delta = (
            self._await_async_source()
            if self.async_source_pipeline
            else self._flush_pending_source()
        )
        if self.target_source_lag_chunks >= 2:
            self._flush_queued_source_context(force=True)
        if self.target_prompt_mode == "stage5_temporary_draft_chat":
            hidden_size = int(self.llm.get_input_embeddings().weight.size(1))
            empty_latent = torch.empty(
                0,
                hidden_size,
                dtype=self.llm.get_input_embeddings().weight.dtype,
                device=self._device(),
            )
            target_draft = self._decode_stage5_temporary_draft(
                empty_latent, is_final=True
            )
            state.finished = True
            return [], target_draft, True
        if self.target_prompt_mode in {"chat_translation", "stage5_target_only_chat"}:
            hidden_size = int(self.llm.get_input_embeddings().weight.size(1))
            empty_latent = torch.empty(
                0,
                hidden_size,
                dtype=self.llm.get_input_embeddings().weight.dtype,
                device=self._device(),
            )
            if self.target_prompt_mode == "stage5_target_only_chat":
                target_logits = self._append_stage5_target_only_turn(empty_latent)
            else:
                target_logits = self._append_target_chat_turn(empty_latent, source_delta)
        else:
            target_logits = self._append_ids(self.target_marker_ids, target=True)
        target_delta, _ = self._decode_branch_cached(
            target_logits, target=True, is_final=True
        )
        state.target_ids.extend(target_delta)
        state.finished = True
        return source_delta, target_delta, True

    @torch.no_grad()
    def accept_source_only(
        self, latent: Tensor, is_final: bool = False
    ) -> tuple[list[int], bool]:
        """Consume audio on the source branch without advancing target text."""
        if not self.source_decode:
            raise RuntimeError("source-only decoding is unavailable in target-only mode")
        state = self._require_state()
        if state.finished:
            return [], True
        if latent.ndim == 3:
            latent = latent.squeeze(0)
        state.acoustic.chunk_ids.append(state.chunks_seen)
        state.acoustic.frame_counts.append(int(latent.size(0)))
        logits = self._append_audio_turn(latent, target=False)
        source_delta, _ = self._decode_branch_cached(
            logits, target=False, is_final=is_final
        )
        state.source_ids.extend(source_delta)
        if is_final:
            state.finished = True
        return source_delta, state.finished

    @torch.no_grad()
    def redecode_source_unit(self, latents: Sequence[Tensor]) -> str:
        """Decode one bounded source unit from a fresh source-route cache."""

        if not latents:
            return ""
        previous_state = self.state
        previous_trace = self.source_topk_trace
        previous_pending = self._pending_source
        previous_queued = self._queued_source_deltas
        previous_future = self._source_future
        if previous_future is not None:
            raise RuntimeError("cannot redecode a source unit during async source work")
        try:
            self.state = None
            self.source_topk_trace = []
            self._pending_source = None
            self._queued_source_deltas = []
            self.start()
            for index, latent in enumerate(latents):
                self.accept_source_only(
                    latent,
                    is_final=index + 1 == len(latents),
                )
            return self.decode(self.source_ids)
        finally:
            self.state = previous_state
            self.source_topk_trace = previous_trace
            self._pending_source = previous_pending
            self._queued_source_deltas = previous_queued

    @torch.no_grad()
    def accept_source_only_with_preview(
        self,
        latent: Tensor,
        is_final: bool = False,
        preview_wait_logit_bias: float = 0.0,
    ) -> tuple[list[int], list[int], bool]:
        """Consume one chunk and force a disposable draft only when source waits."""

        state = self._require_state()
        if state.finished:
            return [], [], True
        if latent.ndim == 3:
            latent = latent.squeeze(0)
        state.acoustic.chunk_ids.append(state.chunks_seen)
        state.acoustic.frame_counts.append(int(latent.size(0)))
        logits = self._append_audio_turn(latent, target=False)
        branch_before_decision = copy.deepcopy(state.source)
        source_delta, _ = self._decode_branch_cached(
            logits, target=False, is_final=is_final
        )
        persistent_source = state.source
        preview_delta: list[int] = []
        if not source_delta and not is_final:
            try:
                state.source = branch_before_decision
                preview_delta, _ = self._decode_branch_cached(
                    logits,
                    target=False,
                    is_final=False,
                    source_initial_lexical_only=True,
                    source_wait_logit_bias=preview_wait_logit_bias,
                )
            finally:
                state.source = persistent_source
        state.source_ids.extend(source_delta)
        if is_final:
            state.finished = True
        return source_delta, preview_delta, state.finished

    def decode(self, values: Iterable[int]) -> str:
        return self.tokenizer.decode(
            list(values), skip_special_tokens=True, clean_up_tokenization_spaces=True
        ).strip()


def build_native_true_dual_target_training_sequence(
    llm,
    tokenizer,
    soft_chunks: Sequence[Tensor],
    source_chunks: Sequence[Sequence[int]],
    target_chunks: Sequence[Sequence[int]],
    wait_id,
    eos_id: int,
    target_loss_weight: float = 1.0,
    boundary_loss_weight: float = 3.0,
    audio_start_id: int = 151669,
    audio_end_id: int = 151670,
    audio_pad_id: int = 151676,
    source_prefix_ids: Optional[Sequence[int]] = None,
    target_marker_ids: Optional[Sequence[int]] = None,
    source_context_ids: Optional[Sequence[int]] = None,
    target_system_instruction: str = "",
    target_prompt_mode: str = "native_asr_marker",
    target_user_instruction: str = (
        "Translate and commit this complete Chinese unit into English:\n"
    ),
    source_wait_id=None,
    target_output_lag_chunks: int = 1,
    target_source_lag_chunks: int = 1,
    target_source_first_chunk_current: bool = False,
    target_source_mode: str = "generated",
) -> tuple[Tensor, Tensor, Tensor]:
    """Build target supervision with zero- or one-chunk output lag."""
    if target_output_lag_chunks not in (0, 1):
        raise ValueError("target_output_lag_chunks must be 0 or 1")
    if target_source_lag_chunks not in (0, 1):
        raise ValueError("target_source_lag_chunks must be 0 or 1")
    if target_source_mode not in {"generated", "none"}:
        raise ValueError("target_source_mode must be generated or none")
    if target_prompt_mode not in {"native_asr_marker", "chat_translation"}:
        raise ValueError(f"unsupported target_prompt_mode: {target_prompt_mode}")
    if target_prompt_mode == "chat_translation" and (
        target_output_lag_chunks != 0
        or target_source_lag_chunks != 0
        or target_source_mode != "generated"
    ):
        raise ValueError(
            "chat_translation requires target_output_lag_chunks=0, "
            "target_source_lag_chunks=0 and target_source_mode=generated"
        )
    if not (len(soft_chunks) == len(source_chunks) == len(target_chunks)):
        raise ValueError("true dual target chunks must have equal lengths")
    if not soft_chunks:
        raise ValueError("true dual target sequence needs at least one chunk")
    device = soft_chunks[0].device
    embedding = llm.get_input_embeddings()
    wait_ids = _wait_ids(wait_id)
    source_wait_ids = _wait_ids(wait_id if source_wait_id is None else source_wait_id)
    user_open, assistant_open, turn_close = turn_prefix_ids(tokenizer)
    source_prefix_ids = list(
        source_prefix_ids or _ids(tokenizer, "language Chinese<asr_text>")
    )
    target_marker_ids = list(target_marker_ids or _ids(tokenizer, "\n<translation>\n"))
    source_context_ids = list(source_context_ids or _ids(tokenizer, "\n<asr_text>"))
    target_user_instruction_ids = _ids(tokenizer, target_user_instruction)
    embeds = [
        embedding(
            native_system_prompt(tokenizer, target_system_instruction).to(device)
        )
    ]
    labels = [torch.full((embeds[0].size(0),), -100, dtype=torch.long, device=device)]
    weights = [torch.zeros(embeds[0].size(0), dtype=torch.float32, device=device)]

    for index, (soft, source, target) in enumerate(
        zip(soft_chunks, source_chunks, target_chunks)
    ):
        if soft.ndim == 3:
            soft = soft.squeeze(0)
        source_current = target_source_mode == "generated" and (
            target_source_lag_chunks == 0
            or (target_source_first_chunk_current and index == 0)
        )
        _append_ids(embeds, labels, weights, embedding, user_open, device)
        if target_prompt_mode != "chat_translation" or soft.size(0):
            latent_ids = [audio_start_id, *([audio_pad_id] * soft.size(0)), audio_end_id]
            token_ids = torch.tensor(latent_ids, dtype=torch.long, device=device)
            latent_embeds = embedding(token_ids)
            latent_embeds[1 : 1 + soft.size(0)] = soft.to(latent_embeds.dtype)
            embeds.append(latent_embeds)
            labels.append(
                torch.full(
                    (latent_embeds.size(0),), -100, dtype=torch.long, device=device
                )
            )
            weights.append(
                torch.zeros(
                    latent_embeds.size(0), dtype=torch.float32, device=device
                )
            )
        if target_prompt_mode == "chat_translation":
            _append_ids(
                embeds,
                labels,
                weights,
                embedding,
                [
                    *target_user_instruction_ids,
                    *_clean(source, source_wait_ids, eos_id),
                    *turn_close,
                    *assistant_open,
                ],
                device,
            )
            current_target = _clean(target, wait_ids, eos_id)
            is_final = index == len(target_chunks) - 1
            _append_supervised(
                embeds,
                labels,
                weights,
                embedding,
                [*current_target, *([eos_id] if is_final else wait_ids)],
                device,
                target_loss_weight,
                boundary_loss_weight,
            )
            if not is_final:
                _append_ids(
                    embeds,
                    labels,
                    weights,
                    embedding,
                    turn_close,
                    device,
                )
            continue

        _append_ids(
            embeds,
            labels,
            weights,
            embedding,
            [*turn_close, *assistant_open, *source_prefix_ids],
            device,
        )
        if source_current:
            # Source decoding happens before target decoding, but the target
            # cache preserves Qwen's native audio_i -> source_i -> target_i
            # token order.
            _append_ids(
                embeds,
                labels,
                weights,
                embedding,
                [*_clean(source, source_wait_ids, eos_id), *source_wait_ids],
                device,
            )
        if index >= target_output_lag_chunks:
            target_index = index - target_output_lag_chunks
            _append_ids(
                embeds,
                labels,
                weights,
                embedding,
                target_marker_ids,
                device,
            )
            delayed_target = _clean(target_chunks[target_index], wait_ids, eos_id)
            _append_supervised(
                embeds,
                labels,
                weights,
                embedding,
                [*delayed_target, *wait_ids],
                device,
                target_loss_weight,
                boundary_loss_weight,
            )
        if (
            target_source_mode == "generated"
            and target_source_lag_chunks == 1
            and not (target_source_first_chunk_current and index == 0)
        ):
            # Source_i is appended after the target tick.  It becomes visible
            # only at the next target tick, exactly matching the runtime route.
            _append_ids(
                embeds,
                labels,
                weights,
                embedding,
                [
                    *source_context_ids,
                    *_clean(source, source_wait_ids, eos_id),
                    *source_wait_ids,
                ],
                device,
            )

    if target_prompt_mode == "chat_translation":
        return (
            torch.cat(embeds, dim=0).unsqueeze(0),
            torch.cat(labels, dim=0).unsqueeze(0),
            torch.cat(weights, dim=0).unsqueeze(0),
        )

    _append_ids(embeds, labels, weights, embedding, target_marker_ids, device)
    if target_output_lag_chunks == 1:
        final_target = _clean(target_chunks[-1], wait_ids, eos_id)
        _append_supervised(
            embeds,
            labels,
            weights,
            embedding,
            [*final_target, eos_id],
            device,
            target_loss_weight,
            boundary_loss_weight,
        )
    else:
        _append_supervised(
            embeds,
            labels,
            weights,
            embedding,
            [eos_id],
            device,
            target_loss_weight,
            boundary_loss_weight,
        )
    return (
        torch.cat(embeds, dim=0).unsqueeze(0),
        torch.cat(labels, dim=0).unsqueeze(0),
        torch.cat(weights, dim=0).unsqueeze(0),
    )
