"""Realtime PCM frontend and persistent-cache decoder for the Qwen-ASR route."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional
import unicodedata

import numpy as np
import torch

from .dual_stream import DualStreamDecoder
from .native_dual_stream import NativePersistentQwenDecoder, source_text_token_ids
from .native_true_dual_stream import NativeTrueDualQwenDecoder
from .qformer import (
    ChunkQFormer,
    DirectChunkAdapter,
)
from .qwen_lora import filter_adapter_state, resolve_lora_targets
from .torchscript_encoder import TorchscriptStreamingZipformer


MINIMUM_AUDIO_TOWER_SAMPLES = 400


def cjk_token_ids(tokenizer) -> set[int]:
    """Return vocabulary entries containing CJK letters.

    This is a target-language invariant for English decoding, not a
    reference-dependent repair. Mixed-script pieces are blocked as well.
    """

    special_ids = set(getattr(tokenizer, "all_special_ids", ()) or ())
    blocked: set[int] = set()
    for token_id in range(len(tokenizer)):
        if token_id in special_ids:
            continue
        text = tokenizer.decode(
            [token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if any(
            unicodedata.name(character, "").startswith(
                ("CJK", "HIRAGANA", "KATAKANA", "HANGUL")
            )
            for character in text
        ):
            blocked.add(token_id)
    return blocked


def pad_short_audio_tower_waveform(waveform: np.ndarray) -> np.ndarray:
    """Right-pad sub-STFT tails without changing their accounted duration."""
    if waveform.size >= MINIMUM_AUDIO_TOWER_SAMPLES:
        return waveform
    return np.pad(
        waveform,
        (0, MINIMUM_AUDIO_TOWER_SAMPLES - waveform.size),
        mode="constant",
    )


@dataclass
class StreamingEvent:
    audio_time: float
    source_delta: str
    source_prefix: str
    target_delta: str
    target_prefix: str
    finished: bool
    cache_attention_length: int
    source_cache_attention_length: int = 0
    target_cache_attention_length: int = 0
    target_delta_token_ids: list[int] | None = None
    target_token_ids: list[int] | None = None
    target_eot_trace: dict | None = None


class QwenASRContinuousFrontend:
    """Turn streaming PCM into prefix-aligned continuous audio-tower deltas.

    Qwen-ASR's cached training features were extracted from the growing audio
    prefix and then sliced to the newly available frames. The audio tower
    therefore retains a prefix for feature extraction, while the decoder sees
    each new latent frame exactly once.
    """

    def __init__(
        self,
        runtime,
        thinker=None,
        chunk_seconds: float = 3.0,
        chunk_local: bool = False,
        context_seconds: float = 0.0,
        right_context_seconds: float = 0.0,
        compile_audio_tower: bool = False,
        device: str = "cuda",
    ) -> None:
        self.runtime = runtime
        self.device = torch.device(device)
        self.audio_tower = (thinker or runtime.model.thinker).audio_tower.to(self.device).eval()
        if compile_audio_tower and hasattr(torch, "compile"):
            # Keep the compiled wrapper outside the nn.Module tree. Storing a
            # compiled module on audio_tower itself registers it as a child
            # module and makes later reset().to(...) recurse indefinitely.
            compiled = getattr(runtime, "_s2tt_compiled_audio_tower", None)
            if compiled is None:
                compiled = torch.compile(
                    self.audio_tower,
                    mode="reduce-overhead",
                    dynamic=True,
                    fullgraph=False,
                )
                setattr(runtime, "_s2tt_compiled_audio_tower", compiled)
            self.audio_tower = compiled
        self.chunk_samples = max(1600, int(round(chunk_seconds * 16000)))
        self.buffer = bytearray()
        self.prefix_pcm = bytearray()
        self.processed_samples = 0
        self.previous_prefix_frames = 0
        self.chunk_local = bool(chunk_local)
        self.context_samples = max(0, int(round(context_seconds * 16000)))
        self.right_context_samples = max(
            0, int(round(right_context_seconds * 16000))
        )
        if self.right_context_samples and not self.chunk_local:
            raise ValueError("right context requires chunk_local frontend mode")
        self.context_pcm = bytearray()
        self.current_audio_samples = 0

    @torch.no_grad()
    def _encode_pcm(self, pcm: bytes) -> torch.Tensor:
        usable = len(pcm) - (len(pcm) % 2)
        waveform = np.frombuffer(pcm[:usable], dtype=np.int16).astype(np.float32) / 32768.0
        waveform = pad_short_audio_tower_waveform(waveform)
        inputs = self.runtime.processor(
            text=[""],
            audio=[waveform],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        mel = inputs["input_features"][0]
        length = int(inputs["feature_attention_mask"][0].sum().item())
        length = max(1, min(length, mel.size(-1)))
        output = self.audio_tower(
            mel[:, :length].to(device=self.device, dtype=torch.bfloat16),
            feature_lens=torch.tensor([length], device=self.device, dtype=torch.long),
        )
        return output.last_hidden_state.to(dtype=torch.bfloat16)

    @torch.no_grad()
    def _encode_prefix_delta(self) -> torch.Tensor:
        features = self._encode_pcm(bytes(self.prefix_pcm))
        total_frames = int(features.size(0))
        previous = min(self.previous_prefix_frames, max(0, total_frames - 1))
        delta = features[previous:]
        self.previous_prefix_frames = total_frames
        return delta

    @torch.no_grad()
    def accept(self, pcm: bytes, final: bool = False) -> list[torch.Tensor]:
        self.buffer.extend(pcm)
        chunks: list[torch.Tensor] = []
        bytes_per_chunk = self.chunk_samples * 2
        if self.chunk_local:
            if self.right_context_samples:
                right_bytes = self.right_context_samples * 2
                while len(self.buffer) >= bytes_per_chunk + right_bytes:
                    current = bytes(self.buffer[:bytes_per_chunk])
                    window_raw = bytes(self.buffer[: bytes_per_chunk + right_bytes])
                    del self.buffer[:bytes_per_chunk]
                    history = bytes(self.context_pcm[-self.context_samples * 2 :])
                    window = history + window_raw
                    hidden = self._encode_pcm(window)
                    start_frame = round(
                        hidden.size(0) * len(history) / max(1, len(window))
                    )
                    current_frames = max(
                        1,
                        round(
                            hidden.size(0)
                            * len(current)
                            / max(1, len(window))
                        ),
                    )
                    start_frame = min(
                        max(0, start_frame), max(0, hidden.size(0) - 1)
                    )
                    end_frame = min(hidden.size(0), start_frame + current_frames)
                    chunks.append(hidden[start_frame:max(start_frame + 1, end_frame)])
                    self.context_pcm.extend(current)
                    del self.context_pcm[:-self.context_samples * 2]
                    self.current_audio_samples += len(current) // 2
                    self.processed_samples = (
                        self.current_audio_samples + self.right_context_samples
                    )
                if final and self.buffer:
                    total_samples = len(self.buffer) // 2
                    current_samples = max(
                        1,
                        total_samples
                        - min(total_samples, self.right_context_samples),
                    )
                    current = bytes(self.buffer[: current_samples * 2])
                    window_raw = bytes(self.buffer)
                    self.buffer.clear()
                    history = bytes(self.context_pcm[-self.context_samples * 2 :])
                    window = history + window_raw
                    hidden = self._encode_pcm(window)
                    start_frame = round(
                        hidden.size(0) * len(history) / max(1, len(window))
                    )
                    current_frames = max(
                        1,
                        round(
                            hidden.size(0)
                            * len(current)
                            / max(1, len(window))
                        ),
                    )
                    start_frame = min(
                        max(0, start_frame), max(0, hidden.size(0) - 1)
                    )
                    end_frame = min(hidden.size(0), start_frame + current_frames)
                    chunks.append(hidden[start_frame:max(start_frame + 1, end_frame)])
                    self.context_pcm.extend(current)
                    del self.context_pcm[:-self.context_samples * 2]
                    self.current_audio_samples += len(current) // 2
                    self.processed_samples = self.current_audio_samples + max(
                        0,
                        len(window_raw) // 2 - len(current) // 2,
                    )
                return chunks
            while len(self.buffer) >= bytes_per_chunk:
                raw = bytes(self.buffer[:bytes_per_chunk])
                del self.buffer[:bytes_per_chunk]
                if self.context_samples:
                    history = bytes(self.context_pcm[-self.context_samples * 2 :])
                    window = history + raw
                    hidden = self._encode_pcm(window)
                    new_frames = max(
                        1,
                        round(hidden.size(0) * len(raw) / max(1, len(window))),
                    )
                    chunks.append(hidden[-new_frames:])
                    self.context_pcm.extend(raw)
                    del self.context_pcm[:-self.context_samples * 2]
                else:
                    chunks.append(self._encode_pcm(raw))
                self.processed_samples += self.chunk_samples
            if final and self.buffer:
                remainder = bytes(self.buffer)
                self.buffer.clear()
                samples = len(remainder) // 2
                if self.context_samples:
                    history = bytes(self.context_pcm[-self.context_samples * 2 :])
                    window = history + remainder
                    hidden = self._encode_pcm(window)
                    new_frames = max(
                        1,
                        round(hidden.size(0) * len(remainder) / max(1, len(window))),
                    )
                    chunks.append(hidden[-new_frames:])
                    self.context_pcm.extend(remainder)
                    del self.context_pcm[:-self.context_samples * 2]
                else:
                    chunks.append(self._encode_pcm(remainder))
                self.processed_samples += samples
            return chunks
        while len(self.buffer) >= bytes_per_chunk:
            raw = bytes(self.buffer[:bytes_per_chunk])
            del self.buffer[:bytes_per_chunk]
            self.prefix_pcm.extend(raw)
            chunks.append(self._encode_prefix_delta())
            self.processed_samples += self.chunk_samples
        if final and self.buffer:
            remainder = bytes(self.buffer)
            self.buffer.clear()
            samples = len(remainder) // 2
            self.prefix_pcm.extend(remainder)
            chunks.append(self._encode_prefix_delta())
            self.processed_samples += samples
        return chunks


class ZipformerContinuousFrontend:
    """Run the causal Zipformer and compress each audio chunk to soft tokens.

    The exported WenetSpeech model consumes 45 fbank frames per encoder step:
    32 new frames plus 13 frames of right padding.  ``kaldifeat`` normally
    supplies these frames incrementally.  The Qwen environment does not ship
    that wheel, so the fallback computes only the next 465 ms window instead
    of recomputing fbank for the complete utterance on every input packet.
    """

    def __init__(
        self,
        zipformer_path: str,
        latent_adapter: torch.nn.Module,
        chunk_seconds: float = 3.0,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.encoder = TorchscriptStreamingZipformer(zipformer_path, device=device)
        self.latent_adapter = latent_adapter.to(self.device).eval()
        self.chunk_seconds = float(chunk_seconds)
        # Zipformer emits 8 frames for every 32 new fbank frames, i.e. 25 Hz.
        # Chunk the continuous stream by latent frames so a 6.0s training
        # chunk is 150 frames, rather than 19 encoder steps / 6.08s.
        self.frames_per_chunk = max(1, round(self.chunk_seconds * 25.0))
        self._use_kaldifeat = self._has_kaldifeat()
        self._fbank = None
        if not self._use_kaldifeat:
            import torchaudio

            self._fbank = torchaudio.compliance.kaldi.fbank
        self.online_fbank = None
        self.states = self.encoder.init_states()
        self.num_processed_frames = 0
        self.received_samples = 0
        self.processed_samples = 0
        self.next_window_start = 0
        self.pcm = bytearray()
        self.pending: list[torch.Tensor] = []
        self.pending_end_times: list[float] = []
        self.tail_added = False

    @staticmethod
    def _has_kaldifeat() -> bool:
        try:
            import kaldifeat  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def _make_kaldifeat_fbank(self):
        from kaldifeat import FbankOptions, OnlineFbank

        options = FbankOptions()
        options.device = "cpu"
        options.frame_opts.dither = 0
        options.frame_opts.snip_edges = False
        options.frame_opts.samp_freq = 16000
        options.mel_opts.num_bins = 80
        options.mel_opts.high_freq = -400
        return OnlineFbank(options)

    def _reset_fbank(self) -> None:
        self.online_fbank = self._make_kaldifeat_fbank() if self._use_kaldifeat else None

    def reset(self) -> None:
        self._reset_fbank()
        self.states = self.encoder.init_states()
        self.num_processed_frames = 0
        self.received_samples = 0
        self.processed_samples = 0
        self.next_window_start = 0
        self.pcm.clear()
        self.pending.clear()
        self.pending_end_times.clear()
        self.tail_added = False

    def _window_fbank(self, start_sample: int, final: bool) -> Optional[torch.Tensor]:
        """Compute exactly one encoder input window from the PCM buffer."""
        input_frames = self.encoder.input_frames
        frame_shift = 160
        frame_length = 400
        window_samples = (input_frames - 1) * frame_shift + frame_length
        end_sample = start_sample + window_samples
        available = min(self.received_samples, end_sample) - start_sample
        if available <= 0:
            return None
        if not final and end_sample > self.received_samples:
            return None
        # ``snip_edges=False`` uses one 10-ms frame of left context at every
        # non-initial boundary.  Computing a window from ``start_sample``
        # directly therefore changes its first fbank frame (and can shift the
        # Zipformer state after each 320-ms advance).  Include one preceding
        # frame, zero-pad only at utterance start, then discard that frame.
        # This makes realtime PCM windows numerically agree with the fbank of
        # the complete utterance while retaining the same 45-frame/32-frame
        # causal encoder schedule.
        left_context_samples = frame_shift
        window_start = max(0, start_sample - left_context_samples)
        prefix_pad = left_context_samples - (start_sample - window_start)
        total_samples = window_samples + left_context_samples
        start_byte = window_start * 2
        raw = bytes(
            self.pcm[start_byte : start_byte + max(0, available + (start_sample - window_start)) * 2]
        )
        values = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if prefix_pad:
            values = np.pad(values, (prefix_pad, 0))
        if len(values) < total_samples:
            values = np.pad(values, (0, total_samples - len(values)))
        waveform = torch.from_numpy(values).unsqueeze(0)
        features = self._fbank(
            waveform,
            num_mel_bins=80,
            frame_length=25.0,
            frame_shift=10.0,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=16000.0,
            snip_edges=False,
        )
        if features.size(0) < input_frames + 1:
            features = torch.nn.functional.pad(
                features,
                (0, 0, 0, input_frames + 1 - features.size(0)),
                value=-23.0258509,
            )
        return features[1 : input_frames + 1]

    def _ready_fbank_chunk(self, final: bool) -> Optional[torch.Tensor]:
        if self._use_kaldifeat:
            if self.online_fbank is None:
                self._reset_fbank()
            if (
                self.online_fbank.num_frames_ready - self.num_processed_frames
                < self.encoder.input_frames
            ):
                return None
            return torch.cat(
                [
                    self.online_fbank.get_frame(self.num_processed_frames + index)
                    for index in range(self.encoder.input_frames)
                ],
                dim=0,
            )
        return self._window_fbank(self.next_window_start, final)

    @torch.no_grad()
    def accept(self, pcm: bytes, final: bool = False) -> list[torch.Tensor]:
        usable = len(pcm) - (len(pcm) % 2)
        if usable:
            waveform = (
                torch.frombuffer(memoryview(pcm[:usable]), dtype=torch.int16)
                .clone()
                .float()
                .div_(32768.0)
            )
            self.received_samples += waveform.numel()
            if self._use_kaldifeat:
                if self.online_fbank is None:
                    self._reset_fbank()
                self.online_fbank.accept_waveform(
                    sampling_rate=16000, waveform=waveform
                )
            else:
                self.pcm.extend(pcm[:usable])

        if final and self._use_kaldifeat and not self.tail_added:
            if self.online_fbank is None:
                self._reset_fbank()
            self.online_fbank.accept_waveform(
                sampling_rate=16000,
                waveform=torch.zeros(4800, dtype=torch.float32),
            )
            self.tail_added = True

        while True:
            frames = self._ready_fbank_chunk(final=final)
            if frames is None:
                break
            frames = frames.to(self.device)
            window_start_frame = self.next_window_start // 160
            valid_input_frames = min(
                self.encoder.chunk_length,
                max(0, math.ceil(self.received_samples / 160.0) - window_start_frame),
            )
            encoder_out, _, self.states = self.encoder.forward_chunk(
                frames.unsqueeze(0), self.states
            )
            valid_outputs = min(
                encoder_out.size(1), max(0, (valid_input_frames + 3) // 4)
            )
            if valid_outputs:
                self.pending.append(
                    encoder_out[0, :valid_outputs].to(dtype=torch.bfloat16)
                )
                self.pending_end_times.extend(
                    [
                        (self.next_window_start + (index + 1) * 4) * 0.01
                        for index in range(valid_outputs)
                    ]
                )
            self.num_processed_frames += self.encoder.chunk_length
            self.next_window_start += self.encoder.chunk_length * 160

        outputs: list[torch.Tensor] = []
        pending_frames = sum(part.size(0) for part in self.pending)
        while pending_frames >= self.frames_per_chunk or (final and self.pending):
            need = self.frames_per_chunk if pending_frames >= self.frames_per_chunk else pending_frames
            pieces = []
            remaining = need
            while remaining:
                part = self.pending[0]
                take = min(remaining, part.size(0))
                pieces.append(part[:take])
                if take == part.size(0):
                    del self.pending[0]
                else:
                    self.pending[0] = part[take:]
                remaining -= take
            frames = torch.cat(pieces, dim=0)
            end_time = self.pending_end_times[need - 1]
            del self.pending_end_times[:need]
            pending_frames -= need
            mask = torch.ones(
                1, frames.size(0), dtype=torch.bool, device=frames.device
            )
            outputs.append(self.latent_adapter(frames.unsqueeze(0), mask)[0])
            self.processed_samples = min(
                self.received_samples, max(1, int(round(end_time * 16000)))
            )
        return outputs


class PersistentQwenASRTranslator:
    """Qwen-ASR PCM frontend with either serial or routed persistent decoding."""

    def __init__(
        self,
        runtime,
        checkpoint: dict,
        zipformer_path: Optional[str] = None,
        max_source_tokens: int = 48,
        max_target_tokens: int = 32,
        source_allowed_ids: Optional[set[int]] = None,
        target_no_repeat_ngram_size: int = 2,
        chunk_seconds: float = 3.0,
        temperature: Optional[float] = None,
        temporally_coupled_sampling: bool = False,
        device: str = "cuda",
        target_latent_mode: str = "full",
        target_source_mode: str = "generated",
        target_output_lag_chunks: Optional[int] = None,
        target_source_lag_chunks: Optional[int] = None,
        target_source_first_chunk_current: bool = False,
        source_decode: bool = True,
        source_only: bool = False,
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
        target_forbid_leading_punctuation: bool = False,
        target_forbid_control_labels: bool = False,
        target_forbid_cjk: bool = False,
        defer_source: bool = False,
        async_source_pipeline: bool = False,
        frontend_mode: Optional[str] = None,
        frontend_context_seconds: float = 0.0,
        frontend_right_context_seconds: float = 0.0,
        decoder_latent_subchunks: int = 1,
        source_full_latent: bool = False,
        target_reuse_full_latent: bool = False,
        compile_audio_tower: bool = False,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        target_system_instruction: str = "",
        target_prompt_mode: Optional[str] = None,
        target_decode_stride_chunks: int = 1,
        target_decode_warmup_chunks: int = 1,
        target_speculative_draft_reuse: bool = False,
        target_speculative_min_margin: float = 0.0,
        target_early_content_margin: Optional[float] = None,
        target_early_min_tokens: int = 0,
        target_user_instruction: Optional[str] = None,
    ) -> None:
        self.runtime = runtime
        self.device = torch.device(device)
        self.zipformer_path = zipformer_path
        self.source_decode = bool(source_decode)
        self.source_only = bool(source_only)
        if self.source_only and not self.source_decode:
            raise ValueError("source_only requires source decoding")
        self.compile_audio_tower = bool(compile_audio_tower)
        llm = runtime.model.thinker.to(self.device)
        config = checkpoint.get("config", {})
        stage5_temporary_draft = bool(
            config.get("stage5_temporary_draft_trajectory", False)
        )
        stage5_target_only = bool(
            config.get("stage5_persistent_trajectory", False)
            or stage5_temporary_draft
        )
        if not self.source_decode and not (
            (
                str(config.get("format", "")).startswith("native-audio-prompt")
                and bool(config.get("true_dual_routing", False))
            )
            or stage5_target_only
        ):
            raise ValueError(
                "source_decode=False requires a native true-dual checkpoint; "
                "the compatibility decoder does not implement a target-only cache"
            )
        self.source_language = str(
            source_language or config.get("source_language", "Chinese")
        )
        self.target_language = str(
            target_language or config.get("target_language", "English")
        )
        resolved_target_system_instruction = str(
            target_system_instruction
            or config.get(
                "target_system_instruction",
                (
                    "You are a professional Chinese to English simultaneous translator. "
                    "Translate only the current Chinese unit and output English text only, except "
                    "when the user explicitly states that no Chinese source text is available. "
                    "The Chinese text is produced by streaming automatic speech recognition and may "
                    "contain missing words, substitutions, homophones, punctuation errors, or errors "
                    "in names and numbers. Infer the intended spoken meaning from the available "
                    "context, correct an error only when the context supports the correction, preserve "
                    "all supported meaning, and do not invent information."
                    if stage5_target_only
                    else ""
                ),
            )
        )
        resolved_target_prompt_mode = str(
            target_prompt_mode
            or config.get(
                "target_prompt_mode",
                (
                    "stage5_temporary_draft_chat"
                    if stage5_temporary_draft
                    else "stage5_target_only_chat"
                )
                if stage5_target_only
                else "native_asr_marker",
            )
        )
        resolved_target_user_instruction = str(
            target_user_instruction
            or config.get(
                "target_user_instruction",
                (
                    (
                        "Translate all Chinese speech heard so far into English. Return only the "
                        "complete cumulative English translation. If no English word is "
                        "supported yet, end this assistant turn without text:\n"
                        if stage5_temporary_draft
                        else
                        "Continue the simultaneous Chinese-to-English translation. Reply only with "
                        "the new English suffix that is safe to make permanent for the current "
                        "Chinese unit. Never repeat or revise earlier assistant text. If no new "
                        "English word is safe, end this assistant turn without text:\n"
                    )
                    if stage5_target_only
                    else "Translate and commit this complete Chinese unit into English:\n"
                ),
            )
        )
        decode_temperature = float(
            config.get("decode_temperature", 0.0)
            if temperature is None else temperature
        )
        from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict

        targets = config.get("lora_targets", "q_proj,k_proj,v_proj,o_proj")
        lora_scope = config.get("lora_scope", "all")
        targets = resolve_lora_targets(llm, targets, lora_scope)
        lora_r = int(config.get("lora_r", 16))
        self.llm = get_peft_model(
            llm,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_r * 2,
                lora_dropout=0.05,
                target_modules=targets,
                bias="none",
            ),
        ).eval()
        set_peft_model_state_dict(
            self.llm,
            filter_adapter_state(checkpoint["llm_adapter"], lora_scope),
        )
        saved_control = checkpoint.get("control_token_rows")
        if saved_control:
            control_ids = [int(value) for value in saved_control.get("ids", [])]
            with torch.no_grad():
                for name, rows in saved_control.get("rows", {}).items():
                    module = (
                        self.llm.get_input_embeddings()
                        if name == "input"
                        else getattr(self.llm, "lm_head", None)
                    )
                    if module is not None and hasattr(module, "weight") and control_ids:
                        module.weight[control_ids].copy_(
                            rows.to(device=module.weight.device, dtype=module.weight.dtype)
                        )
        self.latent_adapter = None
        if self.zipformer_path is not None:
            adapter_type = config.get("latent_adapter", "direct")
            adapter_kwargs = {
                "encoder_dim": int(config.get("encoder_dim", 256)),
                "llm_dim": int(
                    config.get(
                        "llm_dim",
                        self.llm.get_input_embeddings().weight.shape[1],
                    )
                ),
                "num_query_tokens": int(config.get("num_query_tokens", 2)),
                "output_scale": float(config.get("qformer_output_scale", 0.03)),
            }
            if adapter_type == "direct":
                self.latent_adapter = DirectChunkAdapter(**adapter_kwargs)
            elif adapter_type == "qformer":
                self.latent_adapter = ChunkQFormer(
                    **adapter_kwargs,
                    hidden_dim=int(config.get("qformer_hidden_dim", 512)),
                    num_layers=int(config.get("qformer_layers", 2)),
                    num_heads=int(config.get("qformer_heads", 8)),
                    ffn_dim=int(config.get("qformer_ffn_dim", 2048)),
                )
            else:
                raise ValueError(f"unsupported latent adapter: {adapter_type}")
            adapter_state = checkpoint.get("latent_adapter")
            if not adapter_state:
                raise ValueError("Zipformer runtime requires checkpoint['latent_adapter']")
            self.latent_adapter.load_state_dict(adapter_state, strict=True)
            adapter_dtype = (
                torch.float32
                if config.get("latent_adapter_dtype", "bf16") == "fp32"
                else torch.bfloat16
            )
            self.latent_adapter = self.latent_adapter.to(
                device=self.device, dtype=adapter_dtype
            ).eval()
        source_adapter_name = None
        target_adapter_name = None
        if (
            config.get("source_adapter_only", False)
            or config.get("target_adapter_only", False)
            or config.get("dual_adapters", False)
        ):
            source_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_r * 2,
                lora_dropout=0.05,
                target_modules=targets,
                bias="none",
            )
            self.llm.add_adapter("source", source_config)
            source_state = checkpoint.get("llm_source_adapter", checkpoint["llm_adapter"])
            set_peft_model_state_dict(
                self.llm,
                filter_adapter_state(source_state, lora_scope),
                adapter_name="source",
            )
            source_adapter_name = config.get("source_adapter_name", "source")
            target_adapter_name = config.get("target_adapter_name", "default")
            self.llm.set_adapter(target_adapter_name)
            # Adapters added after the initial ``.eval()`` call can retain
            # train-mode dropout.  Streaming evaluation must be deterministic
            # for both persistent branches, especially when source/target
            # adapters are switched between forwards.
            self.llm.eval()
        self.tokenizer = runtime.processor.tokenizer
        self.tokenizer.chat_template = runtime.processor.chat_template
        source_wait_token = config.get("source_wait_token", "<|fim_middle|>")
        target_wait_token = config.get(
            "target_wait_token", config.get("wait_token", "<|fim_middle|>")
        )
        source_wait_ids = self.tokenizer(
            source_wait_token, add_special_tokens=False
        )["input_ids"]
        target_wait_ids = self.tokenizer(
            target_wait_token, add_special_tokens=False
        )["input_ids"]
        wait_ids = target_wait_ids
        latent_open_ids = [
            self.tokenizer.convert_tokens_to_ids(runtime.processor.audio_bos_token)
        ]
        latent_close_ids = [
            self.tokenizer.convert_tokens_to_ids(runtime.processor.audio_eos_token)
        ]
        latent_pad_id = self.tokenizer.convert_tokens_to_ids(
            getattr(runtime.processor, "audio_pad_token", "<|audio_pad|>")
        )
        source_prefix_ids = self.tokenizer(
            f"language {self.source_language}<asr_text>", add_special_tokens=False
        )["input_ids"]
        # Qwen tokenizes the placeholder with and without a leading space as
        # different ids; decode() strips spaces, so both variants must be
        # blocked or the placeholder leaks into displayed translations.
        target_forbidden_ids = sorted(
            {
                token_id
                for value in ("None", " None")
                for token_id in self.tokenizer(
                    value, add_special_tokens=False
                )["input_ids"]
            }
        )
        if target_forbid_cjk:
            target_forbidden_ids = sorted(
                set(target_forbidden_ids) | cjk_token_ids(self.tokenizer)
            )
        target_control_ids = {
            token_id
            for value in (
                "language",
                " language",
                "<translation>",
                " <translation>",
                "<asr_text>",
                " <asr_text>",
            )
            for token_id in self.tokenizer(value, add_special_tokens=False)["input_ids"]
        }
        target_leading_forbidden_ids = []
        if target_forbid_leading_punctuation:
            target_leading_forbidden_ids = sorted(
                {
                    token_id
                    for value in ("-", " -", "/", " /")
                    for token_id in self.tokenizer(
                        value, add_special_tokens=False
                    )["input_ids"]
                }
            )
        if target_forbid_control_labels:
            target_leading_forbidden_ids = sorted(
                set(target_leading_forbidden_ids) | target_control_ids
            )
        if config.get("format", "").startswith("native-audio-prompt") or stage5_target_only:
            decoder_args = (
                self.llm,
                self.tokenizer,
                wait_ids,
                int(self.tokenizer.eos_token_id),
            )
            common_kwargs = {
                "audio_start_id": latent_open_ids[0],
                "audio_end_id": latent_close_ids[0],
                "audio_pad_id": latent_pad_id,
                "max_source_tokens": max_source_tokens,
                "max_target_tokens": max_target_tokens,
                "temperature": decode_temperature,
                "temporally_coupled_sampling": bool(temporally_coupled_sampling),
                "source_allowed_ids": (
                    source_allowed_ids
                    if source_allowed_ids is not None
                    else source_text_token_ids(self.tokenizer, self.source_language)
                ),
                "target_forbidden_ids": target_forbidden_ids,
                "target_leading_forbidden_ids": target_leading_forbidden_ids,
                "source_prefix_ids": source_prefix_ids,
            }
            if config.get("true_dual_routing", False) or stage5_target_only:
                target_marker_ids = self.tokenizer(
                    "\n<translation>\n", add_special_tokens=False
                )["input_ids"]
                source_context_ids = self.tokenizer(
                    "\n<asr_text>", add_special_tokens=False
                )["input_ids"]
                self.decoder = NativeTrueDualQwenDecoder(
                    *decoder_args,
                    **common_kwargs,
                    target_marker_ids=target_marker_ids,
                    source_context_ids=source_context_ids,
                    source_wait_id=source_wait_ids,
                    target_wait_id=target_wait_ids,
                    target_output_lag_chunks=int(
                        config.get("target_output_lag_chunks", 1)
                        if target_output_lag_chunks is None
                        else target_output_lag_chunks
                    ),
                    target_source_lag_chunks=int(
                        config.get("target_source_lag_chunks", 1)
                        if target_source_lag_chunks is None
                        else target_source_lag_chunks
                    ),
                    target_source_first_chunk_current=target_source_first_chunk_current,
                    target_latent_mode=target_latent_mode,
                    target_source_mode=target_source_mode,
                    target_no_repeat_ngram_size=target_no_repeat_ngram_size,
                    source_no_repeat_ngram_size=source_no_repeat_ngram_size,
                    target_min_tokens_before_wait=target_min_tokens_before_wait,
                    target_min_chunks_before_wait=target_min_chunks_before_wait,
                    target_force_token_if_empty=target_force_token_if_empty,
                    target_wait_logit_bias=target_wait_logit_bias,
                    target_wait_bias_first_token_only=target_wait_bias_first_token_only,
                    target_wait_bias_before_first_output=target_wait_bias_before_first_output,
                    target_wait_bias_after_waits=target_wait_bias_after_waits,
                    target_adaptive_wait_bias=target_adaptive_wait_bias,
                    target_adaptive_wait_margin=target_adaptive_wait_margin,
                    defer_source=defer_source,
                    async_source_pipeline=async_source_pipeline,
                    target_system_instruction=resolved_target_system_instruction,
                    target_prompt_mode=resolved_target_prompt_mode,
                    target_decode_stride_chunks=target_decode_stride_chunks,
                    target_decode_warmup_chunks=target_decode_warmup_chunks,
                    target_speculative_draft_reuse=target_speculative_draft_reuse,
                    target_speculative_min_margin=target_speculative_min_margin,
                    target_early_content_margin=target_early_content_margin,
                    target_early_min_tokens=target_early_min_tokens,
                    target_user_instruction=resolved_target_user_instruction,
                    source_adapter_name=source_adapter_name,
                    target_adapter_name=target_adapter_name,
                    source_decode=self.source_decode,
                )
            else:
                self.decoder = NativePersistentQwenDecoder(
                    *decoder_args, **common_kwargs
                )
        else:
            self.decoder = DualStreamDecoder(
                self.llm,
                self.tokenizer,
                wait_ids,
                int(self.tokenizer.eos_token_id),
                max_source_tokens=max_source_tokens,
                max_target_tokens=max_target_tokens,
                target_no_repeat_ngram_size=target_no_repeat_ngram_size,
                source_allowed_ids=source_allowed_ids,
                force_1d_position_ids=True,
                latent_open_ids=latent_open_ids,
                latent_close_ids=latent_close_ids,
                source_prefix_ids=source_prefix_ids,
            )
        self.decoder.start()
        self._frontend_chunk_seconds = chunk_seconds
        if frontend_mode is None or frontend_mode == "checkpoint":
            checkpoint_latent_type = config.get("latent_type")
            self._frontend_chunk_local = checkpoint_latent_type in {
                "qwen3_asr_audio_encoder_chunk_local",
                "qwen3_asr_audio_encoder_rolling_context",
            }
        elif frontend_mode == "chunk_local":
            self._frontend_chunk_local = True
        elif frontend_mode == "prefix":
            self._frontend_chunk_local = False
        elif frontend_mode == "rolling":
            self._frontend_chunk_local = True
        else:
            raise ValueError(f"unsupported frontend_mode: {frontend_mode}")
        if frontend_mode == "rolling":
            self._frontend_context_seconds = float(frontend_context_seconds)
        elif frontend_mode is None or frontend_mode == "checkpoint":
            self._frontend_context_seconds = (
                float(config.get("frontend_context_seconds", 0.0))
                if config.get("latent_type")
                == "qwen3_asr_audio_encoder_rolling_context"
                else 0.0
            )
        else:
            self._frontend_context_seconds = 0.0
        self._frontend_right_context_seconds = (
            float(frontend_right_context_seconds)
            if frontend_mode == "rolling"
            else 0.0
        )
        if decoder_latent_subchunks < 1:
            raise ValueError("decoder_latent_subchunks must be positive")
        self._decoder_latent_subchunks = int(decoder_latent_subchunks)
        self._source_full_latent = bool(source_full_latent)
        self._target_reuse_full_latent = bool(target_reuse_full_latent)
        self._frontend_device = device
        if self.zipformer_path is not None:
            self.frontend = ZipformerContinuousFrontend(
                self.zipformer_path,
                self.latent_adapter,
                chunk_seconds=chunk_seconds,
                device=device,
            )
        else:
            self.frontend = QwenASRContinuousFrontend(
                runtime,
                thinker=self.llm,
                chunk_seconds=chunk_seconds,
                chunk_local=self._frontend_chunk_local,
                context_seconds=self._frontend_context_seconds,
                right_context_seconds=self._frontend_right_context_seconds,
                compile_audio_tower=self.compile_audio_tower,
                device=device,
            )

    @torch.no_grad()
    def set_target_source_schedule(self, chunks: list[list[int]]) -> None:
        if not hasattr(self.decoder, "set_target_source_schedule"):
            raise RuntimeError("the active decoder does not support oracle source schedules")
        self.decoder.set_target_source_schedule(chunks)

    @torch.no_grad()
    def reset(self) -> None:
        """Reset one utterance while reusing the loaded Qwen and LoRA modules."""
        if hasattr(self.decoder, "reset"):
            self.decoder.reset()
        else:
            self.decoder.state = None
            self.decoder.start()
        if self.zipformer_path is not None:
            self.frontend = ZipformerContinuousFrontend(
                self.zipformer_path,
                self.latent_adapter,
                chunk_seconds=self._frontend_chunk_seconds,
                device=self._frontend_device,
            )
        else:
            self.frontend = QwenASRContinuousFrontend(
                self.runtime,
                thinker=self.llm,
                chunk_seconds=self._frontend_chunk_seconds,
                chunk_local=self._frontend_chunk_local,
                context_seconds=self._frontend_context_seconds,
                right_context_seconds=self._frontend_right_context_seconds,
                compile_audio_tower=self.compile_audio_tower,
                device=self._frontend_device,
            )

    @torch.inference_mode()
    def accept_pcm(self, pcm: bytes, final: bool = False) -> list[StreamingEvent]:
        features = self.frontend.accept(pcm, final=final)
        events: list[StreamingEvent] = []
        for index, feature in enumerate(features):
            parts = (
                [feature]
                if self._decoder_latent_subchunks == 1
                else list(
                    torch.tensor_split(
                        feature,
                        min(self._decoder_latent_subchunks, max(1, feature.size(0))),
                        dim=0,
                    )
                )
            )
            feature_end = self.frontend.processed_samples / 16000.0
            if len(features) > 1:
                feature_end -= (len(features) - 1 - index) * self._frontend_chunk_seconds
            feature_start = max(0.0, feature_end - self._frontend_chunk_seconds)
            for part_index, part in enumerate(parts):
                is_final = final and index + 1 == len(features) and part_index + 1 == len(parts)
                if self.source_only:
                    source_ids, finished = self.decoder.accept_source_only(
                        part, is_final=is_final
                    )
                    target_ids = []
                elif self._source_full_latent and len(parts) > 1:
                    target_part = part
                    if self._target_reuse_full_latent:
                        if part_index == 0:
                            target_part = feature
                        else:
                            target_part = feature.new_empty((0, feature.size(-1)))
                    source_ids, target_ids, finished = self.decoder.accept(
                        target_part,
                        is_final=is_final,
                        source_latent=feature,
                        update_source=part_index + 1 == len(parts),
                    )
                else:
                    source_ids, target_ids, finished = self.decoder.accept(
                        part, is_final=is_final
                    )
                source_cache_length, target_cache_length = self._cache_lengths()
                part_time = feature_start + (part_index + 1) * (
                    feature_end - feature_start
                ) / max(1, len(parts))
                events.append(
                    StreamingEvent(
                        audio_time=part_time,
                        source_delta=self.decoder.decode(source_ids),
                        source_prefix=self.decoder.decode(self.decoder.source_ids),
                        target_delta=self.decoder.decode(target_ids),
                        target_prefix=self.decoder.decode(self.decoder.target_ids),
                        finished=finished,
                        cache_attention_length=source_cache_length + target_cache_length,
                        source_cache_attention_length=source_cache_length,
                        target_cache_attention_length=target_cache_length,
                        target_delta_token_ids=[int(value) for value in target_ids],
                        target_token_ids=[int(value) for value in self.decoder.target_ids],
                        target_eot_trace=dict(
                            getattr(self.decoder, "last_stage5_eot_trace", None) or {}
                        ),
                    )
                )
        if final and not features:
            if self.source_only:
                self.decoder.state.finished = True
                source_ids, target_ids, finished = [], [], True
            elif hasattr(self.decoder, "accept_final_without_latent"):
                source_ids, target_ids, finished = self.decoder.accept_final_without_latent()
            else:
                source_ids, target_ids, finished = self.decoder.accept(None, is_final=True)
            source_cache_length, target_cache_length = self._cache_lengths()
            events.append(
                StreamingEvent(
                    audio_time=self.frontend.processed_samples / 16000.0,
                    source_delta=self.decoder.decode(source_ids),
                    source_prefix=self.decoder.decode(self.decoder.source_ids),
                    target_delta=self.decoder.decode(target_ids),
                    target_prefix=self.decoder.decode(self.decoder.target_ids),
                    finished=finished,
                    cache_attention_length=source_cache_length + target_cache_length,
                    source_cache_attention_length=source_cache_length,
                    target_cache_attention_length=target_cache_length,
                    target_delta_token_ids=[int(value) for value in target_ids],
                    target_token_ids=[int(value) for value in self.decoder.target_ids],
                    target_eot_trace=dict(
                        getattr(self.decoder, "last_stage5_eot_trace", None) or {}
                    ),
                )
            )
        return events

    def _cache_lengths(self) -> tuple[int, int]:
        state = self.decoder.state
        if hasattr(state, "attention_length"):
            length = int(state.attention_length)
            return length, length
        return int(state.source.attention_length), int(state.target.attention_length)
