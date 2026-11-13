"""Bounded prefix and KV-cache runtime for long streaming sessions.

The original path is preserved until the configured acoustic capacity is
reached. At capacity, complete old events are retired and the retained events
are replayed from the pinned native prefix.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, asdict
from types import MethodType
from typing import Any


@dataclass(frozen=True)
class SlideReceipt:
    slide_index: int
    retired_event_ids: tuple[int, ...]
    retained_event_ids: tuple[int, ...]
    retired_pcm_bytes: int
    retained_pcm_bytes: int
    retired_frames: int
    retained_frames: int
    semantic_coverage_certified: bool = False


class BoundedPrefixWindowMixin:
    """Mixin installed on an untouched QwenASRContinuousFrontend instance."""

    def initialize_bounded_prefix_window(
        self,
        *,
        window_blocks: int = 12,
        retire_blocks: int = 2,
    ) -> None:
        if bool(getattr(self, "chunk_local", False)):
            raise ValueError("bounded streaming requires the cumulative prefix frontend")
        if bool(getattr(self, "cache_encoder_windows", False)):
            raise ValueError("bounded streaming cannot use the segmented-window frontend")
        if window_blocks < 2 or retire_blocks < 1 or retire_blocks >= window_blocks:
            raise ValueError("invalid block window geometry")
        if getattr(self, "processed_samples", 0) or len(getattr(self, "prefix_pcm", ())):
            raise RuntimeError("install the bounded frontend before the first PCM byte")
        self.window_blocks = int(window_blocks)
        self.retire_blocks = int(retire_blocks)
        self.window_pcm_bytes = self.window_blocks * int(self.chunk_samples) * 2
        self._window_block_pcm_bytes: deque[int] = deque()
        self._window_block_frames: deque[int] = deque()
        self._window_block_ids: deque[int] = deque()
        self._window_total_blocks = 0
        self._window_total_evicted_pcm_bytes = 0
        self._window_total_evicted_frames = 0
        self._window_slide_receipts: list[SlideReceipt] = []
        self._window_pressure_active = False
        self._window_debug_calls: list[dict[str, Any]] = []
        self._window_external_retire_receipts: list[dict[str, Any]] = []

    def _window_ready(self) -> None:
        if not hasattr(self, "_window_block_frames"):
            raise RuntimeError("bounded prefix window was not initialized")

    def _window_retained_bytes(self) -> int:
        return len(self.prefix_pcm) + len(self.buffer)

    def _reconcile_external_retirement(self) -> None:
        """Mirror complete PCM blocks already retired by the native PDO policy."""

        shadow_bytes = sum(self._window_block_pcm_bytes)
        prefix_bytes = len(self.prefix_pcm)
        if shadow_bytes == prefix_bytes:
            return
        if shadow_bytes < prefix_bytes:
            raise RuntimeError("parent prefix grew outside the bounded frontend")
        remaining = shadow_bytes - prefix_bytes
        retired_ids: list[int] = []
        retired_frames = 0
        while remaining:
            if not self._window_block_pcm_bytes:
                raise RuntimeError("external retirement exceeds shadow ownership")
            block_bytes = self._window_block_pcm_bytes[0]
            if block_bytes > remaining:
                raise RuntimeError("external retirement split a frontend block")
            remaining -= block_bytes
            self._window_block_pcm_bytes.popleft()
            retired_frames += self._window_block_frames.popleft()
            retired_ids.append(self._window_block_ids.popleft())
        self._window_external_retire_receipts.append({
            "retired_event_ids": retired_ids,
            "retired_pcm_bytes": shadow_bytes - prefix_bytes,
            "retired_frames": retired_frames,
            "retained_event_ids": list(self._window_block_ids),
            "retained_pcm_bytes": prefix_bytes,
        })

    def _retire_for_room(self, additional_bytes: int) -> None:
        self._window_ready()
        if additional_bytes < 0 or additional_bytes > self.window_pcm_bytes:
            raise ValueError("invalid incoming PCM span")
        while self._window_retained_bytes() + additional_bytes > self.window_pcm_bytes:
            if len(self._window_block_pcm_bytes) < self.retire_blocks:
                raise RuntimeError("capacity pressure arrived before complete retireable blocks")
            retired_bytes = sum(
                self._window_block_pcm_bytes[index]
                for index in range(self.retire_blocks)
            )
            retired_frames = sum(
                self._window_block_frames[index]
                for index in range(self.retire_blocks)
            )
            retired_ids = tuple(
                self._window_block_ids[index]
                for index in range(self.retire_blocks)
            )
            if retired_bytes <= 0 or retired_bytes > len(self.prefix_pcm):
                raise RuntimeError("retired PCM ownership differs from the prefix")
            del self.prefix_pcm[:retired_bytes]
            for _ in range(self.retire_blocks):
                self._window_block_pcm_bytes.popleft()
                self._window_block_frames.popleft()
                self._window_block_ids.popleft()
            self.previous_prefix_frames -= retired_frames
            if self.previous_prefix_frames < 0:
                raise RuntimeError("retired more frontend frames than exist")
            self._window_total_evicted_pcm_bytes += retired_bytes
            self._window_total_evicted_frames += retired_frames
            receipt = SlideReceipt(
                slide_index=len(self._window_slide_receipts),
                retired_event_ids=retired_ids,
                retained_event_ids=tuple(self._window_block_ids),
                retired_pcm_bytes=retired_bytes,
                retained_pcm_bytes=len(self.prefix_pcm),
                retired_frames=retired_frames,
                retained_frames=sum(self._window_block_frames),
            )
            self._window_slide_receipts.append(receipt)
            self._record_retained_pcm()

    def _encode_bounded_block(self, raw: bytes) -> Any:
        if not raw:
            raise ValueError("cannot encode an empty PCM block")
        self._retire_for_room(len(raw))
        self.prefix_pcm.extend(raw)
        self._record_retained_pcm()
        delta = self._encode_prefix_delta()
        frames = int(delta.size(0))
        if frames < 1:
            raise RuntimeError("the parent prefix frontend produced no delta frame")
        self._window_block_pcm_bytes.append(len(raw))
        self._window_block_frames.append(frames)
        self._window_block_ids.append(self._window_total_blocks)
        self._window_total_blocks += 1
        if sum(self._window_block_pcm_bytes) != len(self.prefix_pcm):
            raise RuntimeError("retained block bytes no longer cover the prefix exactly")
        if self._window_retained_bytes() > self.window_pcm_bytes:
            raise RuntimeError("bounded frontend exceeded its physical PCM cap")
        return delta

    def _record_parent_outputs(
        self,
        outputs: list[Any],
        *,
        bytes_per_chunk: int,
        prefix_before: int,
        buffered_before: int,
        incoming_bytes: int,
        final: bool,
    ) -> None:
        """Shadow the original cumulative frontend without changing its work."""

        available = buffered_before + incoming_bytes
        complete = available // bytes_per_chunk
        expected_blocks = complete
        remainder = available % bytes_per_chunk
        if final and remainder:
            expected_blocks += 1
        if expected_blocks != len(outputs):
            raise RuntimeError(
                "parent frontend output count differs from shadow block ownership"
            )
        appended = len(self.prefix_pcm) - prefix_before
        if appended < 0:
            raise RuntimeError("parent frontend prefix shrank before pressure")
        if not outputs:
            if appended:
                raise RuntimeError("parent frontend appended PCM without an output")
            block_bytes = []
        else:
            leading = bytes_per_chunk * (len(outputs) - 1)
            final_bytes = appended - leading
            if final_bytes <= 0 or final_bytes > bytes_per_chunk:
                raise RuntimeError("parent frontend appended an invalid output span")
            block_bytes = [bytes_per_chunk] * (len(outputs) - 1) + [final_bytes]
        for raw_bytes, delta in zip(block_bytes, outputs):
            frames = int(delta.size(0))
            if frames < 1:
                raise RuntimeError("the parent prefix frontend produced no delta frame")
            self._window_block_pcm_bytes.append(raw_bytes)
            self._window_block_frames.append(frames)
            self._window_block_ids.append(self._window_total_blocks)
            self._window_total_blocks += 1
        self._window_debug_calls.append({
            "prefix_before": prefix_before,
            "prefix_after": len(self.prefix_pcm),
            "buffered_before": buffered_before,
            "incoming_bytes": incoming_bytes,
            "bytes_per_chunk": bytes_per_chunk,
            "outputs": len(outputs),
            "appended": appended,
            "recorded": list(block_bytes),
            "blocks_after": list(self._window_block_pcm_bytes),
        })
        if sum(self._window_block_pcm_bytes) != len(self.prefix_pcm):
            raise RuntimeError(
                "shadow block bytes no longer cover the parent prefix: "
                f"shadow={sum(self._window_block_pcm_bytes)} "
                f"prefix={len(self.prefix_pcm)} buffer={len(self.buffer)} "
                f"buffered_before={buffered_before} incoming={incoming_bytes} "
                f"outputs={len(outputs)} final={final} "
                f"bytes_per_chunk={bytes_per_chunk} prefix_before={prefix_before} "
                f"blocks={list(self._window_block_pcm_bytes)} "
                f"calls={self._window_debug_calls[-8:]}"
            )

    def _accept_after_pressure(self, pcm: bytes, final: bool) -> list[Any]:
        """Use bounded prefix rebuilding only after the first real overflow."""

        data = memoryview(pcm)
        outputs: list[Any] = []
        bytes_per_chunk = int(self.chunk_samples) * 2
        cursor = 0
        while cursor < len(data):
            need = bytes_per_chunk - len(self.buffer)
            take = min(need, len(data) - cursor)
            self._retire_for_room(take)
            self.buffer.extend(data[cursor : cursor + take])
            cursor += take
            self._record_retained_pcm()
            if len(self.buffer) == bytes_per_chunk:
                raw = bytes(self.buffer)
                self.buffer.clear()
                outputs.append(self._encode_bounded_block(raw))
                self.processed_samples += self.chunk_samples
        if final and self.buffer:
            raw = bytes(self.buffer)
            self.buffer.clear()
            outputs.append(self._encode_bounded_block(raw))
            self.processed_samples += len(raw) // 2
        if self._window_retained_bytes() > self.window_pcm_bytes:
            raise RuntimeError("PCM cap violated after accept")
        return outputs

    def accept(self, pcm: bytes, final: bool = False) -> list[Any]:
        """Delegate exactly until pressure, then keep a bounded cumulative prefix."""

        self._window_ready()
        self._reconcile_external_retirement()
        if not isinstance(pcm, (bytes, bytearray, memoryview)):
            raise TypeError("PCM must be bytes-like")
        incoming_bytes = len(pcm)
        if (
            not self._window_pressure_active
            and self._window_retained_bytes() + incoming_bytes
            <= self.window_pcm_bytes
        ):
            buffered_before = len(self.buffer)
            prefix_before = len(self.prefix_pcm)
            bytes_per_chunk = int(self.chunk_samples) * 2
            outputs = super().accept(pcm, final=final)
            self._record_parent_outputs(
                outputs,
                bytes_per_chunk=bytes_per_chunk,
                prefix_before=prefix_before,
                buffered_before=buffered_before,
                incoming_bytes=incoming_bytes,
                final=final,
            )
            return outputs

        self._window_pressure_active = True
        return self._accept_after_pressure(pcm, final)

    def bounded_window_stats(self) -> dict[str, Any]:
        self._window_ready()
        return {
            "window_blocks": self.window_blocks,
            "retire_blocks": self.retire_blocks,
            "window_pcm_bytes": self.window_pcm_bytes,
            "retained_pcm_bytes": self._window_retained_bytes(),
            "retained_complete_blocks": len(self._window_block_ids),
            "retained_event_ids": list(self._window_block_ids),
            "retained_frames": sum(self._window_block_frames),
            "total_blocks": self._window_total_blocks,
            "total_evicted_pcm_bytes": self._window_total_evicted_pcm_bytes,
            "total_evicted_frames": self._window_total_evicted_frames,
            "pressure_active": self._window_pressure_active,
            "external_retirements": list(self._window_external_retire_receipts),
            "slides": [asdict(value) for value in self._window_slide_receipts],
        }

    def retention_stats(self) -> dict[str, Any]:
        values = super().retention_stats()
        values.update({f"bounded_{key}": value for key, value in self.bounded_window_stats().items()})
        values["mode"] = (
            f"bounded_prefix_retire{self.retire_blocks}_of{self.window_blocks}"
        )
        return values


def bind_bounded_prefix_frontend(
    frontend: Any,
    *,
    window_blocks: int = 12,
    retire_blocks: int = 2,
) -> Any:
    """Install the mixin without reconstructing or changing the parent frontend."""

    if isinstance(frontend, BoundedPrefixWindowMixin):
        raise RuntimeError("bounded prefix frontend is already installed")
    original_class = frontend.__class__
    bounded_class = type(
        f"Bounded{original_class.__name__}",
        (BoundedPrefixWindowMixin, original_class),
        {"__module__": __name__},
    )
    frontend.__class__ = bounded_class
    frontend.initialize_bounded_prefix_window(
        window_blocks=window_blocks,
        retire_blocks=retire_blocks,
    )
    return frontend


class BoundedKVWindowController:
    """Rebuild the target-only persistent KV after each bounded-window slide."""

    def __init__(
        self,
        decoder: Any,
        *,
        window_blocks: int = 12,
        retire_blocks: int = 2,
        window_adapter: str | None = None,
        source_frame_cap: int = 336,
    ) -> None:
        if getattr(decoder, "target_prompt_mode", None) != "stage5_temporary_draft_chat":
            raise ValueError("bounded streaming requires the temporary-draft route")
        if getattr(decoder, "source_decode", True):
            raise ValueError("bounded streaming requires the one-cache target-only route")
        if window_blocks < 2 or retire_blocks < 1 or retire_blocks >= window_blocks:
            raise ValueError("invalid decoder window geometry")
        if decoder.state is None or decoder.state.target.attention_length < 1:
            raise RuntimeError("decoder must have its pinned native prefix")
        self.decoder = decoder
        self.window_blocks = int(window_blocks)
        self.retire_blocks = int(retire_blocks)
        self.window_adapter = window_adapter
        self.source_frame_cap = int(source_frame_cap)
        self.prefix_template = copy.deepcopy(decoder.state.target)
        self.original_accept = decoder.accept
        self.latents: deque[tuple[int, Any]] = deque()
        self.next_event_id = 0
        self.slides: list[dict[str, Any]] = []
        self.window_adapter_active = False
        self.rebuild_active = False
        self.rebuild_count = 0
        self.max_retained_frames = 0
        self.max_physical_tokens_after_rebuild = int(
            decoder._physical_cache_length(
                decoder.state.target.past_key_values,
                decoder.state.target.attention_length,
            )
        )

    @staticmethod
    def _active_adapter(llm: Any) -> Any:
        return getattr(llm, "active_adapter", None)

    @staticmethod
    def _restore_adapter(llm: Any, value: Any) -> None:
        if value is None or not hasattr(llm, "set_adapter"):
            return
        llm.set_adapter(value)

    def _activate_window_adapter(self) -> None:
        if self.window_adapter is None or self.window_adapter_active:
            return
        available = getattr(self.decoder.llm, "peft_config", {})
        if self.window_adapter not in available:
            raise ValueError("the checkpoint has no requested window adapter")
        self.decoder.llm.set_adapter(self.window_adapter)
        self.window_adapter_active = True

    def _rebuild_after_slide(self) -> dict[str, Any]:
        if len(self.latents) != self.window_blocks:
            raise RuntimeError("decoder slide requires one complete full window")
        retired = [self.latents.popleft() for _ in range(self.retire_blocks)]
        retained = list(self.latents)
        self._activate_window_adapter()
        self.decoder.state.target = copy.deepcopy(self.prefix_template)
        if self.rebuild_active:
            raise RuntimeError("nested decoder window rebuild")
        self.rebuild_active = True
        try:
            for _, latent in retained:
                self.decoder._append_latent(latent, target=True)
        finally:
            self.rebuild_active = False
        physical = int(
            self.decoder._physical_cache_length(
                self.decoder.state.target.past_key_values,
                self.decoder.state.target.attention_length,
            )
        )
        receipt = {
            "slide_index": len(self.slides),
            "retired_event_ids": [event_id for event_id, _ in retired],
            "retained_event_ids": [event_id for event_id, _ in retained],
            "retired_frames": sum(int(latent.size(0)) for _, latent in retired),
            "retained_frames": sum(int(latent.size(0)) for _, latent in retained),
            "physical_tokens_after_rebuild": physical,
            "semantic_coverage_certified": False,
        }
        self.slides.append(receipt)
        self.rebuild_count += 1
        self.max_physical_tokens_after_rebuild = max(
            self.max_physical_tokens_after_rebuild, physical
        )
        return receipt

    def accept(
        self,
        latent: Any,
        is_final: bool = False,
        source_latent: Any | None = None,
        update_source: bool = True,
    ) -> Any:
        if latent.ndim == 3:
            latent = latent.squeeze(0)
        if latent.ndim != 2:
            raise ValueError("actual latent must be [frames, hidden]")
        old_latents = deque(self.latents)
        old_target = self.decoder.state.target
        old_target_ids = list(self.decoder.state.target_ids)
        old_finished = bool(self.decoder.state.finished)
        old_adapter = self._active_adapter(self.decoder.llm)
        old_adapter_active = self.window_adapter_active
        old_slide_count = len(self.slides)
        old_rebuild_count = self.rebuild_count
        old_acoustic = copy.deepcopy(self.decoder.state.acoustic)
        try:
            if len(self.latents) == self.window_blocks:
                self._rebuild_after_slide()
            result = self.original_accept(
                latent,
                is_final=is_final,
                source_latent=source_latent,
                update_source=update_source,
            )
            stored = latent.detach().clone()
            self.latents.append((self.next_event_id, stored))
            self.next_event_id += 1
            retained_frames = sum(int(value.size(0)) for _, value in self.latents)
            self.max_retained_frames = max(self.max_retained_frames, retained_frames)
            if len(self.latents) > self.window_blocks:
                raise RuntimeError("decoder retained more than the configured event cap")
            if retained_frames > self.source_frame_cap:
                raise RuntimeError("decoder retained source frames exceed the configured cap")
            if len(self.decoder.state.target_ids) > int(self.decoder.max_target_tokens):
                raise RuntimeError("replaceable History tail exceeds its existing cap")
            return result
        except BaseException:
            self.latents = old_latents
            self.decoder.state.target = old_target
            self.decoder.state.target_ids = old_target_ids
            self.decoder.state.finished = old_finished
            self.decoder.state.acoustic = old_acoustic
            del self.slides[old_slide_count:]
            self.rebuild_active = False
            self.rebuild_count = old_rebuild_count
            self.window_adapter_active = old_adapter_active
            self._restore_adapter(self.decoder.llm, old_adapter)
            raise

    def stats(self) -> dict[str, Any]:
        return {
            "window_blocks": self.window_blocks,
            "retire_blocks": self.retire_blocks,
            "retained_event_ids": [event_id for event_id, _ in self.latents],
            "retained_frames": sum(int(value.size(0)) for _, value in self.latents),
            "max_retained_frames": self.max_retained_frames,
            "window_adapter": self.window_adapter,
            "window_adapter_active": self.window_adapter_active,
            "slides": list(self.slides),
            "semantic_coverage_certified": False,
        }


def install_bounded_window(
    translator: Any,
    *,
    window_blocks: int = 12,
    retire_blocks: int = 2,
    window_adapter: str | None = None,
    source_frame_cap: int = 336,
) -> BoundedKVWindowController:
    """Bind frontend and decoder before the first packet of one new session."""

    bind_bounded_prefix_frontend(
        translator.frontend,
        window_blocks=window_blocks,
        retire_blocks=retire_blocks,
    )
    controller = BoundedKVWindowController(
        translator.decoder,
        window_blocks=window_blocks,
        retire_blocks=retire_blocks,
        window_adapter=window_adapter,
        source_frame_cap=source_frame_cap,
    )
    translator.decoder.accept = MethodType(
        lambda _decoder, latent, is_final=False, source_latent=None, update_source=True: controller.accept(
            latent,
            is_final=is_final,
            source_latent=source_latent,
            update_source=update_source,
        ),
        translator.decoder,
    )
    translator._bounded_window_controller = controller
    return controller
