"""Causal transcript ledger for bounded streaming inference.

The decoder continues to see only its replaceable active draft.  Text that
belongs to retired acoustic frames is moved to an external delivery ledger and
is prepended only when producing user-visible events.  This preserves the
training contract while preventing a window rebuild from deleting already
translated content.
"""

from __future__ import annotations

import math
from types import MethodType
from typing import Any

from .bounded_window import (
    BoundedKVWindowController,
    bind_bounded_prefix_frontend,
)


def _lcp(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def select_commit_boundary(
    previous: list[int],
    current: list[int],
    *,
    retired_frames: int,
    retained_frames: int,
) -> dict[str, Any]:
    """Choose one reference-free causal boundary for a completed slide."""

    if retired_frames <= 0 or retained_frames <= 0:
        raise ValueError("slide frame ownership must be positive")
    candidates: list[tuple[int, int]] = []
    for split in range(len(previous)):
        overlap = _lcp(previous[split:], current)
        if overlap:
            candidates.append((overlap, split))
    best = max((overlap for overlap, _ in candidates), default=0)
    best_splits = [split for overlap, split in candidates if overlap == best]
    exact_split = best_splits[0] if len(best_splits) == 1 else None
    geometric_split = math.floor(
        len(previous) * retired_frames / (retired_frames + retained_frames)
    )
    if exact_split is not None and best >= 2:
        selected = exact_split
        reason = "unique_exact_anchor"
    else:
        selected = geometric_split
        reason = "retired_frame_fraction"
    selected = max(0, min(len(previous), int(selected)))
    return {
        "exact_split": exact_split,
        "exact_overlap_tokens": best,
        "geometric_split": geometric_split,
        "selected_split": selected,
        "selection_reason": reason,
    }


class CausalLedgerBoundedKVWindowController(BoundedKVWindowController):
    """Keep the internal draft bounded while preserving delivered coverage."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.external_committed_ids: list[int] = []
        self.delivery_slides: list[dict[str, Any]] = []
        self.max_active_draft_tokens = 0

    def visible_ids(self, active_ids: list[int] | None = None) -> list[int]:
        active = (
            list(self.decoder.state.target_ids)
            if active_ids is None
            else [int(value) for value in active_ids]
        )
        return [*self.external_committed_ids, *active]

    def accept(
        self,
        latent: Any,
        is_final: bool = False,
        source_latent: Any | None = None,
        update_source: bool = True,
    ) -> Any:
        previous = list(self.decoder.state.target_ids)
        old_slide_count = len(self.slides)
        old_committed_count = len(self.external_committed_ids)
        old_delivery_count = len(self.delivery_slides)
        try:
            result = super().accept(
                latent,
                is_final=is_final,
                source_latent=source_latent,
                update_source=update_source,
            )
            if len(self.slides) == old_slide_count + 1:
                slide = self.slides[-1]
                current = list(self.decoder.state.target_ids)
                boundary = select_commit_boundary(
                    previous,
                    current,
                    retired_frames=int(slide["retired_frames"]),
                    retained_frames=int(slide["retained_frames"]),
                )
                moved = previous[: boundary["selected_split"]]
                self.external_committed_ids.extend(moved)
                receipt = {
                    "slide_index": slide["slide_index"],
                    "retired_event_ids": list(slide["retired_event_ids"]),
                    "previous_tokens": len(previous),
                    "current_tokens": len(current),
                    **boundary,
                    "committed_now": len(moved),
                    "external_committed_tokens": len(self.external_committed_ids),
                    "active_draft_tokens": len(current),
                    "semantic_coverage_certified": False,
                }
                self.delivery_slides.append(receipt)
            elif len(self.slides) != old_slide_count:
                raise RuntimeError("one decoder accept produced multiple slides")
            self.max_active_draft_tokens = max(
                self.max_active_draft_tokens,
                len(self.decoder.state.target_ids),
            )
            return result
        except BaseException:
            del self.external_committed_ids[old_committed_count:]
            del self.delivery_slides[old_delivery_count:]
            raise

    def rewrite_events(self, events: list[Any]) -> list[Any]:
        """Expose committed+active text without modifying private decoder state."""

        if not self.external_committed_ids:
            return events
        for event in events:
            if event.target_token_ids is None:
                raise RuntimeError("temporary-draft event omits target token IDs")
            visible = self.visible_ids(list(event.target_token_ids))
            text = self.decoder.decode(visible)
            event.target_prefix = text
            event.target_token_ids = visible
            # In this route target_delta already denotes a full replaceable
            # draft rather than an append-only delta, so keep the two fields
            # internally consistent after external-ledger composition.
            event.target_delta = text
            event.target_delta_token_ids = list(visible)
            trace = dict(event.target_eot_trace or {})
            trace["causal_delivery_ledger"] = {
                "external_committed_tokens": len(self.external_committed_ids),
                "active_draft_tokens": len(visible) - len(self.external_committed_ids),
                "delivery_slides": len(self.delivery_slides),
            }
            event.target_eot_trace = trace
        return events

    def stats(self) -> dict[str, Any]:
        values = super().stats()
        values.update(
            {
                "external_committed_tokens": len(self.external_committed_ids),
                "active_draft_tokens": len(self.decoder.state.target_ids),
                "max_active_draft_tokens": self.max_active_draft_tokens,
                "delivery_slides": list(self.delivery_slides),
                "delivery_ledger_external_to_model_state": True,
                "semantic_coverage_certified": False,
            }
        )
        return values

    def uninstall(self, translator: Any) -> None:
        """Restore the translator methods before resetting a session."""

        translator.decoder.accept = self.original_accept
        translator.accept_pcm = self.original_accept_pcm


def install_causal_window(
    translator: Any,
    *,
    window_blocks: int = 12,
    retire_blocks: int = 2,
    window_adapter: str | None = None,
    source_frame_cap: int = 336,
) -> CausalLedgerBoundedKVWindowController:
    """Install the lazy physical window and causal delivery ledger together."""

    bind_bounded_prefix_frontend(
        translator.frontend,
        window_blocks=window_blocks,
        retire_blocks=retire_blocks,
    )
    controller = CausalLedgerBoundedKVWindowController(
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
    original_accept_pcm = translator.accept_pcm
    controller.original_accept_pcm = original_accept_pcm

    def accept_pcm(_translator, pcm: bytes, final: bool = False):
        return controller.rewrite_events(original_accept_pcm(pcm, final=final))

    translator.accept_pcm = MethodType(accept_pcm, translator)
    translator._bounded_window_controller = controller
    return controller
