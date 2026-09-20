"""LoRA target selection for the native Qwen streaming route."""

from __future__ import annotations

from collections.abc import Iterable


def parse_lora_targets(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def resolve_lora_targets(model, requested: str | Iterable[str], scope: str) -> list[str]:
    """Resolve suffixes to exact language-model module names when requested.

    Qwen's thinker contains both ``audio_tower.layers`` and ``model.layers``.
    Selecting bare ``q_proj`` names adapts both branches, so the streaming
    latent must be extracted with the adapted audio tower at inference. The
    ``thinker`` scope restricts adapters to the language-model layers and keeps
    the Audio Tower identical between feature preparation and runtime.
    """
    targets = parse_lora_targets(requested)
    if scope == "all":
        return targets
    if scope != "thinker":
        raise ValueError(f"unsupported LoRA scope: {scope}")
    suffixes = {
        target.rsplit(".", 1)[-1]
        for target in targets
        if target.rsplit(".", 1)[-1]
    }
    # PEFT wraps the base model after the student adapter is installed, so
    # the same thinker modules may be named either ``model.layers.*`` or
    # ``base_model.model.model.layers.*``.  Match the stable path segment and
    # keep Audio Tower modules out of the thinker-only scope.
    resolved = [
        name
        for name, _ in model.named_modules()
        if "model.layers." in name
        and "audio_tower" not in name
        and name.rsplit(".", 1)[-1] in suffixes
    ]
    if not resolved:
        raise ValueError("no thinker LoRA targets resolved")
    return resolved


def filter_adapter_state(state: dict, scope: str) -> dict:
    if scope == "all":
        return state
    return {key: value for key, value in state.items() if "audio_tower." not in key}
