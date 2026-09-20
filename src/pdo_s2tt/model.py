"""Load the released PDO policy and run revision-capable streaming inference."""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch

from .history import PrivateHistoryAdapter, install_private_runtime
from .languages import instructions


CHECKPOINT_FORMAT = "pdo-s2tt-inference-v1"


def load_checkpoint(path: str | Path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {
        "format", "model_id", "rounds", "adam_updates", "config",
        "llm_adapter", "history", "history_architecture", "source_artifacts",
    }
    if set(payload) != required:
        raise ValueError("checkpoint fields do not match the public inference format")
    if payload["format"] != CHECKPOINT_FORMAT:
        raise ValueError("expected the released PDO checkpoint format")
    if len(payload["llm_adapter"]) != 368 or len(payload["history"]) != 24:
        raise ValueError("checkpoint tensor inventory is incomplete")
    return payload


def _history_adapter(llm, payload: dict) -> PrivateHistoryAdapter:
    architecture = payload["history_architecture"]
    blocks = architecture["blocks"]
    if not blocks or any(block != blocks[0] for block in blocks):
        raise ValueError("unsupported non-uniform history architecture")
    adapter = PrivateHistoryAdapter(
        architecture["layer_names"], **blocks[0],
        init_seed=architecture["init_seed"],
    ).to(device=llm.get_input_embeddings().weight.device)
    adapter.load_state_dict(payload["history"], strict=True)
    adapter.requires_grad_(False).eval().attach(llm)
    return adapter


@contextmanager
def _install_before_initial_cache(payload: dict, target_code: str):
    from streaming_s2tt.native_true_dual_stream import NativeTrueDualQwenDecoder

    original = NativeTrueDualQwenDecoder.__init__
    created = []
    target, system, user = instructions(target_code)

    def initialize(decoder, *args, **kwargs):
        original(decoder, *args, **kwargs)
        if decoder.state is not None:
            raise ValueError("history must be installed before the initial cache")
        decoder.target_system_instruction = system
        decoder.target_user_instruction = user
        decoder.target_user_instruction_ids = decoder.tokenizer(
            user, add_special_tokens=False
        )["input_ids"]
        adapter = _history_adapter(decoder.llm, payload)
        stats, remove = install_private_runtime(decoder, adapter)
        decoder._pdo_history_adapter = adapter
        decoder._pdo_history_stats = stats
        decoder._pdo_history_remove = remove
        decoder._pdo_target_language = target
        created.append(decoder)

    NativeTrueDualQwenDecoder.__init__ = initialize
    try:
        yield created
    finally:
        NativeTrueDualQwenDecoder.__init__ = original


class PDOS2TT:
    """One loaded English-to-X PDO streaming translator."""

    def __init__(self, checkpoint: str | Path, target_language: str,
                 base_model: str | None = None,
                 device: str = "cuda:0") -> None:
        from qwen_asr import Qwen3ASRModel
        from streaming_s2tt.qwen_asr_streaming_runtime import PersistentQwenASRTranslator

        self.payload = load_checkpoint(checkpoint)
        target, _, _ = instructions(target_language)
        local_base = Path("checkpoints/Qwen3-ASR-1.7B")
        model_id = base_model or (
            str(local_base) if local_base.is_dir() else self.payload["model_id"]
        )
        runtime = Qwen3ASRModel.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device, max_new_tokens=64,
        )
        with _install_before_initial_cache(self.payload, target_language) as created:
            self.translator = PersistentQwenASRTranslator(
                runtime, self.payload, max_source_tokens=48, max_target_tokens=128,
                chunk_seconds=2.0, temperature=0.0, device=device,
                target_latent_mode="full", target_source_mode="none",
                target_output_lag_chunks=0, target_source_lag_chunks=0,
                source_decode=False, source_language="English", target_language=target,
            )
        if len(created) != 1 or created[0] is not self.translator.decoder:
            raise RuntimeError("the history-conditioned decoder was not installed exactly once")

    def reset(self) -> None:
        self.translator.reset()

    def accept_pcm(self, pcm: bytes, final: bool = False):
        return self.translator.accept_pcm(pcm, final=final)

    def stream(self, packets: Iterator[tuple[bytes, bool]]):
        for pcm, final in packets:
            yield from self.accept_pcm(pcm, final=final)
