"""Load the released SFT policy with the exact PDO trainable scope."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch

from pdo_s2tt.history import PrivateHistoryAdapter
from pdo_s2tt.languages import instructions
from pdo_s2tt.revisions import QWEN3_ASR_REVISION
from .policy import install_masked_history


SFT_CHECKPOINT_FORMAT = "pdo-s2tt-sft-v1"


def load_sft_checkpoint(path: str | Path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {
        "format", "model_id", "config", "llm_adapter", "history",
        "history_architecture",
    }
    if set(payload) != required or payload["format"] != SFT_CHECKPOINT_FORMAT:
        raise ValueError("expected the released PDO SFT checkpoint")
    if len(payload["llm_adapter"]) != 368 or len(payload["history"]) != 24:
        raise ValueError("the SFT checkpoint tensor inventory is incomplete")
    return payload


def _build_history(llm, payload: dict) -> PrivateHistoryAdapter:
    architecture = payload["history_architecture"]
    blocks = architecture["blocks"]
    if not blocks or any(block != blocks[0] for block in blocks):
        raise ValueError("unsupported history architecture")
    adapter = PrivateHistoryAdapter(
        architecture["layer_names"], **blocks[0], init_seed=architecture["init_seed"],
    ).to(device=llm.get_input_embeddings().weight.device)
    adapter.load_state_dict(payload["history"], strict=True)
    adapter.attach(llm)
    install_masked_history(adapter)
    return adapter


@contextmanager
def _attach_before_first_cache(payload: dict):
    from streaming_s2tt.native_true_dual_stream import NativeTrueDualQwenDecoder

    original = NativeTrueDualQwenDecoder.__init__
    created = []

    def initialize(decoder, *args, **kwargs):
        original(decoder, *args, **kwargs)
        if decoder.state is not None:
            raise ValueError("history must be attached before the initial cache")
        decoder._pdo_history_adapter = _build_history(decoder.llm, payload)
        created.append(decoder)

    NativeTrueDualQwenDecoder.__init__ = initialize
    try:
        yield created
    finally:
        NativeTrueDualQwenDecoder.__init__ = original


def set_language(translator, code: str) -> None:
    decoder = translator.decoder
    if decoder.state is not None:
        decoder.state = None
    target, system, user = instructions(code)
    translator.source_language = "English"
    translator.target_language = target
    decoder.target_system_instruction = system
    decoder.target_user_instruction = user
    decoder.target_user_instruction_ids = translator.tokenizer(
        user, add_special_tokens=False,
    )["input_ids"]


def _is_acoustic(name: str) -> bool:
    return name.startswith("audio_tower.") or ".audio_tower." in name


def _cast_text_to_fp32(llm) -> None:
    """Match the paper run: FP32 text decoder, unchanged acoustic tower."""
    with torch.no_grad():
        for name, parameter in llm.named_parameters():
            if not _is_acoustic(name) and parameter.is_floating_point():
                parameter.data = parameter.data.float()
        for name, buffer in list(llm.named_buffers()):
            if not _is_acoustic(name) and buffer.is_floating_point():
                parent, _, leaf = name.rpartition(".")
                owner = llm.get_submodule(parent) if parent else llm
                setattr(owner, leaf, buffer.float())
    if llm.get_input_embeddings().weight.dtype != torch.float32:
        raise RuntimeError("text decoder was not converted to FP32")


def _disable_dropout(module) -> None:
    kinds = (
        torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d,
        torch.nn.Dropout3d, torch.nn.AlphaDropout, torch.nn.FeatureAlphaDropout,
    )
    for child in module.modules():
        if isinstance(child, kinds):
            child.eval()


class TrainingModel:
    """One resident SFT policy and its text-LoRA/history optimizer."""

    def __init__(
        self, checkpoint: str | Path, base_model: str | None = None,
        device: str = "cuda:0", learning_rate: float = 1e-6,
    ) -> None:
        from qwen_asr import Qwen3ASRModel
        from streaming_s2tt.qwen_asr_streaming_runtime import PersistentQwenASRTranslator

        self.payload = load_sft_checkpoint(checkpoint)
        local_base = Path("checkpoints/Qwen3-ASR-1.7B")
        model_id = base_model or (
            str(local_base) if local_base.is_dir() else self.payload["model_id"]
        )
        load_options = {
            "dtype": torch.bfloat16,
            "device_map": device,
            "max_new_tokens": 64,
        }
        if not Path(model_id).is_dir():
            load_options["revision"] = QWEN3_ASR_REVISION
        runtime = Qwen3ASRModel.from_pretrained(model_id, **load_options)
        with _attach_before_first_cache(self.payload) as created:
            self.translator = PersistentQwenASRTranslator(
                runtime, self.payload,
                max_source_tokens=48,
                max_target_tokens=128,
                chunk_seconds=2.0,
                temperature=0.2,
                device=device,
                target_latent_mode="full",
                target_source_mode="none",
                target_output_lag_chunks=0,
                target_source_lag_chunks=0,
                source_decode=False,
                source_language="English",
                target_language="Chinese",
            )
        if len(created) != 1:
            raise RuntimeError("history adapter was not attached exactly once")
        self.decoder = self.translator.decoder
        self.llm = self.translator.llm
        self.tokenizer = self.translator.tokenizer
        self.history = self.decoder._pdo_history_adapter
        _cast_text_to_fp32(self.llm)

        self.named_parameters = []
        for name, parameter in self.llm.named_parameters():
            trainable = ".model.layers." in name and ".lora_" in name
            parameter.requires_grad_(trainable)
            parameter.grad = None
            if trainable:
                self.named_parameters.append((name, parameter))
        self.history.float().requires_grad_(True)
        self.named_parameters.extend(
            (f"history.{name}", parameter)
            for name, parameter in self.history.named_parameters()
        )
        if not self.named_parameters:
            raise RuntimeError("no trainable PDO parameters were found")
        self.parameters = [parameter for _, parameter in self.named_parameters]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": [p for name, p in self.named_parameters if not name.startswith("history.")]},
                {"params": [p for name, p in self.named_parameters if name.startswith("history.")]},
            ],
            lr=learning_rate,
            weight_decay=0.01,
        )
        self.llm.eval()
        self.history.eval()
        _disable_dropout(self.llm)

    def prepare(self, language: str) -> None:
        set_language(self.translator, language)
        self.translator.reset()

    def train_mode(self) -> None:
        self.llm.train()
        self.history.train()
        _disable_dropout(self.llm)

    def eval_mode(self) -> None:
        self.llm.eval()
        self.history.eval()

    def load_training_state(self, payload: dict) -> tuple[int, int]:
        """Restore a public training-state checkpoint before distributed init."""
        from peft import set_peft_model_state_dict
        if payload.get("format") != "pdo-s2tt-training-state-v1":
            raise ValueError("unsupported PDO training-state checkpoint")
        policy = payload["policy"]
        set_peft_model_state_dict(self.llm, policy["llm_adapter"])
        self.history.load_state_dict(policy["history"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        return int(payload["rounds"]), int(payload["adam_updates"])

    def state_dict(self, *, rounds: int, adam_updates: int) -> dict:
        from peft import get_peft_model_state_dict
        return {
            "format": "pdo-s2tt-inference-v1",
            "model_id": self.payload["model_id"],
            "rounds": rounds,
            "adam_updates": adam_updates,
            "config": self.payload["config"],
            "llm_adapter": {
                name: value.detach().cpu()
                for name, value in get_peft_model_state_dict(self.llm).items()
            },
            "history": {
                name: value.detach().cpu()
                for name, value in self.history.state_dict().items()
            },
            "history_architecture": self.payload["history_architecture"],
            "source_artifacts": {
                "initialization": "released PDO SFT checkpoint",
                "training": "public PDO G=4 full-return recipe",
            },
        }
