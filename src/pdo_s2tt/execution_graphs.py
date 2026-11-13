"""Opt-in CUDA-graph resources for long-lived PDO inference services.

This module accelerates repeated query-one decoder operations. It contains the
resource lifecycle, fixed-shape operator banks, and a decoder-layer resource
that covers the two static regions around native cache mutation and attention.
Every supported shape is prewarmed and sealed before request processing.

Unknown shapes, changed parameters/adapters, gradients, CPU tensors, and
concurrent calls always use the original eager method.  No graph is captured on
the request path after :meth:`CudaGraphModuleBank.seal`.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import threading
import time
from types import MethodType
from typing import Any

import torch
from torch import Tensor, nn

from .execution_signatures import SignatureRegistry
from .execution_decoder_guards import DecoderGuardPlan


Predicate = Callable[[], bool]


def _module_signature(module: nn.Module) -> tuple:
    """Return the state that makes a captured call safe to replay."""

    parameters = tuple(
        (name, id(value), value._version, value.data_ptr(), value.dtype, value.device)
        for name, value in module.named_parameters()
    )
    buffers = tuple(
        (name, id(value), value._version, value.data_ptr(), value.dtype, value.device)
        for name, value in module.named_buffers()
    )
    def scaling_state(child: nn.Module):
        value = getattr(child, "scaling", {})
        if isinstance(value, Mapping):
            return tuple(sorted(value.items()))
        return value

    route = tuple(
        (
            name,
            tuple(getattr(child, "active_adapters", ())),
            bool(getattr(child, "disable_adapters", False)),
            tuple(getattr(child, "merged_adapters", ())),
            scaling_state(child),
            bool(child.training),
        )
        for name, child in module.named_modules()
    )
    return parameters, buffers, route


@dataclass
class _PatchedMethod:
    module: nn.Module
    had_instance_value: bool
    instance_value: Any
    bound_original: Callable

    @classmethod
    def capture(cls, module: nn.Module) -> "_PatchedMethod":
        return cls(
            module=module,
            had_instance_value="forward" in module.__dict__,
            instance_value=module.__dict__.get("forward"),
            bound_original=module.forward,
        )

    def restore(self) -> None:
        if self.had_instance_value:
            self.module.forward = self.instance_value
        elif "forward" in self.module.__dict__:
            del self.module.forward


class CudaGraphModuleBank:
    """Prewarmed CUDA graphs for tensor-to-tensor module calls.

    ``rows`` must be a stable list of unique ``(name, module)`` pairs.  The
    wrapper is installed immediately but is an eager no-op until matching CUDA
    samples are explicitly supplied to :meth:`prewarm`.
    """

    def __init__(
        self,
        rows: Iterable[tuple[str, nn.Module]],
        predicate: Predicate,
        *,
        label: str,
        borrow_output: bool = False,
        signature_provider: Callable[[nn.Module], tuple] = _module_signature,
    ) -> None:
        materialized = list(rows)
        if not materialized:
            raise ValueError(f"{label} requires at least one module")
        names = [name for name, _ in materialized]
        modules = [module for _, module in materialized]
        if len(set(names)) != len(names) or len({id(module) for module in modules}) != len(modules):
            raise ValueError(f"{label} names and modules must be unique")

        self.label = str(label)
        self._signature = signature_provider
        self.borrow_output = bool(borrow_output)
        self.names = tuple(names)
        self.modules = tuple(modules)
        self.predicate = predicate
        self._owner_thread = threading.get_ident()
        self._patches = tuple(_PatchedMethod.capture(module) for module in modules)
        self._slots: dict[tuple[int, tuple[int, ...], tuple[int, ...]], dict[str, Any]] = {}
        self._sealed = False
        self._closed = False
        self._busy = False
        self.stats = {
            "captures": 0,
            "replays": 0,
            "exact_first_use": 0,
            "fallbacks": 0,
            "capture_seconds": 0.0,
            "borrowed_outputs": 0,
            "cloned_outputs": 0,
        }

        try:
            for index, module in enumerate(self.modules):
                def forward(_module, value, _index=index):
                    return self._call(_index, value)

                module.forward = MethodType(forward, module)
        except Exception:
            for patch in self._patches:
                patch.restore()
            raise

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def closed(self) -> bool:
        return self._closed

    def _capture_eligible(self, index: int, value: Tensor) -> bool:
        module = self.modules[index]
        return (
            not module.training
            and not torch.is_grad_enabled()
            and value.is_cuda
            and value.dtype == torch.bfloat16
            and value.is_contiguous()
            and value.ndim >= 2
            and value.device == next(module.parameters()).device
        )

    @staticmethod
    def _slot_key(index: int, value: Tensor) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        return index, tuple(value.shape), tuple(value.stride())

    def prewarm(self, samples: Mapping[str, Tensor]) -> None:
        """Capture every supplied slot before serving requests.

        A sample must be supplied for every wrapped module.  Partial prewarming
        is rejected so the service cannot silently capture more graphs later.
        """

        if self._closed or self._sealed:
            raise RuntimeError(f"{self.label} cannot prewarm after seal/close")
        if threading.get_ident() != self._owner_thread or self._busy:
            raise RuntimeError(f"{self.label} prewarm requires its owner thread")
        if set(samples) != set(self.names):
            missing = sorted(set(self.names) - set(samples))
            extra = sorted(set(samples) - set(self.names))
            raise ValueError(f"{self.label} sample mismatch: missing={missing}, extra={extra}")

        for index, name in enumerate(self.names):
            value = samples[name]
            with torch.inference_mode():
                if not self._capture_eligible(index, value):
                    raise ValueError(f"{self.label}:{name} is not a supported CUDA BF16 sample")
                self._capture(index, value)

    def _capture(self, index: int, value: Tensor) -> None:
        key = self._slot_key(index, value)
        if key in self._slots:
            raise RuntimeError(
                f"{self.label}:{self.names[index]} shape/stride was already captured"
            )
        module = self.modules[index]
        native = self._patches[index].bound_original
        signature = self._signature(module)
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(value.device).clone()
        static_input = value.detach().clone()
        stream = torch.cuda.Stream(device=value.device)
        stream.wait_stream(torch.cuda.current_stream(value.device))
        begin = time.perf_counter()
        with torch.cuda.stream(stream):
            for _ in range(3):
                native(static_input)
        torch.cuda.current_stream(value.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        # Use the side stream only for warmup,
        # then capture on the caller's current stream.  Capturing on the warmup
        # stream may select a different BF16 GEMM workspace/algorithm and fail
        # the byte-exact eager comparison.
        with torch.cuda.graph(graph):
            static_output = native(static_input)
        if not isinstance(static_output, Tensor):
            raise TypeError(f"{self.label} only supports tensor outputs")
        if not torch.equal(cpu_rng, torch.get_rng_state()):
            raise RuntimeError(f"{self.label} capture changed the CPU RNG")
        if not torch.equal(cuda_rng, torch.cuda.get_rng_state(value.device)):
            raise RuntimeError(f"{self.label} capture changed the CUDA RNG")
        if signature != self._signature(module):
            raise RuntimeError(f"{self.label} module changed during capture")
        graph.replay()
        replay_output = static_output.clone()
        reference = native(value)
        if not torch.equal(replay_output, reference):
            raise RuntimeError(f"{self.label} prewarm exact-output gate failed")
        self._slots[key] = {
            "input": static_input,
            "output": static_output,
            "graph": graph,
            "shape": tuple(value.shape),
            "stride": tuple(value.stride()),
            "signature": signature,
            "verified": False,
        }
        self.stats["captures"] += 1
        self.stats["capture_seconds"] += time.perf_counter() - begin

    def seal(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self.label} is closed")
        covered = {key[0] for key in self._slots}
        if covered != set(range(len(self.modules))):
            raise RuntimeError(
                f"{self.label} cannot seal with {len(covered)}/{len(self.modules)} modules"
            )
        self._sealed = True

    def _call(self, index: int, value: Tensor):
        module = self.modules[index]
        native = self._patches[index].bound_original
        slot = self._slots.get(self._slot_key(index, value))
        eligible = (
            not self._closed
            and self._sealed
            and self.predicate()
            and threading.get_ident() == self._owner_thread
            and not self._busy
            and not torch.is_grad_enabled()
            and value.is_cuda
            and value.dtype == torch.bfloat16
            and value.is_contiguous()
            and slot is not None
            and tuple(value.shape) == slot["shape"]
            and tuple(value.stride()) == slot["stride"]
            and self._signature(module) == slot["signature"]
        )
        if not eligible:
            self.stats["fallbacks"] += 1
            return native(value)

        self._busy = True
        try:
            slot["input"].copy_(value)
            slot["graph"].replay()
            output = slot["output"] if self.borrow_output else slot["output"].clone()
            self.stats["replays"] += 1
            if not slot["verified"]:
                reference = native(value)
                if not torch.equal(output, reference):
                    raise RuntimeError(f"{self.label} first-use exact-output gate failed")
                slot["verified"] = True
                self.stats["exact_first_use"] += 1
            if self.borrow_output:
                self.stats["borrowed_outputs"] += 1
            else:
                self.stats["cloned_outputs"] += 1
            return output
        finally:
            self._busy = False

    def retention(self) -> dict[str, int | bool]:
        explicit = 0
        for slot in self._slots.values():
            explicit += slot["input"].numel() * slot["input"].element_size()
            explicit += slot["output"].numel() * slot["output"].element_size()
        return {
            "slots": len(self._slots),
            "explicit_tensor_bytes": explicit,
            "graph_allocator_workspace_included": False,
            "borrowed_output": self.borrow_output,
            "borrowed_output_contract": (
                "single-threaded synchronous consumer"
                if self.borrow_output
                else "owned clone"
            ),
        }

    def close(self) -> None:
        if self._busy:
            raise RuntimeError(f"{self.label} cannot close during replay")
        if self._closed:
            return
        for patch in self._patches:
            patch.restore()
        self._slots.clear()
        self._closed = True

    def __enter__(self) -> "CudaGraphModuleBank":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _qkv_signature(module: nn.Module, signature_provider: Callable[[nn.Module], tuple] = _module_signature) -> tuple:
    return (
        signature_provider(module.q_proj),
        signature_provider(module.k_proj),
        signature_provider(module.v_proj),
        signature_provider(module.q_norm),
        signature_provider(module.k_norm),
        int(module.layer_idx),
        int(module.head_dim),
        float(module.scaling),
        str(module.config._attn_implementation),
    )


def _qkv_prelude(module: nn.Module, value: Tensor, cos: Tensor, sin: Tensor):
    from qwen_asr.core.transformers_backend import modeling_qwen3_asr as qwen

    hidden_shape = (*value.shape[:-1], -1, module.head_dim)
    query = module.q_norm(module.q_proj(value).view(hidden_shape)).transpose(1, 2)
    key = module.k_norm(module.k_proj(value).view(hidden_shape)).transpose(1, 2)
    value_states = module.v_proj(value).view(hidden_shape).transpose(1, 2)
    query, key = qwen.apply_rotary_pos_emb(query, key, cos, sin)
    return query, key, value_states


class CudaGraphQKVBank:
    """Prewarmed QKV+norm+RoPE graphs with native cache and attention.

    Only the prefix of ``Qwen3ASRTextAttention.forward`` is captured. Cache
    mutation, attention-mask handling, SDPA/eager attention, and ``o_proj`` stay
    on the installed Qwen implementation's ordinary path.
    """

    def __init__(
        self,
        rows: Iterable[tuple[str, nn.Module]],
        predicate: Predicate,
        *,
        borrow_outputs: bool = False,
        signature_provider: Callable[[nn.Module], tuple] = _module_signature,
    ) -> None:
        materialized = list(rows)
        if not materialized:
            raise ValueError("QKV graph bank requires at least one attention module")
        self.names = tuple(name for name, _ in materialized)
        self.modules = tuple(module for _, module in materialized)
        if len(set(self.names)) != len(self.names) or len({id(m) for m in self.modules}) != len(self.modules):
            raise ValueError("QKV names and modules must be unique")
        self.predicate = predicate
        self.borrow_outputs = bool(borrow_outputs)
        self._signature = signature_provider
        self._owner_thread = threading.get_ident()
        self._patches = tuple(_PatchedMethod.capture(module) for module in self.modules)
        self._slots: dict[
            tuple[
                int,
                tuple[int, ...],
                tuple[int, ...],
                tuple[int, ...],
                tuple[int, ...],
                tuple[int, ...],
                tuple[int, ...],
            ],
            dict[str, Any],
        ] = {}
        self._sealed = False
        self._closed = False
        self._busy = False
        self.stats = {
            "captures": 0,
            "replays": 0,
            "exact_first_use": 0,
            "fallbacks": 0,
            "capture_seconds": 0.0,
            "borrowed_calls": 0,
            "cloned_calls": 0,
            "cache_update_calls": 0,
            "cache_owner_checks": 0,
        }
        try:
            for index, module in enumerate(self.modules):
                def forward(
                    _module,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    past_key_values=None,
                    cache_position=None,
                    _index=index,
                    **kwargs,
                ):
                    return self._call(
                        _index,
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                        past_key_values,
                        cache_position,
                        kwargs,
                    )

                module.forward = MethodType(forward, module)
        except Exception:
            for patch in self._patches:
                patch.restore()
            raise

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def closed(self) -> bool:
        return self._closed

    def prewarm(
        self,
        samples: Mapping[str, tuple[Tensor, Tensor, Tensor]],
    ) -> None:
        if self._closed or self._sealed:
            raise RuntimeError("QKV cannot prewarm after seal/close")
        if threading.get_ident() != self._owner_thread or self._busy:
            raise RuntimeError("QKV prewarm requires its owner thread")
        if set(samples) != set(self.names):
            missing = sorted(set(self.names) - set(samples))
            extra = sorted(set(samples) - set(self.names))
            raise ValueError(f"QKV sample mismatch: missing={missing}, extra={extra}")
        for index, name in enumerate(self.names):
            value, cos, sin = samples[name]
            with torch.inference_mode():
                if not self._capture_eligible(index, value, cos, sin):
                    raise ValueError(f"QKV:{name} is not a supported CUDA BF16 sample")
                self._capture(index, value, cos, sin)

    def _capture_eligible(
        self,
        index: int,
        value: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> bool:
        module = self.modules[index]
        return (
            not module.training
            and not torch.is_grad_enabled()
            and value.is_cuda
            and value.dtype == torch.bfloat16
            and value.is_contiguous()
            and value.ndim == 3
            and value.shape[0] == 1
            # The public service path supports query-one replay only.
            # Larger query shapes were measured separately: they add material
            # startup/capture cost for negligible end-to-end gain, so they
            # remain on the exact eager fallback.
            and value.shape[1] == 1
            and value.shape[2] == 2048
            and cos.is_cuda
            and sin.is_cuda
            and cos.dtype == value.dtype
            and sin.dtype == value.dtype
            and cos.is_contiguous()
            and sin.is_contiguous()
            and cos.ndim == 3
            and sin.ndim == 3
            and tuple(cos.shape) == (1, value.shape[1], module.head_dim)
            and tuple(sin.shape) == tuple(cos.shape)
            and cos.device == value.device
            and sin.device == value.device
            and module.q_proj.weight.device == value.device
        )

    @staticmethod
    def _slot_key(
        index: int,
        value: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> tuple[
        int,
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]:
        return (
            index,
            tuple(value.shape),
            tuple(value.stride()),
            tuple(cos.shape),
            tuple(cos.stride()),
            tuple(sin.shape),
            tuple(sin.stride()),
        )

    def _capture(
        self,
        index: int,
        value: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> None:
        key = self._slot_key(index, value, cos, sin)
        if key in self._slots:
            raise RuntimeError(
                f"QKV:{self.names[index]} shape/stride was already captured"
            )
        module = self.modules[index]
        signature = _qkv_signature(module, self._signature)
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(value.device).clone()
        static_value = value.detach().clone()
        static_cos = cos.detach().clone()
        static_sin = sin.detach().clone()
        stream = torch.cuda.Stream(device=value.device)
        stream.wait_stream(torch.cuda.current_stream(value.device))
        begin = time.perf_counter()
        with torch.cuda.stream(stream):
            for _ in range(3):
                _qkv_prelude(module, static_value, static_cos, static_sin)
        torch.cuda.current_stream(value.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = _qkv_prelude(module, static_value, static_cos, static_sin)
        if not torch.equal(cpu_rng, torch.get_rng_state()):
            raise RuntimeError("QKV capture changed the CPU RNG")
        if not torch.equal(cuda_rng, torch.cuda.get_rng_state(value.device)):
            raise RuntimeError("QKV capture changed the CUDA RNG")
        if signature != _qkv_signature(module, self._signature):
            raise RuntimeError("QKV module changed during capture")
        graph.replay()
        replay = tuple(output.clone() for output in outputs)
        reference = _qkv_prelude(module, value, cos, sin)
        if not all(torch.equal(left, right) for left, right in zip(replay, reference)):
            raise RuntimeError("QKV prewarm exact-output gate failed")
        self._slots[key] = {
            "input": static_value,
            "cos": static_cos,
            "sin": static_sin,
            "outputs": outputs,
            "graph": graph,
            "input_shape": tuple(value.shape),
            "input_stride": tuple(value.stride()),
            "cos_shape": tuple(cos.shape),
            "cos_stride": tuple(cos.stride()),
            "sin_shape": tuple(sin.shape),
            "sin_stride": tuple(sin.stride()),
            "signature": signature,
            "verified": False,
        }
        self.stats["captures"] += 1
        self.stats["capture_seconds"] += time.perf_counter() - begin

    def seal(self) -> None:
        if self._closed:
            raise RuntimeError("QKV graph bank is closed")
        covered = {key[0] for key in self._slots}
        if covered != set(range(len(self.modules))):
            raise RuntimeError(
                f"QKV cannot seal with {len(covered)}/{len(self.modules)} modules"
            )
        self._sealed = True

    def _call(
        self,
        index: int,
        value: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask,
        past_key_values,
        cache_position,
        kwargs: dict[str, Any],
    ):
        module = self.modules[index]
        native = self._patches[index].bound_original
        cos, sin = position_embeddings
        slot = self._slots.get(self._slot_key(index, value, cos, sin))
        eligible = (
            not self._closed
            and self._sealed
            and self.predicate()
            and threading.get_ident() == self._owner_thread
            and not self._busy
            and not torch.is_grad_enabled()
            and value.is_cuda
            and value.dtype == torch.bfloat16
            and value.is_contiguous()
            and slot is not None
            and tuple(value.shape) == slot["input_shape"]
            and tuple(value.stride()) == slot["input_stride"]
            and tuple(cos.shape) == slot["cos_shape"]
            and tuple(cos.stride()) == slot["cos_stride"]
            and tuple(sin.shape) == slot["sin_shape"]
            and tuple(sin.stride()) == slot["sin_stride"]
            and cos.dtype == slot["cos"].dtype
            and sin.dtype == slot["sin"].dtype
            and cos.device == value.device
            and sin.device == value.device
            and "past_key_value" not in kwargs
            and _qkv_signature(module, self._signature) == slot["signature"]
        )
        if not eligible:
            self.stats["fallbacks"] += 1
            return native(
                value,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        self._busy = True
        try:
            slot["input"].copy_(value)
            slot["cos"].copy_(cos)
            slot["sin"].copy_(sin)
            slot["graph"].replay()
            if self.borrow_outputs:
                query, key, value_states = slot["outputs"]
                borrowed_key_ptr = key.data_ptr()
                borrowed_value_ptr = value_states.data_ptr()
                self.stats["borrowed_calls"] += 1
            else:
                query, key, value_states = (
                    output.clone() for output in slot["outputs"]
                )
                borrowed_key_ptr = borrowed_value_ptr = None
                self.stats["cloned_calls"] += 1
            self.stats["replays"] += 1
            if not slot["verified"]:
                reference = _qkv_prelude(module, value, cos, sin)
                if not all(
                    torch.equal(left, right)
                    for left, right in zip((query, key, value_states), reference)
                ):
                    raise RuntimeError("QKV first-use exact-output gate failed")
                slot["verified"] = True
                self.stats["exact_first_use"] += 1

            from qwen_asr.core.transformers_backend import modeling_qwen3_asr as qwen

            if past_key_values is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key, value_states = past_key_values.update(
                    key, value_states, module.layer_idx, cache_kwargs
                )
                self.stats["cache_update_calls"] += 1
                if self.borrow_outputs:
                    if (
                        key.data_ptr() == borrowed_key_ptr
                        or value_states.data_ptr() == borrowed_value_ptr
                    ):
                        raise RuntimeError(
                            "QKV cache retained a borrowed graph output"
                        )
                    self.stats["cache_owner_checks"] += 2
            attention = qwen.eager_attention_forward
            if module.config._attn_implementation != "eager":
                attention = qwen.ALL_ATTENTION_FUNCTIONS[
                    module.config._attn_implementation
                ]
            output, weights = attention(
                module,
                query,
                key,
                value_states,
                attention_mask,
                dropout=0.0 if not module.training else module.attention_dropout,
                scaling=module.scaling,
                **kwargs,
            )
            output = module.o_proj(output.reshape(*value.shape[:-1], -1).contiguous())
            return output, weights
        finally:
            self._busy = False

    def retention(self) -> dict[str, int | bool]:
        explicit = 0
        for slot in self._slots.values():
            tensors = (
                slot["input"],
                slot["cos"],
                slot["sin"],
                *slot["outputs"],
            )
            explicit += sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        return {
            "slots": len(self._slots),
            "explicit_tensor_bytes": explicit,
            "graph_allocator_workspace_included": False,
            "native_cache_mask_attention_and_output": True,
            "borrowed_outputs": self.borrow_outputs,
            "borrowed_output_contract": (
                "single-threaded synchronous attention with independent cache K/V"
                if self.borrow_outputs
                else "owned clones"
            ),
        }

    def close(self) -> None:
        if self._busy:
            raise RuntimeError("QKV cannot close during replay")
        if self._closed:
            return
        for patch in self._patches:
            patch.restore()
        self._slots.clear()
        self._closed = True


def _decoder_prelude(
    layer: nn.Module,
    hidden_states: Tensor,
    cos: Tensor,
    sin: Tensor,
):
    """Run input RMSNorm plus the native QKV/normalization/RoPE prefix."""

    normalized = layer.input_layernorm(hidden_states)
    return _qkv_prelude(layer.self_attn, normalized, cos, sin)


def _decoder_tail(
    layer: nn.Module,
    residual: Tensor,
    attention_output: Tensor,
) -> Tensor:
    """Run the fixed-shape decoder tail in native operation order."""

    hidden_states = residual + layer.self_attn.o_proj(attention_output)
    tail_residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    return tail_residual + hidden_states


def _forward_token(module: nn.Module) -> tuple[str, int]:
    """Identity token for the forward implementation used by a child call."""

    if "forward" in module.__dict__:
        return "instance", id(module.__dict__["forward"])
    return "type", id(getattr(type(module), "forward", None))


def _decoder_contract(layer: nn.Module) -> tuple:
    """Non-parameter state used by the fused fixed-shape layer regions."""

    from qwen_asr.core.transformers_backend import modeling_qwen3_asr as qwen

    attention = layer.self_attn
    descendants = tuple(layer.modules())[1:]
    implementation = str(attention.config._attn_implementation)
    # Attention executes outside the captured graphs and is looked up afresh
    # in _call. The required DecoderAudioObserver replaces the registry entry
    # during a stream, so its identity is not captured fixed-region state.
    return (
        id(layer.input_layernorm),
        id(attention),
        id(attention.o_proj),
        id(layer.post_attention_layernorm),
        id(layer.mlp),
        tuple((id(child), _forward_token(child)) for child in descendants),
        id(qwen.apply_rotary_pos_emb),
        id(qwen.eager_attention_forward),
        int(attention.layer_idx),
        int(attention.head_dim),
        float(attention.scaling),
        float(attention.attention_dropout),
        implementation,
        float(layer.input_layernorm.variance_epsilon),
        float(layer.post_attention_layernorm.variance_epsilon),
        id(layer.mlp.act_fn),
    )


def _decoder_child_hooks_clear(layer: nn.Module) -> bool:
    """The fused path bypasses child ``Module.__call__`` hook dispatch."""

    from torch.nn.modules import module as module_runtime

    if module_runtime._global_forward_pre_hooks or module_runtime._global_forward_hooks:
        return False
    for child in tuple(layer.modules())[1:]:
        if child._forward_pre_hooks or child._forward_hooks:
            return False
    return True


class CudaGraphDecoderLayerBank:
    """Two graphs per query-one decoder layer around native attention.

    The first graph contains input RMSNorm plus QKV/norm/RoPE.  Native cache
    update, the dynamic attention mask and the installed attention function
    remain ordinary eager operations.  The second graph contains output
    projection, both residual additions, post-attention RMSNorm and MLP.
    """

    def __init__(
        self,
        rows: Iterable[tuple[str, nn.Module]],
        predicate: Predicate,
        *,
        signature_provider: Callable[[nn.Module], tuple] = _module_signature,
    ) -> None:
        materialized = list(rows)
        if not materialized:
            raise ValueError("decoder-layer graph bank requires at least one layer")
        self.names = tuple(name for name, _ in materialized)
        self.modules = tuple(module for _, module in materialized)
        if len(set(self.names)) != len(self.names) or len({id(m) for m in self.modules}) != len(self.modules):
            raise ValueError("decoder-layer names and modules must be unique")
        self.predicate = predicate
        self._signature = signature_provider
        self._owner_thread = threading.get_ident()
        self._patches = tuple(_PatchedMethod.capture(module) for module in self.modules)
        self._slots: dict[tuple, dict[str, Any]] = {}
        self._sealed = False
        self._closed = False
        self._busy = False
        self.stats = {
            "captures": 0,
            "replays": 0,
            "calls": 0,
            "exact_first_use": 0,
            "fallbacks": 0,
            "capture_seconds": 0.0,
            "cache_update_calls": 0,
            "cache_owner_checks": 0,
            "borrowed_outputs": 0,
            "cloned_outputs": 0,
        }
        try:
            for index, module in enumerate(self.modules):
                def forward(
                    _module,
                    hidden_states,
                    position_embeddings,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=None,
                    _index=index,
                    **kwargs,
                ):
                    return self._call(
                        _index,
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        use_cache,
                        cache_position,
                        kwargs,
                    )

                module.forward = MethodType(forward, module)
        except Exception:
            for patch in self._patches:
                patch.restore()
            raise

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def closed(self) -> bool:
        return self._closed

    @staticmethod
    def _slot_key(index: int, value: Tensor, cos: Tensor, sin: Tensor) -> tuple:
        return (
            index,
            tuple(value.shape),
            tuple(value.stride()),
            tuple(cos.shape),
            tuple(cos.stride()),
            tuple(sin.shape),
            tuple(sin.stride()),
        )

    def _capture_eligible(self, index: int, value: Tensor, cos: Tensor, sin: Tensor) -> bool:
        layer = self.modules[index]
        attention = layer.self_attn
        return (
            not layer.training
            and not torch.is_grad_enabled()
            and _decoder_child_hooks_clear(layer)
            and value.is_cuda
            and value.dtype == torch.bfloat16
            and value.is_contiguous()
            and tuple(value.shape) == (1, 1, 2048)
            and cos.is_cuda
            and sin.is_cuda
            and cos.dtype == value.dtype
            and sin.dtype == value.dtype
            and cos.is_contiguous()
            and sin.is_contiguous()
            and tuple(cos.shape) == (1, 1, attention.head_dim)
            and tuple(sin.shape) == tuple(cos.shape)
            and cos.device == value.device
            and sin.device == value.device
            and attention.q_proj.weight.device == value.device
        )

    def prewarm(self, samples: Mapping[str, tuple[Tensor, Tensor, Tensor]]) -> None:
        if self._closed or self._sealed:
            raise RuntimeError("decoder-layer bank cannot prewarm after seal/close")
        if threading.get_ident() != self._owner_thread or self._busy:
            raise RuntimeError("decoder-layer prewarm requires its owner thread")
        if set(samples) != set(self.names):
            missing = sorted(set(self.names) - set(samples))
            extra = sorted(set(samples) - set(self.names))
            raise ValueError(f"decoder-layer sample mismatch: missing={missing}, extra={extra}")
        for index, name in enumerate(self.names):
            value, cos, sin = samples[name]
            with torch.inference_mode():
                if not self._capture_eligible(index, value, cos, sin):
                    raise ValueError(f"decoder-layer:{name} is not a supported query-one sample")
                self._capture(index, value, cos, sin)

    def _capture(self, index: int, value: Tensor, cos: Tensor, sin: Tensor) -> None:
        key = self._slot_key(index, value, cos, sin)
        if key in self._slots:
            raise RuntimeError(f"decoder-layer:{self.names[index]} was already captured")
        layer = self.modules[index]
        signature = self._signature(layer)
        contract = _decoder_contract(layer)
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(value.device).clone()
        static_input = value.detach().clone()
        static_cos = cos.detach().clone()
        static_sin = sin.detach().clone()
        static_attention = value.detach().clone()
        stream = torch.cuda.Stream(device=value.device)
        stream.wait_stream(torch.cuda.current_stream(value.device))
        begin = time.perf_counter()
        with torch.cuda.stream(stream):
            for _ in range(3):
                _decoder_prelude(layer, static_input, static_cos, static_sin)
                _decoder_tail(layer, static_input, static_attention)
        torch.cuda.current_stream(value.device).wait_stream(stream)

        prelude_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(prelude_graph):
            prelude_outputs = _decoder_prelude(layer, static_input, static_cos, static_sin)
        tail_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(tail_graph):
            tail_output = _decoder_tail(layer, static_input, static_attention)

        if not torch.equal(cpu_rng, torch.get_rng_state()):
            raise RuntimeError("decoder-layer capture changed CPU RNG")
        if not torch.equal(cuda_rng, torch.cuda.get_rng_state(value.device)):
            raise RuntimeError("decoder-layer capture changed CUDA RNG")
        if signature != self._signature(layer) or contract != _decoder_contract(layer):
            raise RuntimeError("decoder-layer changed during capture")

        prelude_graph.replay()
        replayed_prelude = tuple(output.clone() for output in prelude_outputs)
        reference_prelude = _decoder_prelude(layer, value, cos, sin)
        if not all(torch.equal(left, right) for left, right in zip(replayed_prelude, reference_prelude)):
            raise RuntimeError("decoder-layer prelude prewarm exact-output gate failed")
        tail_graph.replay()
        replayed_tail = tail_output.clone()
        reference_tail = _decoder_tail(layer, value, static_attention)
        if not torch.equal(replayed_tail, reference_tail):
            raise RuntimeError("decoder-layer tail prewarm exact-output gate failed")

        self._slots[key] = {
            "input": static_input,
            "cos": static_cos,
            "sin": static_sin,
            "attention": static_attention,
            "prelude_outputs": prelude_outputs,
            "tail_output": tail_output,
            "prelude_graph": prelude_graph,
            "tail_graph": tail_graph,
            "signature": signature,
            "contract": contract,
            "guard": DecoderGuardPlan(layer, _decoder_contract, _decoder_child_hooks_clear),
            "verified": False,
        }
        self.stats["captures"] += 2
        self.stats["capture_seconds"] += time.perf_counter() - begin

    def seal(self) -> None:
        if self._closed:
            raise RuntimeError("decoder-layer graph bank is closed")
        covered = {key[0] for key in self._slots}
        if covered != set(range(len(self.modules))):
            raise RuntimeError(
                f"decoder-layer bank cannot seal with {len(covered)}/{len(self.modules)} layers"
            )
        self._sealed = True

    def _call(
        self,
        index: int,
        hidden_states: Tensor,
        position_embeddings,
        attention_mask,
        position_ids,
        past_key_values,
        use_cache,
        cache_position,
        kwargs: dict[str, Any],
    ):
        layer = self.modules[index]
        native = self._patches[index].bound_original
        if not (
            isinstance(position_embeddings, tuple)
            and len(position_embeddings) == 2
            and all(isinstance(value, Tensor) for value in position_embeddings)
        ):
            self.stats["fallbacks"] += 1
            return native(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
        cos, sin = position_embeddings
        slot = self._slots.get(self._slot_key(index, hidden_states, cos, sin))
        eligible = (
            not self._closed
            and self._sealed
            and self.predicate()
            and threading.get_ident() == self._owner_thread
            and not self._busy
            and not torch.is_grad_enabled()
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and slot is not None
            and "past_key_value" not in kwargs
            and slot["guard"]()
            and self._signature(layer) == slot["signature"]
        )
        if not eligible:
            self.stats["fallbacks"] += 1
            return native(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        self._busy = True
        try:
            slot["input"].copy_(hidden_states)
            slot["cos"].copy_(cos)
            slot["sin"].copy_(sin)
            slot["prelude_graph"].replay()
            query, key, value_states = slot["prelude_outputs"]
            borrowed_key_ptr = key.data_ptr()
            borrowed_value_ptr = value_states.data_ptr()
            self.stats["replays"] += 1

            if not slot["verified"]:
                reference = _decoder_prelude(layer, hidden_states, cos, sin)
                if not all(
                    torch.equal(left, right)
                    for left, right in zip((query, key, value_states), reference)
                ):
                    raise RuntimeError("decoder-layer first-use prelude exact-output gate failed")

            attention = layer.self_attn
            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key, value_states = past_key_values.update(
                    key, value_states, attention.layer_idx, cache_kwargs
                )
                self.stats["cache_update_calls"] += 1
                if key.data_ptr() == borrowed_key_ptr or value_states.data_ptr() == borrowed_value_ptr:
                    raise RuntimeError("decoder cache retained a borrowed graph output")
                self.stats["cache_owner_checks"] += 2

            from qwen_asr.core.transformers_backend import modeling_qwen3_asr as qwen

            attention_function = qwen.eager_attention_forward
            if attention.config._attn_implementation != "eager":
                attention_function = qwen.ALL_ATTENTION_FUNCTIONS[
                    attention.config._attn_implementation
                ]
            attention_kwargs = dict(kwargs)
            attention_kwargs["position_ids"] = position_ids
            attention_kwargs["use_cache"] = use_cache
            attention_output, _ = attention_function(
                attention,
                query,
                key,
                value_states,
                attention_mask,
                dropout=0.0 if not attention.training else attention.attention_dropout,
                scaling=attention.scaling,
                **attention_kwargs,
            )
            attention_output = attention_output.reshape(*hidden_states.shape[:-1], -1).contiguous()

            slot["attention"].copy_(attention_output)
            slot["tail_graph"].replay()
            output = slot["tail_output"].clone()
            self.stats["replays"] += 1
            self.stats["calls"] += 1
            self.stats["cloned_outputs"] += 1
            if not slot["verified"]:
                reference = _decoder_tail(layer, hidden_states, attention_output)
                if not torch.equal(output, reference):
                    raise RuntimeError("decoder-layer first-use tail exact-output gate failed")
                slot["verified"] = True
                self.stats["exact_first_use"] += 1
            return output
        finally:
            self._busy = False

    def retention(self) -> dict[str, int | bool]:
        explicit = 0
        for slot in self._slots.values():
            tensors = (
                slot["input"],
                slot["cos"],
                slot["sin"],
                slot["attention"],
                *slot["prelude_outputs"],
                slot["tail_output"],
            )
            explicit += sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        return {
            "slots": len(self._slots),
            "graphs": 2 * len(self._slots),
            "guard_plans": len(self._slots),
            "guard_calls": sum(slot["guard"].calls for slot in self._slots.values()),
            "guard_fallbacks": sum(slot["guard"].fallbacks for slot in self._slots.values()),
            "guard_source_bytes": sum(len(slot["guard"].generated_source.encode("utf8")) for slot in self._slots.values()),
            "explicit_tensor_bytes": explicit,
            "graph_allocator_workspace_included": False,
            "native_cache_mask_and_attention": True,
            "borrowed_prelude_outputs": True,
            "tail_output_contract": "owned clone on every decoder-layer replay",
        }

    def close(self) -> None:
        if self._busy:
            raise RuntimeError("decoder-layer graph bank cannot close during replay")
        if self._closed:
            return
        for patch in self._patches:
            patch.restore()
        for slot in self._slots.values():
            slot["guard"].close()
        self._slots.clear()
        self._closed = True


def select_decoder_layer_resource(
    llm: nn.Module,
    predicate: Predicate,
    *,
    signature_provider: Callable[[nn.Module], tuple] = _module_signature,
) -> CudaGraphDecoderLayerBank:
    rows = [
        (name, module)
        for name, module in llm.named_modules()
        if type(module).__name__ == "Qwen3ASRThinkerTextDecoderLayer"
    ]
    if len(rows) != 28:
        raise ValueError(f"unsupported Qwen decoder-layer inventory: {len(rows)}")
    return CudaGraphDecoderLayerBank(rows, predicate, signature_provider=signature_provider)


def select_qkv_resource(llm: nn.Module, predicate: Predicate, *, signature_provider: Callable[[nn.Module], tuple] = _module_signature) -> CudaGraphQKVBank:
    rows = [
        (name, module)
        for name, module in llm.named_modules()
        if type(module).__name__ == "Qwen3ASRTextAttention"
    ]
    if len(rows) != 28:
        raise ValueError(f"unsupported Qwen attention inventory: {len(rows)}")
    return CudaGraphQKVBank(rows, predicate, borrow_outputs=True, signature_provider=signature_provider)


def select_query_one_resources(
    llm: nn.Module,
    private_route_active: Predicate,
    *,
    signature_provider: Callable[[nn.Module], tuple] = _module_signature,
) -> dict[str, CudaGraphModuleBank]:
    """Attach the simple graph banks used by the inference service.

    The returned final-norm bank is unprewarmed and therefore eager-only.
    Decoder-layer fixed regions are owned separately so cache update, the
    dynamic attention mask and the installed attention implementation remain
    native operations between the two graphs.
    """

    modules = list(llm.named_modules())
    # Decoder-layer input/post norms, MLP and attention output projection are
    # owned by the two-graph decoder resource.  Only the model's final norm is
    # independent of that layer boundary.
    norms = [
        (name, module)
        for name, module in modules
        if type(module).__name__ == "Qwen3ASRTextRMSNorm"
        and name.rsplit(".", 1)[-1] == "norm"
    ]
    if len(norms) != 1:
        raise ValueError(
            f"unsupported Qwen final-norm inventory: norm={len(norms)}"
        )
    created: dict[str, CudaGraphModuleBank] = {}
    try:
        created["final_norm"] = CudaGraphModuleBank(
            norms,
            private_route_active,
            label="private final RMSNorm",
            signature_provider=signature_provider,
        )
        return created
    except Exception:
        for bank in reversed(tuple(created.values())):
            bank.close()
        raise


def close_query_one_resources(resources: Mapping[str, CudaGraphModuleBank]) -> None:
    """Restore patched methods in reverse installation order."""

    for bank in reversed(tuple(resources.values())):
        bank.close()


class QueryOneExecutionGraphs:
    """Own query-one graph resources for one loaded Qwen model."""

    def __init__(self, llm: nn.Module, predicate: Predicate) -> None:
        self.simple: dict[str, CudaGraphModuleBank] = {}
        self.decoder: CudaGraphDecoderLayerBank | None = None
        self.closed = False
        self._signatures = SignatureRegistry(_module_signature)
        try:
            self.simple = select_query_one_resources(llm, predicate, signature_provider=self._signatures)
            self.decoder = select_decoder_layer_resource(
                llm, predicate, signature_provider=self._signatures
            )
            device = next(llm.parameters()).device
            base = torch.linspace(
                -1.0,
                1.0,
                2048,
                device=device,
                dtype=torch.float32,
            ).to(torch.bfloat16).view(1, 1, 2048)
            with torch.inference_mode():
                for bank in self.simple.values():
                    bank.prewarm({name: base for name in bank.names})
                    bank.seal()
                decoder_samples = {}
                for name, module in zip(self.decoder.names, self.decoder.modules):
                    cos = torch.ones(
                        (1, 1, module.self_attn.head_dim),
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    sin = torch.zeros_like(cos)
                    decoder_samples[name] = (base, cos, sin)
                self.decoder.prewarm(decoder_samples)
                self.decoder.seal()
        except Exception:
            self.close()
            raise

    def stats(self) -> dict[str, dict[str, Any]]:
        if self.decoder is None:
            return {}
        resources: dict[str, Any] = dict(self.simple)
        resources["decoder"] = self.decoder
        return {
            name: {
                "stats": dict(bank.stats),
                "retention": bank.retention(),
                "sealed": bool(bank.sealed),
            }
            for name, bank in resources.items()
        }

    def signature_stats(self) -> dict:
        return self._signatures.stats()

    def close(self) -> None:
        if self.closed:
            return
        if self.decoder is not None:
            self.decoder.close()
        close_query_one_resources(self.simple)
        self._signatures.close()
        self.closed = True
