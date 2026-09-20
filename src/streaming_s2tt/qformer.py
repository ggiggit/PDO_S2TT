"""Chunk-local Q-Former adapters for streaming speech features."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _QFormerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        encoder_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=encoder_dim,
            vdim=encoder_dim,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, queries: Tensor, memory: Tensor, memory_padding_mask: Optional[Tensor]
    ) -> Tensor:
        x = self.norm1(queries)
        x = queries + self.dropout(self.self_attn(x, x, x, need_weights=False)[0])
        q = self.norm2(x)
        x = x + self.dropout(
            self.cross_attn(
                q,
                memory,
                memory,
                key_padding_mask=memory_padding_mask,
                need_weights=False,
            )[0]
        )
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class ChunkQFormer(nn.Module):
    """Compress one Zipformer chunk into a fixed number of LLM soft tokens.

    Unlike MCAT's utterance-level Q-Former, this module is called once per
    streaming chunk. No future chunk is visible to the learned queries.
    """

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        num_query_tokens: int = 4,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
        implementation: str = "custom",
        output_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if not 1 <= num_query_tokens <= 150:
            raise ValueError("num_query_tokens must be in [1, 150]")
        if implementation not in {"custom", "blip2"}:
            raise ValueError("implementation must be custom or blip2")
        self.num_query_tokens = num_query_tokens
        self.implementation = implementation
        self.log_output_scale = nn.Parameter(
            torch.tensor(float(output_scale)).log()
        )

        if implementation == "blip2":
            from transformers import Blip2QFormerConfig, Blip2QFormerModel

            config = Blip2QFormerConfig(
                encoder_hidden_size=encoder_dim,
                hidden_size=hidden_dim,
                intermediate_size=ffn_dim,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                cross_attention_frequency=2,
            )
            self.query_tokens = nn.Parameter(
                torch.empty(1, num_query_tokens, hidden_dim)
            )
            # MCAT initializes the learned queries at unit scale.
            nn.init.normal_(self.query_tokens, mean=0.0, std=1.0)
            self.qformer = Blip2QFormerModel(config)
            if llm_dim <= 1536:
                self.output = nn.Sequential(
                    nn.Linear(hidden_dim, llm_dim),
                    nn.LayerNorm(llm_dim),
                )
            else:
                self.output = nn.Sequential(
                    nn.Linear(hidden_dim, 1536),
                    nn.ReLU(),
                    nn.Linear(1536, llm_dim),
                    nn.LayerNorm(llm_dim),
                )
            self.input_summary = nn.Sequential(
                nn.LayerNorm(encoder_dim),
                nn.Linear(encoder_dim, llm_dim),
            )
            nn.init.zeros_(self.input_summary[1].weight)
            nn.init.zeros_(self.input_summary[1].bias)
            return

        self.query_tokens = nn.Parameter(
            torch.empty(1, num_query_tokens, hidden_dim)
        )
        nn.init.normal_(self.query_tokens, std=0.02)
        self.layers = nn.ModuleList(
            _QFormerLayer(
                hidden_dim, encoder_dim, num_heads, ffn_dim, dropout
            )
            for _ in range(num_layers)
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, llm_dim), nn.LayerNorm(llm_dim)
        )
        self.input_summary = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, llm_dim),
        )
        nn.init.zeros_(self.input_summary[1].weight)
        nn.init.zeros_(self.input_summary[1].bias)

    def _summary_queries(
        self, encoder_out: Tensor, encoder_mask: Optional[Tensor]
    ) -> Tensor:
        if encoder_mask is None:
            encoder_mask = torch.ones(
                encoder_out.shape[:2], dtype=torch.bool, device=encoder_out.device
            )
        pooled = []
        for batch_index in range(encoder_out.size(0)):
            valid = encoder_out[batch_index][encoder_mask[batch_index].bool()]
            if valid.numel() == 0:
                valid = encoder_out[batch_index, :1]
            pieces = []
            for query_index in range(self.num_query_tokens):
                start = (query_index * valid.size(0)) // self.num_query_tokens
                end = ((query_index + 1) * valid.size(0)) // self.num_query_tokens
                end = max(start + 1, end)
                end = min(end, valid.size(0))
                pieces.append(valid[start:end].mean(dim=0))
            pooled.append(torch.stack(pieces))
        return torch.stack(pooled)

    def forward(
        self,
        encoder_out: Tensor,
        encoder_mask: Optional[Tensor] = None,
        frame_token_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            encoder_out: ``(batch, chunk_frames, encoder_dim)``.
            encoder_mask: ``(batch, chunk_frames)`` with True for valid frames.
        """
        if encoder_out.ndim != 3:
            raise ValueError("encoder_out must have shape (B, T, D)")
        encoder_out = encoder_out.to(dtype=self.query_tokens.dtype)
        padding_mask = None if encoder_mask is None else ~encoder_mask.bool()
        queries = self.query_tokens.expand(encoder_out.size(0), -1, -1)
        if self.implementation == "blip2":
            output = self.qformer(
                query_embeds=queries,
                encoder_hidden_states=encoder_out,
                encoder_attention_mask=encoder_mask.bool()
                if encoder_mask is not None
                else None,
                return_dict=True,
            ).last_hidden_state
            projected = self.output(output)
            pooled = self._summary_queries(encoder_out, encoder_mask)
            projected = projected + self.input_summary(pooled)
            return projected * self.log_output_scale.exp().to(projected.dtype)

        for layer in self.layers:
            queries = layer(queries, encoder_out, padding_mask)
        projected = self.output(queries)
        pooled = self._summary_queries(encoder_out, encoder_mask)
        projected = projected + self.input_summary(pooled)
        return projected * self.log_output_scale.exp().to(projected.dtype)


class DirectChunkAdapter(nn.Module):
    """A small ordered latent-to-LLM adapter for diagnosing the Q-Former path."""

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        num_query_tokens: int = 8,
        output_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.projection = nn.Linear(encoder_dim, llm_dim)
        self.query_bias = nn.Parameter(torch.zeros(1, num_query_tokens, llm_dim))
        self.output_norm = nn.LayerNorm(llm_dim)
        self.log_output_scale = nn.Parameter(torch.tensor(float(output_scale)).log())
        nn.init.normal_(self.query_bias, std=0.02)

    def forward(
        self, encoder_out: Tensor, encoder_mask: Optional[Tensor] = None
    ) -> Tensor:
        if encoder_out.ndim != 3:
            raise ValueError("encoder_out must have shape (B, T, D)")
        if encoder_mask is None:
            encoder_mask = torch.ones(
                encoder_out.shape[:2], dtype=torch.bool, device=encoder_out.device
            )
        pooled = []
        for batch_index in range(encoder_out.size(0)):
            valid = encoder_out[batch_index][encoder_mask[batch_index].bool()]
            if valid.numel() == 0:
                valid = encoder_out[batch_index, :1]
            pieces = []
            for query_index in range(self.num_query_tokens):
                start = (query_index * valid.size(0)) // self.num_query_tokens
                end = ((query_index + 1) * valid.size(0)) // self.num_query_tokens
                end = max(start + 1, end)
                end = min(end, valid.size(0))
                pieces.append(valid[start:end].mean(dim=0))
            pooled.append(torch.stack(pieces))
        pooled_tensor = torch.stack(pooled).to(self.projection.weight.dtype)
        output = self.projection(pooled_tensor) + self.query_bias
        projected = self.output_norm(output)
        return projected * self.log_output_scale.exp().to(projected.dtype)


class ResidualMeanChunkAdapter(nn.Module):
    """Compress aligned Qwen audio embeddings without destroying their space.

    Qwen3-ASR Audio Tower outputs already live in the thinker embedding space.
    Ordered segment means provide the short cache representation, while a
    zero-initialized residual projection lets S2TT training adapt that evidence
    without replacing the pretrained acoustic geometry on the first step.
    """

    uses_discrete_bridge = False

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        num_query_tokens: int = 4,
        output_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if encoder_dim != llm_dim:
            raise ValueError(
                "ResidualMeanChunkAdapter requires encoder_dim == llm_dim"
            )
        if num_query_tokens < 1:
            raise ValueError("num_query_tokens must be positive")
        if output_scale <= 0:
            raise ValueError("output_scale must be positive")
        self.num_query_tokens = int(num_query_tokens)
        self.input_norm = nn.LayerNorm(encoder_dim)
        self.residual = nn.Linear(encoder_dim, llm_dim)
        self.log_output_scale = nn.Parameter(torch.tensor(float(output_scale)).log())
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(
        self, encoder_out: Tensor, encoder_mask: Optional[Tensor] = None
    ) -> Tensor:
        if encoder_out.ndim != 3:
            raise ValueError("encoder_out must have shape (B, T, D)")
        if encoder_mask is None:
            encoder_mask = torch.ones(
                encoder_out.shape[:2], dtype=torch.bool, device=encoder_out.device
            )
        pooled = []
        for batch_index in range(encoder_out.size(0)):
            valid = encoder_out[batch_index][encoder_mask[batch_index].bool()]
            if valid.numel() == 0:
                valid = encoder_out[batch_index, :1]
            pieces = []
            for query_index in range(self.num_query_tokens):
                start = (query_index * valid.size(0)) // self.num_query_tokens
                end = ((query_index + 1) * valid.size(0)) // self.num_query_tokens
                end = min(valid.size(0), max(start + 1, end))
                pieces.append(valid[start:end].mean(dim=0))
            pooled.append(torch.stack(pieces))
        means = torch.stack(pooled).to(self.residual.weight.dtype)
        output = means + self.residual(self.input_norm(means))
        return output * self.log_output_scale.exp().to(output.dtype)


class PredictiveHiddenChunkAdapter(nn.Module):
    """Compress causal source-decoder states into target-side soft tokens.

    The source states are final-layer predictive states and therefore have a
    much larger norm than input token embeddings. Ordered pooling preserves
    their within-chunk progression; normalization and a small output scale put
    them back in the target LLM's input range. The zero-initialized residual
    starts as a direction-preserving mapping instead of a random projection.
    """

    uses_discrete_bridge = False

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        num_query_tokens: int = 4,
        output_scale: float = 0.03,
    ) -> None:
        super().__init__()
        if encoder_dim != llm_dim:
            raise ValueError(
                "PredictiveHiddenChunkAdapter requires encoder_dim == llm_dim"
            )
        if num_query_tokens < 1:
            raise ValueError("num_query_tokens must be positive")
        if output_scale <= 0:
            raise ValueError("output_scale must be positive")
        self.num_query_tokens = int(num_query_tokens)
        self.input_norm = nn.LayerNorm(encoder_dim)
        self.residual = nn.Linear(encoder_dim, llm_dim)
        self.log_output_scale = nn.Parameter(torch.tensor(float(output_scale)).log())
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(
        self, encoder_out: Tensor, encoder_mask: Optional[Tensor] = None
    ) -> Tensor:
        if encoder_out.ndim != 3:
            raise ValueError("encoder_out must have shape (B, T, D)")
        if encoder_mask is None:
            encoder_mask = torch.ones(
                encoder_out.shape[:2], dtype=torch.bool, device=encoder_out.device
            )
        pooled = []
        for batch_index in range(encoder_out.size(0)):
            valid = encoder_out[batch_index][encoder_mask[batch_index].bool()]
            if valid.numel() == 0:
                valid = encoder_out[batch_index, :1]
            pieces = []
            for query_index in range(self.num_query_tokens):
                start = (query_index * valid.size(0)) // self.num_query_tokens
                end = ((query_index + 1) * valid.size(0)) // self.num_query_tokens
                end = min(valid.size(0), max(start + 1, end))
                pieces.append(valid[start:end].mean(dim=0))
            pooled.append(torch.stack(pieces))
        means = torch.stack(pooled).to(self.residual.weight.dtype)
        normalized = self.input_norm(means)
        output = normalized + self.residual(normalized)
        return output * self.log_output_scale.exp().to(output.dtype)


class PredictiveResidualSourceAdapter(nn.Module):
    """Map ASR predictive states to a zero-initialized source residual.

    The target keeps each ordinary source token embedding and adds this
    continuous residual at the same position. Initialization is exactly zero,
    so no cache position or parent-model behavior changes before training.
    """

    uses_discrete_bridge = False

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, predictive_hidden: Tensor) -> Tensor:
        if predictive_hidden.ndim != 3:
            raise ValueError("predictive_hidden must have shape (B, T, D)")
        values = predictive_hidden.to(self.projection.weight.dtype)
        return self.projection(self.input_norm(values))


class SoftPosteriorTokenAdapter(nn.Module):
    """Adapt posterior-weighted token embeddings without adding cache tokens."""

    uses_discrete_bridge = False

    def __init__(self, hidden_dim: int, output_scale: float = 1.0) -> None:
        super().__init__()
        if output_scale <= 0:
            raise ValueError("output_scale must be positive")
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.residual = nn.Linear(hidden_dim, hidden_dim)
        self.log_output_scale = nn.Parameter(torch.tensor(float(output_scale)).log())
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(
        self,
        posterior_embeddings: Tensor,
        encoder_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if posterior_embeddings.ndim != 3:
            raise ValueError("posterior_embeddings must have shape (B, T, D)")
        if encoder_mask is not None and encoder_mask.shape[:2] != posterior_embeddings.shape[:2]:
            raise ValueError("encoder_mask must match posterior embedding dimensions")
        values = posterior_embeddings.to(self.residual.weight.dtype)
        output = values + self.residual(self.input_norm(values))
        return output * self.log_output_scale.exp().to(output.dtype)


class PureLatentAdapter(nn.Module):
    """Map continuous streaming speech states directly to Qwen soft tokens.

    This adapter deliberately has no RNNT vocabulary, token lookup table, or
    discrete speech hint. Each valid acoustic frame is projected independently
    so the output at time ``t`` cannot depend on a future frame.
    """

    uses_discrete_bridge = False

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        hidden_dim: int = 1024,
        output_scale: float = 0.03,
    ) -> None:
        super().__init__()
        if output_scale <= 0:
            raise ValueError("output_scale must be positive")
        self.input_norm = nn.LayerNorm(encoder_dim)
        self.temporal_projection = nn.Linear(encoder_dim, encoder_dim)
        self.temporal_conv = nn.Conv1d(
            encoder_dim, encoder_dim, kernel_size=5, padding=0
        )
        # Start as an identity path so an existing frame adapter can be
        # continued with causal context without changing its initial scale.
        nn.init.eye_(self.temporal_projection.weight)
        nn.init.zeros_(self.temporal_projection.bias)
        nn.init.zeros_(self.temporal_conv.weight)
        nn.init.zeros_(self.temporal_conv.bias)
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
        )
        self.output_norm = nn.LayerNorm(llm_dim)
        self.log_output_scale = nn.Parameter(torch.tensor(float(output_scale)).log())

    def forward(
        self,
        encoder_out: Tensor,
        encoder_mask: Optional[Tensor] = None,
        frame_token_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if encoder_out.ndim != 3:
            raise ValueError("encoder_out must have shape (B, T, D)")
        if frame_token_ids is not None:
            raise ValueError("PureLatentAdapter does not accept discrete token hints")
        if encoder_mask is None:
            encoder_mask = torch.ones(
                encoder_out.shape[:2], dtype=torch.bool, device=encoder_out.device
            )
        if encoder_mask.shape[:2] != encoder_out.shape[:2]:
            raise ValueError("encoder_mask must match encoder_out batch and frame dimensions")
        x = encoder_out.to(dtype=self.input_norm.weight.dtype)
        x = self.input_norm(x)
        temporal = self.temporal_projection(x)
        temporal = temporal.transpose(1, 2)
        temporal = F.pad(temporal, (self.temporal_conv.kernel_size[0] - 1, 0))
        temporal = self.temporal_conv(temporal).transpose(1, 2)
        x = x + torch.tanh(temporal)
        soft = self.output_norm(self.projection(x))
        soft = soft * self.log_output_scale.exp().to(soft.dtype)
        return soft


class IdentityAdapter(nn.Module):
    """Pass precomputed LLM embeddings through without an acoustic projector."""

    uses_discrete_bridge = False

    def __init__(self) -> None:
        super().__init__()
        self.log_output_scale = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(
        self,
        encoder_out: Tensor,
        encoder_mask: Optional[Tensor] = None,
        frame_token_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if frame_token_ids is not None:
            raise ValueError("IdentityAdapter does not accept discrete token hints")
        if encoder_out.ndim != 3:
            raise ValueError("encoder_out must have shape (B, T, D)")
        if encoder_mask is None:
            return encoder_out * self.log_output_scale.exp().to(encoder_out.dtype)
        outputs = []
        max_length = 0
        for batch_index in range(encoder_out.size(0)):
            valid = encoder_out[batch_index][encoder_mask[batch_index].bool()]
            if valid.numel() == 0:
                valid = encoder_out[batch_index, :1]
            outputs.append(valid)
            max_length = max(max_length, valid.size(0))
        padded = encoder_out.new_zeros(
            encoder_out.size(0), max_length, encoder_out.size(-1)
        )
        for index, value in enumerate(outputs):
            padded[index, : value.size(0)] = value
        return padded * self.log_output_scale.exp().to(padded.dtype)


class LinearTokenBridgeAdapter(nn.Module):
    """Learn a continuous per-token bridge from decoder hidden states to Qwen."""

    uses_discrete_bridge = False

    def __init__(self, encoder_dim: int, llm_dim: int, output_scale: float = 1.0):
        super().__init__()
        if output_scale <= 0:
            raise ValueError("output_scale must be positive")
        self.projection = nn.Linear(encoder_dim, llm_dim)
        self.log_output_scale = nn.Parameter(torch.tensor(float(output_scale)).log())

    def forward(
        self,
        encoder_out: Tensor,
        encoder_mask: Optional[Tensor] = None,
        frame_token_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if frame_token_ids is not None:
            raise ValueError("LinearTokenBridgeAdapter does not accept token IDs")
        if encoder_out.ndim != 3:
            raise ValueError("encoder_out must have shape (B, T, D)")
        if encoder_mask is None:
            encoder_mask = torch.ones(
                encoder_out.shape[:2], dtype=torch.bool, device=encoder_out.device
            )
        outputs = []
        max_length = 0
        for batch_index in range(encoder_out.size(0)):
            valid = encoder_out[batch_index][encoder_mask[batch_index].bool()]
            if valid.numel() == 0:
                valid = encoder_out[batch_index, :1]
            projected = self.projection(valid.to(self.projection.weight.dtype))
            outputs.append(projected)
            max_length = max(max_length, projected.size(0))
        padded = encoder_out.new_zeros(
            encoder_out.size(0), max_length, self.projection.out_features,
            dtype=self.projection.weight.dtype,
        )
        for index, value in enumerate(outputs):
            padded[index, : value.size(0)] = value
        return padded * self.log_output_scale.exp().to(padded.dtype)


class TokenBridgeAdapter(nn.Module):
    """Predict soft Qwen source tokens directly from RNNT decoder states."""

    uses_discrete_bridge = True

    def __init__(
        self,
        encoder_dim: int,
        llm_dim: int,
        rnnt_to_llm_table: Tensor,
        output_scale: float = 0.01,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.num_query_tokens = 1
        self.temperature = float(temperature)
        self.classifier = nn.Linear(encoder_dim, rnnt_to_llm_table.size(0))
        self.register_buffer("rnnt_to_llm_table", rnnt_to_llm_table)
        self.output_norm = nn.LayerNorm(llm_dim)
        self.acoustic_norm = nn.LayerNorm(encoder_dim)
        self.acoustic_projection = nn.Linear(encoder_dim, llm_dim)
        # Start from the proven token bridge and let training add acoustic detail.
        nn.init.zeros_(self.acoustic_projection.weight)
        nn.init.zeros_(self.acoustic_projection.bias)
        self.log_output_scale = nn.Parameter(torch.tensor(float(output_scale)).log())

    def forward(
        self,
        encoder_out: Tensor,
        encoder_mask: Optional[Tensor] = None,
        frame_token_ids: Optional[Tensor] = None,
    ) -> Tensor:
        if encoder_out.ndim != 3:
            raise ValueError("encoder_out must have shape (B, T, D)")
        if encoder_mask is None:
            encoder_mask = torch.ones(
                encoder_out.shape[:2], dtype=torch.bool, device=encoder_out.device
            )
        if frame_token_ids is not None:
            if frame_token_ids.ndim == 1:
                frame_token_ids = frame_token_ids.unsqueeze(0)
            if frame_token_ids.shape[:2] != encoder_out.shape[:2]:
                raise ValueError(
                    "frame_token_ids must match encoder_out batch and frame dimensions"
                )
            frame_token_ids = frame_token_ids.to(
                device=encoder_out.device, dtype=torch.long
            )
        outputs = []
        logits_for_loss = []
        max_length = 0
        for batch_index in range(encoder_out.size(0)):
            valid = encoder_out[batch_index][encoder_mask[batch_index].bool()]
            if valid.numel() == 0:
                valid = encoder_out[batch_index, :1]
            logits = self.classifier(valid.to(self.classifier.weight.dtype))
            probabilities = torch.softmax(logits.float() / self.temperature, dim=-1).to(
                self.rnnt_to_llm_table.dtype
            )
            soft = probabilities @ self.rnnt_to_llm_table
            acoustic = self.acoustic_projection(
                self.acoustic_norm(valid.to(self.acoustic_norm.weight.dtype))
            ).to(soft.dtype)
            soft = self.output_norm(soft + acoustic)
            if frame_token_ids is not None:
                valid_token_ids = frame_token_ids[batch_index][
                    encoder_mask[batch_index].bool()
                ]
                hint = self.rnnt_to_llm_table[valid_token_ids]
                soft = self.output_norm(soft + hint.to(soft.dtype))
            soft = soft * self.log_output_scale.exp().to(soft.dtype)
            outputs.append(soft)
            logits_for_loss.append(logits)
            max_length = max(max_length, soft.size(0))
        padded = encoder_out.new_zeros(
            encoder_out.size(0), max_length, self.rnnt_to_llm_table.size(1),
            dtype=self.rnnt_to_llm_table.dtype,
        )
        for index, soft in enumerate(outputs):
            padded[index, : soft.size(0)] = soft
        self.last_token_logits_flat = torch.cat(logits_for_loss, dim=0)
        return padded
