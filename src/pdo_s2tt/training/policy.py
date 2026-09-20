"""Sampling and likelihood utilities for the PDO G=4 behavior policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MethodType

import torch
import torch.nn.functional as F

from pdo_s2tt.history import Memory, visible_ids


@dataclass
class MaskedMemory(Memory):
    key_mask: torch.Tensor | None = None


def _masked_forward(block, hidden: torch.Tensor, memory: MaskedMemory):
    values, mask = memory.embeddings, memory.key_mask
    if mask is None or mask.dtype != torch.bool or mask.shape != values.shape[:2]:
        raise ValueError("history requires an explicit right-padding mask")
    if values.shape[1] == 0 or not bool(mask.any()):
        return hidden
    clean = torch.where(mask[..., None], values.float(), torch.zeros_like(values, dtype=torch.float32))
    split = lambda value: value.reshape(
        value.shape[0], value.shape[1], block.heads, -1,
    ).transpose(1, 2)
    positioned = clean + block.position[: values.shape[1]].float()
    normalized = F.layer_norm(positioned, (block.hidden_size,))
    key = split(F.linear(normalized, block.key.weight.float()))
    value = split(F.linear(normalized, block.value.weight.float()))
    query = split(F.linear(
        F.layer_norm(hidden.float(), (block.hidden_size,)), block.query.weight.float(),
    ))
    scores = query @ key.transpose(-1, -2) / math.sqrt(block.inner_size // block.heads)
    active_history = mask.any(-1)
    scores = scores.masked_fill(~mask[:, None, None, :], -torch.inf)
    scores = torch.where(active_history[:, None, None, None], scores, torch.zeros_like(scores))
    attention = torch.softmax(scores, -1)
    attended = (attention @ value).transpose(1, 2).reshape(
        *hidden.shape[:2], block.inner_size,
    )
    residual = F.linear(attended, block.output.weight.float()) * torch.tanh(block.gate.float())
    active = active_history[:, None].expand(hidden.shape[:2])
    if memory.query_mask is not None:
        active = active & memory.query_mask
    result = (hidden.float() + residual).to(hidden.dtype)
    return torch.where(active[..., None], result, hidden)


def install_masked_history(adapter):
    """Extend one history adapter with batched, padded history masks."""
    originals = [(block, block.forward) for block in adapter.blocks]
    for block, original in originals:
        def forward(current, hidden, memory, original=original):
            if isinstance(memory, MaskedMemory):
                return _masked_forward(current, hidden, memory)
            return original(hidden, memory)
        block.forward = MethodType(forward, block)
    adapter._pdo_masked_history = True

    def remove():
        for block, original in originals:
            block.forward = original
        adapter._pdo_masked_history = False
    return remove


def policy_from_decoder(decoder) -> dict:
    if decoder.source_decode or decoder.target_latent_mode != "full" or decoder.target_source_mode != "none":
        raise ValueError("PDO training requires the target-only full-latent route")
    return {
        "eos_id": int(decoder.eos_id),
        "temperature": float(decoder.temperature),
        "max_tokens": int(decoder.max_target_tokens),
        "wait_ids": sorted(map(int, decoder.target_wait_ids)),
        "forbidden_ids": sorted(map(int, decoder.target_forbidden_ids)),
        "leading_forbidden_ids": sorted(map(int, decoder.target_leading_forbidden_ids)),
        "no_repeat_ngram_size": int(getattr(decoder, "target_no_repeat_ngram_size", 0)),
        "force_token_if_empty": bool(getattr(decoder, "target_force_token_if_empty", False)),
    }


def _mask_logits(logits: torch.Tensor, generated: list[int], policy: dict) -> torch.Tensor:
    current = logits.clone()
    current[:, policy["wait_ids"]] = -torch.inf
    current[:, policy["forbidden_ids"]] = -torch.inf
    current[:, policy["leading_forbidden_ids"]] = -torch.inf
    if not generated and policy["force_token_if_empty"]:
        current[:, policy["eos_id"]] = -torch.inf
    width = policy["no_repeat_ngram_size"]
    if width > 1 and len(generated) >= width - 1:
        prefix = generated[-(width - 1):]
        blocked = {
            generated[index + width - 1]
            for index in range(len(generated) - width + 1)
            if generated[index:index + width - 1] == prefix
        }
        if blocked:
            current[:, sorted(blocked)] = -torch.inf
    return current


class ConditionedModel:
    def __init__(self, llm, adapter, histories: list[list[int]], private_start: int):
        self.llm, self.adapter, self.private_start = llm, adapter, private_start
        device = llm.get_input_embeddings().weight.device
        lengths = [len(history) for history in histories]
        maximum = max(lengths, default=0)
        ids = torch.zeros(len(histories), maximum, dtype=torch.long, device=device)
        for lane, history in enumerate(histories):
            if history:
                ids[lane, :len(history)] = torch.tensor(history, device=device)
        self.embeddings = llm.get_input_embeddings()(ids).detach()
        self.key_mask = (
            torch.arange(maximum, device=device)[None, :]
            < torch.tensor(lengths, device=device)[:, None]
        )

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def __call__(self, **kwargs):
        positions = kwargs["position_ids"][0]
        memory = MaskedMemory(
            self.embeddings,
            query_mask=positions >= self.private_start,
            key_mask=self.key_mask,
        )
        with self.adapter.condition(memory):
            return self.llm(**kwargs)


@torch.no_grad()
def sample_group(
    llm, adapter, tokenizer, prefix: torch.Tensor, policy: dict,
    generators: list[torch.Generator], histories: list[list[int]], private_start: int,
) -> list[dict]:
    """Draw four complete visible drafts from one actual acoustic prefix."""
    if len(generators) != 4 or len(histories) != 4:
        raise ValueError("PDO requires four persistent RNG lanes")
    clean = [visible_ids(tokenizer, history, adapter.blocks[0].max_tokens) for history in histories]
    model = ConditionedModel(llm, adapter, clean, private_start)
    width, device, prefix_length = 4, prefix.device, prefix.shape[1]
    generated = [[] for _ in range(width)]
    traces = [[] for _ in range(width)]
    base_masks = [None] * width
    closed = [False] * width
    inputs = prefix.expand(width, -1, -1).contiguous()
    cache = None
    last = [policy["eos_id"]] * width
    for step in range(policy["max_tokens"]):
        start = 0 if step == 0 else prefix_length + step - 1
        length = inputs.shape[1]
        positions = torch.arange(start, start + length, device=device).view(1, 1, length).expand(3, width, length)
        state = model(
            inputs_embeds=inputs,
            attention_mask=torch.ones(width, start + length, device=device, dtype=torch.long),
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = state.past_key_values
        for lane in range(width):
            if closed[lane]:
                continue
            logits = _mask_logits(state.logits[lane:lane + 1, -1].float(), generated[lane], policy)
            scaled = logits[0] / policy["temperature"]
            blocked = ~torch.isfinite(scaled)
            probabilities = torch.softmax(scaled, 0)
            token = int(torch.multinomial(probabilities, 1, generator=generators[lane]).item())
            if base_masks[lane] is None:
                base_masks[lane] = blocked.clone()
            base = base_masks[lane]
            traces[lane].append({
                "token_id": token,
                "old_logp": float(torch.log(probabilities[token])),
                "additional_blocked_ids": (blocked & ~base).nonzero().flatten().cpu().tolist(),
                "restored_ids": (base & ~blocked).nonzero().flatten().cpu().tolist(),
            })
            last[lane] = token
            if token == policy["eos_id"]:
                closed[lane] = True
            else:
                generated[lane].append(token)
        if all(closed):
            break
        inputs = llm.get_input_embeddings()(
            torch.tensor(last, device=device, dtype=torch.long)[:, None]
        )
    result = []
    for lane in range(width):
        result.append({
            "prefix_token_ids": generated[lane],
            "sampled_eot": closed[lane],
            "eot_id": policy["eos_id"],
            "trace": {
                "temperature": policy["temperature"],
                "vocabulary_size": int(base_masks[lane].numel()),
                "base_blocked_ids": base_masks[lane].nonzero().flatten().cpu().tolist(),
                "actions": traces[lane],
            },
            "history": clean[lane],
            "private_start": private_start,
            "prefix_length": prefix_length,
        })
    return result


def action_logps(logits: torch.Tensor, trace: dict) -> torch.Tensor:
    actions = trace["actions"]
    scaled = logits.float() / float(trace["temperature"])
    mask = torch.zeros_like(scaled, dtype=torch.bool)
    mask[:, trace["base_blocked_ids"]] = True
    for index, action in enumerate(actions):
        mask[index, action["additional_blocked_ids"]] = True
        mask[index, action["restored_ids"]] = False
    distribution = torch.log_softmax(scaled.masked_fill(mask, -torch.inf), dim=-1)
    ids = torch.tensor([action["token_id"] for action in actions], device=logits.device)
    selected = distribution.gather(1, ids[:, None]).squeeze(1)
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("non-finite selected policy log-probability")
    return selected


def pdo_loss(
    current: torch.Tensor, proximal: torch.Tensor, behavior: torch.Tensor,
    advantage: float, epsilon: float = 0.2, tis_cap: float = 2.0,
) -> torch.Tensor:
    """Eq. (4): clipped PPO loss with truncated behavior correction."""
    correction = (proximal.detach() - behavior.detach()).exp().clamp(max=tis_cap)
    ratio = (current - proximal.detach()).exp()
    signed = torch.as_tensor(advantage, dtype=current.dtype, device=current.device)
    terms = torch.minimum(
        ratio * signed,
        ratio.clamp(1 - epsilon, 1 + epsilon) * signed,
    )
    return -(correction * terms).sum()
