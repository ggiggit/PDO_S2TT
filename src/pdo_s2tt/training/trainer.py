"""On-policy G=4 PDO rollouts and distributed policy updates."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import math
import threading
from types import MethodType

import torch
import torch.distributed as dist

from pdo_s2tt.audio import packets, read_wav
from pdo_s2tt.history import visible_ids
from .model import TrainingModel
from .policy import ConditionedModel, action_logps, pdo_loss, policy_from_decoder, sample_group
from .reward import persistent_delivery_returns


@contextmanager
def _instance_methods(instance, replacements):
    missing = object()
    previous = {name: instance.__dict__.get(name, missing) for name in replacements}
    try:
        for name, function in replacements.items():
            setattr(instance, name, MethodType(function, instance))
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                instance.__dict__.pop(name, None)
            else:
                setattr(instance, name, value)


def rollout(model: TrainingModel, row: dict, seed: int) -> dict:
    """Sample four revision trajectories from one real streaming utterance."""
    translator, decoder, llm = model.translator, model.decoder, model.llm
    tokenizer, adapter = model.tokenizer, model.history
    tail = [
        *tokenizer("\n", add_special_tokens=False)["input_ids"],
        *decoder.turn_close,
        *decoder.assistant_open,
    ]
    old_start = decoder.start
    old_forward = decoder._forward
    old_append = decoder._append_ids
    state = {
        "groups": [],
        "prefixes": [],
        "boundaries": [],
        "histories": [[], [], [], []],
    }
    device = llm.get_input_embeddings().weight.device
    generators = [torch.Generator(device=device).manual_seed(seed + lane) for lane in range(4)]

    def start(current):
        captured = []
        def observe(module, args, kwargs):
            ids, embeddings = kwargs.get("input_ids"), kwargs.get("inputs_embeds")
            value = module.get_input_embeddings()(ids) if embeddings is None else embeddings
            captured.append(value.detach().cpu().clone())
        handle = llm.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            result = old_start()
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("failed to capture the native initial decoder prefix")
        current._pdo_input_buffer = captured[0]
        current._pdo_private_start = None
        return result

    def forward(current, *, target, input_ids=None, inputs_embeds=None, return_all_logits=False):
        if not target:
            raise ValueError("PDO training uses no source-text decoder branch")
        length = current._branch(True).attention_length
        previous = current._pdo_input_buffer
        value = llm.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        current._pdo_input_buffer = torch.cat(
            (previous[:, :length], value.detach().cpu().clone()), dim=1,
        )
        return old_forward(
            target=target,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            return_all_logits=return_all_logits,
        )

    def append(current, ids, *, target):
        if target and list(ids) == tail:
            current._pdo_private_start = int(current._branch(True).attention_length)
        return old_append(ids, target=target)

    def branch(current, logits, *, target, is_final, **kwargs):
        if not target or not is_final or kwargs.get("generated_prefix"):
            raise ValueError("unexpected decoder branch during PDO rollout")
        prefix = current._pdo_input_buffer.to(device)
        boundary = current._pdo_private_start
        if not isinstance(boundary, int) or prefix.shape[1] != current._branch(True).attention_length:
            raise RuntimeError("native private-history boundary was not observed")
        group = sample_group(
            llm, adapter, tokenizer, prefix, policy_from_decoder(current),
            generators, state["histories"], boundary,
        )
        state["groups"].append(group)
        state["prefixes"].append(prefix.detach().cpu())
        state["boundaries"].append(boundary)
        state["histories"] = [
            visible_ids(tokenizer, item["prefix_token_ids"], adapter.blocks[0].max_tokens)
            for item in group
        ]
        return [], False

    model.eval_mode()
    with _instance_methods(decoder, {
        "start": start,
        "_forward": forward,
        "_append_ids": append,
        "_decode_branch_cached": branch,
    }), torch.inference_mode():
        model.prepare(row["target_lang"])
        observed = []
        for pcm, final in packets(read_wav(row["audio"]), 0.1):
            observed.extend(translator.accept_pcm(pcm, final=final))
    times = [float(event.audio_time) for event in observed]
    if len(times) != len(state["groups"]) or not times:
        raise RuntimeError("streaming event grid and sampled drafts differ")
    trajectories = []
    for lane in range(4):
        trajectories.append({
            "drafts": [
                tokenizer.decode(
                    group[lane]["prefix_token_ids"],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ).strip()
                for group in state["groups"]
            ],
            "times": times,
        })
    advantages = persistent_delivery_returns(
        trajectories, row["target_text"], row["target_lang"],
    )
    events = []
    for event, group in enumerate(state["groups"]):
        events.append({
            "uid": row["id"],
            "event": event,
            "prefix": state["prefixes"][event],
            "private_start": state["boundaries"][event],
            "transports": group,
            "advantages": [advantages[lane][event] for lane in range(4)],
        })
    return {"events": events, "trajectories": trajectories}


def _event_logps(model: TrainingModel, event: dict) -> list[torch.Tensor]:
    llm, adapter = model.llm, model.history
    device = llm.get_input_embeddings().weight.device
    prefix = event["prefix"].to(device)
    transports = event["transports"]
    token_lists = [[action["token_id"] for action in item["trace"]["actions"]] for item in transports]
    maximum = max(map(len, token_lists))
    padded = torch.full((4, maximum), transports[0]["eot_id"], dtype=torch.long, device=device)
    for lane, tokens in enumerate(token_lists):
        padded[lane, :len(tokens)] = torch.tensor(tokens, device=device)
    with torch.no_grad():
        suffix = llm.get_input_embeddings()(padded[:, :-1])
    inputs = torch.cat((prefix.expand(4, -1, -1), suffix), dim=1)
    if torch.is_grad_enabled():
        inputs.requires_grad_(True)
    length = inputs.shape[1]
    conditioned = ConditionedModel(
        llm, adapter, [item["history"] for item in transports], event["private_start"],
    )
    output = conditioned(
        inputs_embeds=inputs,
        attention_mask=torch.ones(4, length, dtype=torch.long, device=device),
        position_ids=torch.arange(length, device=device).view(1, 1, length).expand(3, 4, length),
        use_cache=False,
        return_dict=True,
    )
    start = prefix.shape[1] - 1
    return [
        action_logps(output.logits[lane, start:start + len(tokens)], transports[lane]["trace"])
        for lane, tokens in enumerate(token_lists)
    ]


def capture_proximal(model: TrainingModel, events: list[dict]) -> None:
    model.eval_mode()
    with torch.no_grad():
        for event in events:
            event["proximal"] = [value.detach().cpu() for value in _event_logps(model, event)]
            maximum = max(
                abs(float(saved) - action["old_logp"])
                for lane, transport in enumerate(event["transports"])
                for saved, action in zip(event["proximal"][lane], transport["trace"]["actions"])
            )
            if maximum > 5e-3:
                raise RuntimeError(f"sampler/replay log-probability mismatch: {maximum:.6f}")


def _all_reduce_gradients(parameters: list[torch.nn.Parameter]) -> None:
    if not dist.is_initialized():
        return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


@contextmanager
def _history_checkpointing(model: TrainingModel):
    """Keep the per-event history context active during checkpoint recompute."""
    llm, adapter = model.llm, model.history
    original = llm.gradient_checkpointing_enable
    stats = {"forward": 0, "recompute": 0}

    def context_factory():
        memory = adapter._context.get()
        if memory is None:
            raise RuntimeError("gradient checkpoint was created outside a history condition")

        @contextmanager
        def enter(kind):
            stats[kind] += 1
            with adapter.condition(memory):
                yield
        return enter("forward"), enter("recompute")

    def enable(current, gradient_checkpointing_kwargs=None):
        kwargs = dict(gradient_checkpointing_kwargs or {})
        kwargs.update(use_reentrant=False, context_fn=context_factory)
        return original(gradient_checkpointing_kwargs=kwargs)

    llm.gradient_checkpointing_enable = MethodType(enable, llm)
    llm.gradient_checkpointing_enable()
    try:
        yield stats
    finally:
        llm.gradient_checkpointing_disable()
        llm.gradient_checkpointing_enable = original


def update(
    model: TrainingModel, events: list[dict], global_utterances: int,
) -> dict:
    """Make one synchronized AdamW update from complete local utterances."""
    if global_utterances <= 0:
        raise ValueError("positive global minibatch size required")
    model.train_mode()
    model.optimizer.zero_grad(set_to_none=True)
    scale = 1.0 / (4 * global_utterances)
    total = 0.0
    with _history_checkpointing(model) as checkpoint_stats:
        for event in events:
            current = _event_logps(model, event)
            event_loss = torch.zeros((), device=current[0].device)
            for lane, values in enumerate(current):
                proximal = event["proximal"][lane].to(values.device, values.dtype)
                behavior = torch.tensor(
                    [action["old_logp"] for action in event["transports"][lane]["trace"]["actions"]],
                    device=values.device,
                    dtype=values.dtype,
                )
                loss = pdo_loss(
                    values, proximal, behavior, event["advantages"][lane],
                ) * scale
                event_loss = event_loss + loss
                total += float(loss.detach())
            event_loss.backward()
    _all_reduce_gradients(model.parameters)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters, 1.0, error_if_nonfinite=True)
    model.optimizer.step()
    return {
        "loss": total,
        "gradient_norm_before_clip": float(norm),
        "checkpoint_recomputations": checkpoint_stats["recompute"],
    }
