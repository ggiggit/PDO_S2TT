"""Owner-local direct topology, hook and captured-operation guards.

The direct path is equivalent to the original decoder hook/contract checks.
Any failed direct condition runs those original checks again. Parameter,
buffer and adapter state remain guarded by the separate signature registry.
"""
from __future__ import annotations

import threading
from torch import nn
from torch.nn.modules import module as module_runtime

_MODULES = nn.Module.modules
_NAMED_MODULES = nn.Module.named_modules


class DecoderGuardPlan:
    def __init__(self, layer, contract, hooks_clear):
        from qwen_asr.core.transformers_backend import modeling_qwen3_asr as qwen
        self.layer = layer
        self.contract = contract
        self.hooks_clear = hooks_clear
        self.expected = contract(layer)
        self.owner_thread = threading.get_ident()
        self.closed = False
        self.calls = self.fallbacks = 0
        self.generated_source = ''
        self._fast = lambda: False
        nodes = tuple(layer.modules())
        if (
            'modules' in layer.__dict__ or type(layer).modules is not _MODULES
            or any('named_modules' in node.__dict__ or type(node).named_modules is not _NAMED_MODULES for node in nodes)
        ):
            return
        attention = layer.self_attn
        namespace = {
            '_root': layer, '_nn': nn, '_modules': _MODULES,
            '_named_modules': _NAMED_MODULES, '_runtime': module_runtime,
            '_qwen': qwen, '_rope': qwen.apply_rotary_pos_emb,
            '_eager': qwen.eager_attention_forward,
            '_attention': attention,
            '_input_norm': layer.input_layernorm,
            '_post_norm': layer.post_attention_layernorm,
            '_mlp': layer.mlp, '_act': layer.mlp.act_fn,
            '_out_proj': attention.o_proj,
        }
        guards = [
            '_nn.Module.modules is _modules',
            '_nn.Module.named_modules is _named_modules',
            "'modules' not in _root.__dict__",
            'type(_root).modules is _modules',
            'not _runtime._global_forward_pre_hooks',
            'not _runtime._global_forward_hooks',
            '_qwen.apply_rotary_pos_emb is _rope',
            '_qwen.eager_attention_forward is _eager',
            '_mlp.act_fn is _act',
            '_root.input_layernorm is _input_norm',
            '_root.self_attn is _attention',
            '_root.post_attention_layernorm is _post_norm',
            '_root.mlp is _mlp',
            '_attention.o_proj is _out_proj',
        ]
        for field in ('layer_idx', 'head_dim', 'scaling', 'attention_dropout'):
            guards.append(f'_attention.{field} == {getattr(attention, field)!r}')
        guards.append(f'_attention.config._attn_implementation == {str(attention.config._attn_implementation)!r}')
        guards.append(f'_input_norm.variance_epsilon == {float(layer.input_layernorm.variance_epsilon)!r}')
        guards.append(f'_post_norm.variance_epsilon == {float(layer.post_attention_layernorm.variance_epsilon)!r}')
        for i, node in enumerate(nodes):
            key = f'_node{i}'
            namespace[key] = node
            guards.extend([
                f"'named_modules' not in {key}.__dict__",
                f'type({key}).named_modules is _named_modules',
                f'tuple({key}._modules) == {tuple(node._modules)!r}',
            ])
            for j, (name, child) in enumerate(node._modules.items()):
                ref = f'_child{i}_{j}'
                namespace[ref] = child
                guards.append(f'{key}._modules[{name!r}] is {ref}')
            if i == 0:
                continue
            guards.extend([f'not {key}._forward_pre_hooks', f'not {key}._forward_hooks'])
            if 'forward' in node.__dict__:
                ref = f'_forward{i}'
                namespace[ref] = node.__dict__['forward']
                guards.append(f'{key}.__dict__.get("forward") is {ref}')
            else:
                ref = f'_forward{i}'
                namespace[ref] = getattr(type(node), 'forward', None)
                guards.extend([f"'forward' not in {key}.__dict__", f'getattr(type({key}), "forward", None) is {ref}'])
        source = 'def _direct():\n    return (\n        ' + '\n        and '.join(guards) + '\n    )\n'
        exec(compile(source, '<pdo-decoder-guard-plan>', 'exec'), namespace)
        self.generated_source = source
        self._fast = namespace['_direct']
        if not self._fast():
            raise RuntimeError('decoder direct guard disagrees at construction')

    def __call__(self):
        if self.closed or threading.get_ident() != self.owner_thread:
            raise RuntimeError('decoder guard requires its live owner thread')
        self.calls += 1
        if self._fast():
            return True
        self.fallbacks += 1
        return self.hooks_clear(self.layer) and self.contract(self.layer) == self.expected

    def close(self):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError('decoder guard close requires its owner thread')
        if self.closed:
            return
        self._fast = None
        self.layer = self.contract = self.hooks_clear = self.expected = None
        self.generated_source = ''
        self.closed = True
