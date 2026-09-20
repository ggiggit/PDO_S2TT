"""Function-preserving previous-display memory for a private rewrite branch.

Independent cross-attention residuals borrow Flamingo's zero tanh gate.
No tokens/positions are inserted in the native LM sequence. Not a trained model.
Training must keep condition() active through backward if checkpointing is used.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass,field
import math
from types import MethodType

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Memory:
    embeddings: torch.Tensor
    query_mask: torch.Tensor | None = None
    inference_kv: dict = field(default_factory=dict)


class HistoryCrossAttention(nn.Module):
    def __init__(self,hidden_size,inner_size=256,heads=4,max_tokens=128):
        super().__init__()
        if hidden_size<=0 or inner_size<=0 or heads<=0 or inner_size%heads or max_tokens<=0:
            raise ValueError('invalid memory dimensions')
        self.hidden_size,self.inner_size,self.heads,self.max_tokens=hidden_size,inner_size,heads,max_tokens
        self.query=nn.Linear(hidden_size,inner_size,bias=False)
        self.key=nn.Linear(hidden_size,inner_size,bias=False)
        self.value=nn.Linear(hidden_size,inner_size,bias=False)
        self.output=nn.Linear(inner_size,hidden_size,bias=False)
        self.position=nn.Parameter(torch.empty(max_tokens,hidden_size))
        nn.init.normal_(self.position,std=.02)
        # Only the gate is zero; zeroing output as well would kill all initial gradients.
        self.gate=nn.Parameter(torch.zeros(()))

    def forward(self,hidden,memory):
        x=memory.embeddings
        if (hidden.ndim!=3 or x.ndim!=3 or hidden.shape[0]!=x.shape[0]
                or hidden.shape[-1]!=self.hidden_size or x.shape[-1]!=self.hidden_size
                or x.shape[1]>self.max_tokens or x.device!=hidden.device):
            raise ValueError('history/hidden shape or device mismatch')
        if x.shape[1]==0:return hidden
        if memory.query_mask is not None:
            if memory.query_mask.dtype!=torch.bool or tuple(memory.query_mask.shape)!=tuple(hidden.shape[:2]):
                raise ValueError('explicit query mask must match the current forward')
        # Parameters and attention math stay fp32; do not globally cast this
        # adapter to bf16 just because the frozen backbone runs in bf16.
        split=lambda z:z.reshape(z.shape[0],z.shape[1],self.heads,-1).transpose(1,2)
        cache=memory.inference_kv.get(id(self)) if not torch.is_grad_enabled() else None
        if cache is None:
            positioned=x.float()+self.position[:x.shape[1]].float()
            normalized=F.layer_norm(positioned,(self.hidden_size,))
            k=split(F.linear(normalized,self.key.weight.float()))
            v=split(F.linear(normalized,self.value.weight.float()))
            if not torch.is_grad_enabled():memory.inference_kv[id(self)]=(k,v)
        else:k,v=cache
        q=split(F.linear(F.layer_norm(hidden.float(),(self.hidden_size,)),self.query.weight.float()))
        attention=torch.softmax(q@k.transpose(-1,-2)/math.sqrt(self.inner_size//self.heads),dim=-1)
        attended=(attention@v).transpose(1,2).reshape(*hidden.shape[:2],self.inner_size)
        residual=F.linear(attended,self.output.weight.float())*torch.tanh(self.gate.float())
        if memory.query_mask is not None:residual=residual*memory.query_mask.to(hidden.device)[...,None]
        return (hidden.float()+residual).to(hidden.dtype)


class PrivateHistoryAdapter(nn.Module):
    def __init__(self,layer_names,hidden_size,inner_size=256,heads=4,max_tokens=128,init_seed=0):
        super().__init__()
        self.layer_names=tuple(layer_names)
        if not self.layer_names or len(set(self.layer_names))!=len(self.layer_names):raise ValueError('unique explicit layers required')
        self.init_seed=init_seed
        # Do not perturb the original policy's CPU/CUDA sampling streams.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(init_seed)
            self.blocks=nn.ModuleList([HistoryCrossAttention(hidden_size,inner_size,heads,max_tokens) for _ in layer_names])
        self._context=ContextVar('private_history_'+str(id(self)),default=None)
        self._handles=[]

    @contextmanager
    def condition(self,memory):
        if memory is not None and not isinstance(memory,Memory):raise ValueError('typed private memory required')
        token=self._context.set(memory)
        try:yield
        finally:self._context.reset(token)

    def attach(self,llm):
        if self._handles:raise ValueError('already attached')
        modules=dict(llm.named_modules())
        if any(name not in modules for name in self.layer_names):raise ValueError('decoder layer missing')
        def hook(block):
            def run(module,args,output):
                memory=self._context.get()
                if memory is None:return output
                if isinstance(output,torch.Tensor):return block(output,memory)
                if isinstance(output,tuple) and output and isinstance(output[0],torch.Tensor):
                    return (block(output[0],memory),*output[1:])
                raise TypeError('unsupported decoder block output')
            return run
        self._handles=[modules[name].register_forward_hook(hook(block)) for name,block in zip(self.layer_names,self.blocks)]
        return self

    def detach(self):
        for h in self._handles:h.remove()
        self._handles=[]


def visible_ids(tokenizer,ids,max_tokens=128):
    ids=list(ids)
    if len(ids)>max_tokens or any(type(i)!=int or i<0 for i in ids):raise ValueError('invalid previous action')
    clean=[i for i in ids if i not in set(tokenizer.all_special_ids)]
    decode=lambda v:tokenizer.decode(v,skip_special_tokens=True,clean_up_tokenization_spaces=True).strip()
    if decode(clean)!=decode(ids):raise ValueError('filtering changed visible history')
    return clean


def install_private_runtime(decoder,adapter):
    """Instance-only wrapper. Attach adapter to decoder.llm separately.

    The persistent branch object remains the same while native audio is appended.
    Only after native code switches to its deepcopy is history made visible.
    No mutation of the class, old checkpoints, native prompt, or persistent KV.
    """
    if getattr(decoder,'_private_history_v1',False):raise ValueError('already installed')
    original_draft=decoder._decode_stage5_temporary_draft
    original_forward=decoder._forward
    context=ContextVar('private_runtime_'+str(id(decoder)),default=None)
    stats=dict(drafts=0,persistent_forwards=0,private_forwards=0,nonempty_memory_drafts=0)
    def forward(self,*args,**kwargs):
        active=context.get()
        private=bool(active is not None and kwargs.get('target') is True and self._branch(True) is not active[0])
        if active is not None:stats['private_forwards' if private else 'persistent_forwards']+=1
        with adapter.condition(active[1] if private else None):
            return original_forward(*args,**kwargs)
    def draft(self,latent,*,is_final):
        state=self._require_state();persistent=state.target
        ids=visible_ids(self.tokenizer,state.target_ids,adapter.blocks[0].max_tokens)
        embedding=self.llm.get_input_embeddings()
        device=embedding.weight.device
        with torch.no_grad():
            memory=Memory(embedding(torch.tensor([ids],dtype=torch.long,device=device)).detach())
        token=context.set((persistent,memory));stats['drafts']+=1;stats['nonempty_memory_drafts']+=bool(ids)
        try:
            result=original_draft(latent,is_final=is_final)
            if not is_final and self._require_state().target is not persistent:
                raise ValueError('native private cache was retained before utterance end')
            return result
        finally:
            context.reset(token)
            memory.inference_kv.clear()
    decoder._forward=MethodType(forward,decoder)
    decoder._decode_stage5_temporary_draft=MethodType(draft,decoder)
    decoder._private_history_v1=True
    def remove():
        if context.get() is not None:raise ValueError('cannot uninstall active decoder')
        decoder._forward=original_forward;decoder._decode_stage5_temporary_draft=original_draft
        decoder._private_history_v1=False
    return stats,remove
