"""Decoder-only transformer with incremental KV-cached generation."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .config import ModelConfig
from artemide.inference_cache import KVCache

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.weight=nn.Parameter(torch.ones(dim)); self.eps=eps
    def forward(self,x):
        return x*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+self.eps).to(x.dtype)*self.weight

def _rotary(x, positions):
    d=x.shape[-1]
    if d%2: raise ValueError("attention head dimension must be even")
    inv=1.0/(10000**(torch.arange(0,d,2,device=x.device,dtype=torch.float32)/d))
    angle=positions.float().view(1,1,-1,1)*inv.view(1,1,1,-1)
    co,si=angle.cos().to(x.dtype),angle.sin().to(x.dtype)
    return torch.stack((x[...,0::2]*co-x[...,1::2]*si,x[...,0::2]*si+x[...,1::2]*co),-1).flatten(-2)

class CausalSelfAttention(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.heads=cfg.n_heads; self.dim=cfg.d_model//cfg.n_heads
        self.qkv=nn.Linear(cfg.d_model,3*cfg.d_model,bias=False); self.out=nn.Linear(cfg.d_model,cfg.d_model,bias=False); self.dropout=cfg.dropout
    def forward(self,x,layer_idx,cache=None,use_cache=False):
        b,t,c=x.shape; q,k,v=self.qkv(x).chunk(3,-1)
        shape=(b,t,self.heads,self.dim)
        q,k,v=[z.view(shape).transpose(1,2) for z in (q,k,v)]
        past=cache.sequence_length(layer_idx) if cache is not None and cache.get(layer_idx) is not None else 0
        pos=torch.arange(past,past+t,device=x.device); q,k=_rotary(q,pos),_rotary(k,pos)
        if use_cache:
            if cache is None: raise ValueError("cache required when use_cache=True")
            ka,va=cache.update(layer_idx,k,v)
        else: ka,va=k,v
        qpos=torch.arange(past,past+t,device=x.device); kpos=torch.arange(ka.shape[-2],device=x.device)
        allowed=kpos[None,:]<=qpos[:,None]
        mask=torch.zeros((t,ka.shape[-2]),device=x.device,dtype=q.dtype).masked_fill(~allowed,float("-inf"))
        y=F.scaled_dot_product_attention(q,ka,va,attn_mask=mask,dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1,2).contiguous().view(b,t,c))

class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__(); h=cfg.d_model*cfg.ff_mult
        self.gate=nn.Linear(cfg.d_model,h,bias=False); self.up=nn.Linear(cfg.d_model,h,bias=False); self.down=nn.Linear(h,cfg.d_model,bias=False)
    def forward(self,x): return self.down(F.silu(self.gate(x))*self.up(x))

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.n1=RMSNorm(cfg.d_model); self.n2=RMSNorm(cfg.d_model); self.attn=CausalSelfAttention(cfg); self.ff=FeedForward(cfg)
    def forward(self,x,i,cache=None,use_cache=False):
        x=x+self.attn(self.n1(x),i,cache,use_cache); return x+self.ff(self.n2(x))

class ArtemideLM(nn.Module):
    def __init__(self,config:ModelConfig):
        super().__init__(); self.config=config
        self.token_embedding=nn.Embedding(config.vocab_size,config.d_model)
        self.blocks=nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm=RMSNorm(config.d_model); self.lm_head=nn.Linear(config.d_model,config.vocab_size,bias=False)
        self.lm_head.weight=self.token_embedding.weight; self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,mean=0,std=.02)
    def forward(self,input_ids,cache=None,use_cache=False):
        if input_ids.ndim!=2 or input_ids.shape[1]<1: raise ValueError("input_ids must be non-empty [batch, sequence]")
        past=cache.sequence_length() if cache is not None and len(cache) else 0
        if past+input_ids.shape[1]>self.config.max_seq_len: raise ValueError("input plus cache exceeds max_seq_len")
        x=self.token_embedding(input_ids)
        for i,block in enumerate(self.blocks): x=block(x,i,cache,use_cache)
        return self.lm_head(self.norm(x))
    @torch.inference_mode()
    def generate(self,input_ids,max_new_tokens=128,temperature=.8,top_k=40,use_cache=True):
        if max_new_tokens<0 or top_k<0 or not math.isfinite(temperature) or temperature<0: raise ValueError("invalid generation settings")
        if input_ids.ndim!=2 or input_ids.shape[1]<1: raise ValueError("input_ids must be non-empty [batch, sequence]")
        if input_ids.shape[1]>self.config.max_seq_len: input_ids=input_ids[:,-self.config.max_seq_len:]
        out=input_ids; cache=KVCache(max_length=self.config.max_seq_len) if use_cache else None
        logits=self.forward(out,cache=cache,use_cache=use_cache)[:,-1,:]
        for _ in range(max_new_tokens):
            if temperature==0: nxt=logits.argmax(-1,keepdim=True)
            else:
                scores=logits/temperature
                if top_k:
                    k=min(top_k,scores.shape[-1]); cutoff=torch.topk(scores,k,dim=-1).values[...,-1,None]
                    scores=scores.masked_fill(scores<cutoff,float("-inf"))
                nxt=torch.multinomial(F.softmax(scores,dim=-1),1)
            out=torch.cat((out,nxt),1)
            if out.shape[1]>=self.config.max_seq_len: break
            logits=(self.forward(nxt,cache=cache,use_cache=True) if use_cache else self.forward(out[:,-self.config.max_seq_len:]))[:,-1,:]
        return out
