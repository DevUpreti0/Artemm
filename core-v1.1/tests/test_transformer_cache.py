import torch
import pytest
from artemide.model.config import ModelConfig
from artemide.model.transformer import ArtemideLM

def tiny():
    torch.manual_seed(7)
    return ArtemideLM(ModelConfig(vocab_size=32,d_model=32,n_layers=2,n_heads=4,max_seq_len=32)).eval()

def test_cached_incremental_logits_match_full_forward():
    model=tiny(); ids=torch.randint(0,32,(2,9))
    with torch.inference_mode():
        from artemide.inference_cache import KVCache
        cache=KVCache(); model(ids[:,:6],cache=cache,use_cache=True)
        inc=model(ids[:,6:],cache=cache,use_cache=True); full=model(ids)
    assert torch.allclose(inc,full[:,6:,:],atol=1e-5,rtol=1e-5)

def test_cached_and_uncached_greedy_generation_match():
    model=tiny(); ids=torch.randint(0,32,(1,7))
    with torch.inference_mode():
        a=model.generate(ids,5,temperature=0,use_cache=True)
        b=model.generate(ids,5,temperature=0,use_cache=False)
    assert torch.equal(a,b)

def test_context_limit_rejected():
    with pytest.raises(ValueError): tiny()(torch.ones((1,33),dtype=torch.long))

def test_odd_head_dim_rejected():
    with pytest.raises(ValueError): ModelConfig(vocab_size=10,d_model=33,n_heads=3)
