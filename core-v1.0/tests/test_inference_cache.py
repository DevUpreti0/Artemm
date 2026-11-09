import pytest
import torch

from artemide.inference_cache import KVCache


def test_append_and_read():
    cache = KVCache()
    k = torch.randn(2, 4, 3, 8)
    v = torch.randn(2, 4, 3, 8)
    out_k, out_v = cache.update(0, k, v)
    assert torch.equal(out_k, k)
    assert torch.equal(out_v, v)
    assert cache.sequence_length(0) == 3


def test_incremental_append():
    cache = KVCache()
    first = torch.randn(1, 2, 3, 4)
    second = torch.randn(1, 2, 1, 4)
    cache.update(0, first, first)
    out, _ = cache.update(0, second, second)
    assert out.shape == (1, 2, 4, 4)
    assert torch.equal(out[..., :3, :], first)


def test_bounded_cache_keeps_recent_tokens():
    cache = KVCache(max_length=3)
    a = torch.arange(4.).view(1, 1, 2, 2)
    b = torch.arange(4., 8.).view(1, 1, 2, 2)
    cache.update(0, a, a)
    out, _ = cache.update(0, b, b)
    assert out.shape[-2] == 3
    assert torch.equal(out, torch.cat((a, b), dim=-2)[..., -3:, :])


def test_invalid_update_does_not_mutate():
    cache = KVCache()
    a = torch.randn(1, 2, 2, 4)
    cache.update(0, a, a)
    with pytest.raises(ValueError):
        cache.update(0, torch.randn(1, 3, 1, 4), torch.randn(1, 3, 1, 4))
    assert cache.sequence_length(0) == 2


def test_clear_and_validation():
    with pytest.raises(ValueError):
        KVCache(0)
    cache = KVCache()
    a = torch.randn(1, 1, 1, 2)
    cache.update(0, a, a)
    cache.clear()
    assert len(cache) == 0
    assert cache.sequence_length() == 0
