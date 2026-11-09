import torch
import pytest

from artemide.model.config import ModelConfig
from artemide_streaming import StreamingArtemideLM


def tiny():
    torch.manual_seed(19)
    return StreamingArtemideLM(
        ModelConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4, max_seq_len=32)
    ).eval()


def test_stream_yields_one_token_at_a_time_and_reconstructs_fast_output():
    model = tiny()
    prompt = torch.randint(0, 32, (2, 5))
    with torch.inference_mode():
        streamed = list(model.stream_generate(prompt, 6, temperature=0, use_cache=True))
        fast = model.generate_fast(prompt, 6, temperature=0, use_cache=True)
    assert len(streamed) == 6
    assert all(token.shape == (2, 1) for token in streamed)
    assert torch.equal(torch.cat([prompt, *streamed], dim=1), fast)


def test_streamed_cached_and_uncached_greedy_tokens_match():
    model = tiny()
    prompt = torch.randint(0, 32, (1, 7))
    with torch.inference_mode():
        cached = list(model.stream_generate(prompt, 5, temperature=0, use_cache=True))
        uncached = list(model.stream_generate(prompt, 5, temperature=0, use_cache=False))
    assert all(torch.equal(a, b) for a, b in zip(cached, uncached))


def test_stream_stops_at_context_limit():
    model = tiny()
    prompt = torch.randint(0, 32, (1, 30))
    assert len(list(model.stream_generate(prompt, 8, temperature=0))) == 2


def test_stream_rejects_invalid_inputs():
    model = tiny()
    with pytest.raises(ValueError):
        list(model.stream_generate(torch.ones((1, 2), dtype=torch.float), 2))
    with pytest.raises(ValueError):
        list(model.stream_generate(torch.ones((1, 2), dtype=torch.long), 2, temperature=-1))
