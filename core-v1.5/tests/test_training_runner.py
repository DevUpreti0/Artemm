import math
import torch
import pytest

from artemide_training_runner import TokenStream, build_scheduler, validate


def test_scheduler_warmup_and_cosine_end():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = build_scheduler(
        optimizer, total_steps=10, warmup_steps=2, min_lr_ratio=0.2
    )
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] == pytest.approx(0.5)
    assert rates[-1] == pytest.approx(0.2)
    assert rates[2] == pytest.approx(1.0)


def test_token_stream_samples_shifted_targets(tmp_path):
    path = tmp_path / "tokens.bin"
    torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.uint16).numpy().tofile(path)
    stream = TokenStream(path, vocab_size=10)
    x, y = stream.sample(2, 3, torch.device("cpu"), torch.Generator().manual_seed(1))
    assert x.shape == (2, 3)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.all((y >= 1) & (y <= 6))


def test_token_stream_rejects_vocab_overflow(tmp_path):
    path = tmp_path / "tokens.bin"
    torch.tensor([1, 2, 9], dtype=torch.uint16).numpy().tofile(path)
    with pytest.raises(ValueError, match="outside model vocabulary"):
        TokenStream(path, vocab_size=8)


def test_validation_restores_training_mode():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 8))
            self.config = type("Config", (), {"max_seq_len": 4})()
        def forward(self, ids):
            return self.weight[ids]

    class FakeStream:
        path = "synthetic"
        def sample(self, batch_size, sequence_length, device, generator=None):
            ids = torch.randint(0, 8, (batch_size, sequence_length), device=device)
            return ids, ids

    model = Tiny()
    model.train()
    metrics = validate(
        model, FakeStream(), batches=2, batch_size=2, sequence_length=3,
        device=torch.device("cpu"), precision="fp32",
    )
    assert model.training is True
    assert math.isfinite(metrics["val_loss"])
    assert metrics["val_perplexity"] >= 1.0
