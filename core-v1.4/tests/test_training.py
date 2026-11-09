import torch
import pytest

from artemide.model.config import ModelConfig
from artemide.model.transformer import ArtemideLM
from artemide_training import train_batch, save_checkpoint, load_checkpoint


def tiny():
    torch.manual_seed(23)
    return ArtemideLM(ModelConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4, max_seq_len=32))


def test_train_batch_updates_parameters_and_reports_metrics():
    model = tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    before = model.token_embedding.weight.detach().clone()
    metrics = train_batch(model, optimizer, x, y, precision="fp32")
    assert metrics["loss"] > 0
    assert metrics["grad_norm"] >= 0
    assert not torch.equal(before, model.token_embedding.weight)


def test_checkpoint_restores_model_optimizer_and_step(tmp_path):
    model = tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 32, (1, 5))
    y = torch.randint(0, 32, (1, 5))
    train_batch(model, optimizer, x, y, precision="fp32")
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    path = tmp_path / "nested" / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, step=17, metadata={"run": "unit-test"})
    restored = tiny()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    result = load_checkpoint(path, restored, restored_optimizer, restore_rng=False)
    assert result == {"step": 17, "metadata": {"run": "unit-test"}}
    for key, value in expected.items():
        assert torch.equal(value, restored.state_dict()[key])


def test_train_batch_rejects_shape_mismatch():
    model = tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError):
        train_batch(model, optimizer, torch.ones((1, 4), dtype=torch.long), torch.ones((1, 3), dtype=torch.long))


def test_load_rejects_invalid_checkpoint(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"format_version": 99}, path)
    with pytest.raises(ValueError):
        load_checkpoint(path, tiny(), restore_rng=False)
