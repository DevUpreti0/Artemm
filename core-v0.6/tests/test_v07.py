import pytest
from artemide.chat import ChatMessage, format_chat_prompt
from artemide.model.config import ModelConfig
from artemide.model.transformer import ArtemideLM
from artemide.training.evaluate import evaluate_text

def tiny_model():
    cfg = ModelConfig(vocab_size=259, d_model=16, n_layers=1, n_heads=2,
                      n_kv_heads=2, max_seq_len=8, dropout=0.2)
    return ArtemideLM(cfg)

def test_chat_serialization_and_assistant_prefix():
    prompt = format_chat_prompt([ChatMessage("system", "Be concise."), ChatMessage("user", "Hello")])
    assert prompt == "<|system|>\nBe concise.\n<|user|>\nHello\n<|assistant|>\n"

def test_chat_rejects_unknown_role():
    with pytest.raises(ValueError, match="Unsupported chat role"):
        format_chat_prompt([ChatMessage("tool", "run")])

def test_chat_rejects_empty_messages():
    with pytest.raises(ValueError, match="At least one"):
        format_chat_prompt([])

def test_eval_returns_metrics_and_restores_mode():
    model = tiny_model()
    model.train()
    result = evaluate_text(model, "Artemide evaluation", seq_len=8)
    assert result["tokens"] == len("Artemide evaluation".encode("utf-8")) - 1
    assert result["loss"] > 0 and result["perplexity"] >= 1
    assert model.training

def test_eval_rejects_context_overflow():
    with pytest.raises(ValueError, match="max_seq_len"):
        evaluate_text(tiny_model(), "Long enough text", seq_len=9)
