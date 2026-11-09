import json
import torch
import pytest
from fastapi.testclient import TestClient
from artemide import ArtemideLM, ByteTokenizer, ModelConfig
from artemide.data import TokenStream, split_documents
from artemide.api import app


def tiny():
    return ArtemideLM(ModelConfig(vocab_size=259,d_model=32,n_layers=1,n_heads=2,max_seq_len=32,ff_mult=2))


def test_package_import_and_unicode_tokenizer():
    tokenizer = ByteTokenizer()
    text = "Artemide नमस्ते 🌻"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tiny()(torch.tensor([[1, 4, 5]])).shape == (1, 3, 259)


def test_cached_greedy_generation_matches_uncached():
    torch.manual_seed(5)
    model = tiny().eval()
    prompt = torch.tensor([[1, 40, 50]])
    assert torch.equal(model.generate(prompt, 5, temperature=0, use_cache=True),
                       model.generate(prompt, 5, temperature=0, use_cache=False))


def test_split_is_reproducible_and_disjoint():
    docs = [(str(i), str(i)) for i in range(10)]
    train_a, val_a = split_documents(docs, .2, 7)
    train_b, val_b = split_documents(docs, .2, 7)
    assert train_a == train_b and val_a == val_b
    assert {x[0] for x in train_a}.isdisjoint({x[0] for x in val_a})


def test_token_stream_little_endian(tmp_path):
    path = tmp_path / "ids.bin"
    path.write_bytes(bytes([3,0, 4,0, 5,0, 6,0]))
    x, y = TokenStream(path, 259).sample(1, 2, torch.device("cpu"), torch.Generator().manual_seed(1))
    assert x.shape == y.shape == (1,2)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["model"] == "Artemide"
