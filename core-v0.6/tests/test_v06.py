import json
import pytest
import torch
from artemide.training.corpus import prepare_corpus
from artemide.training.train import get_batch
from artemide.tokenizer import ByteTokenizer

def test_prepare_real_text_corpus(tmp_path):
    source=tmp_path/"source"; source.mkdir()
    (source/"a.txt").write_text("Artemide learns from real text.",encoding="utf-8")
    (source/"b.jsonl").write_text(json.dumps({"text":"Independent model training."})+"\n",encoding="utf-8")
    manifest=prepare_corpus(source,tmp_path/"out",val_fraction=0.5,seed=7,license_name="CC0")
    assert manifest["documents"]==2
    assert (tmp_path/"out"/"train.bin").exists()
    assert manifest["train"]["tokens"]>2
    assert manifest["tokenizer"]=="ByteTokenizer"

def test_batch_sampling_from_real_token_tensor():
    tokens=torch.arange(100,dtype=torch.long)
    x,y=get_batch(tokens,4,12)
    assert x.shape==(4,12) and y.shape==(4,12)
    assert torch.equal(y[:,:-1],x[:,1:])
    assert torch.equal(y[:,-1],x[:,-1]+1)

def test_batch_sampling_accepts_exact_minimum_dataset():
    tokens=torch.arange(13,dtype=torch.long)
    x,y=get_batch(tokens,3,12)
    assert torch.equal(x,torch.arange(12).repeat(3,1))
    assert torch.equal(y,torch.arange(1,13).repeat(3,1))

def test_batch_sampling_rejects_too_short_dataset():
    with pytest.raises(ValueError,match="at least seq_len"):
        get_batch(torch.arange(12),2,12)

def test_tokenizer_vocab_config():
    assert ByteTokenizer.vocab_size==259
