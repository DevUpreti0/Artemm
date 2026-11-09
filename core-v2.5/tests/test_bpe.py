import json
import pytest
from artemide.bpe_tokenizer import BPETokenizer
from artemide.data import prepare_corpus, TokenStream
import torch

def test_bpe_train_unicode_roundtrip_and_special_tokens():
    texts=["hello hello hello", "hello world hello", "नमस्ते 🌻 नमस्ते"]
    tok=BPETokenizer.train(texts,vocab_size=280,min_frequency=2)
    assert tok.vocab_size>259
    ids=tok.encode("hello नमस्ते 🌻",add_bos=True,add_eos=True)
    assert ids[0]==1 and ids[-1]==2
    assert tok.decode(ids)=="hello नमस्ते 🌻"

def test_bpe_save_load_preserves_encoding(tmp_path):
    tok=BPETokenizer.train(["repeat repeat repeat","repeat token"],vocab_size=270,min_frequency=2)
    path=tmp_path/"tokenizer.json"; tok.save(path)
    restored=BPETokenizer.load(path)
    assert restored.vocab_size==tok.vocab_size
    assert restored.encode("repeat token")==tok.encode("repeat token")

def test_bpe_corpus_prep_writes_model_compatible_ids(tmp_path):
    src=tmp_path/"src"; src.mkdir()
    (src/"one.txt").write_text("repeat repeat",encoding="utf-8")
    (src/"two.txt").write_text("repeat token",encoding="utf-8")
    tok=BPETokenizer.train(["repeat repeat","repeat token"],vocab_size=270,min_frequency=2)
    manifest=prepare_corpus(src,tmp_path/"out",validation_fraction=0.5,tokenizer=tok,license_name="test fixture")
    assert manifest["vocab_size"]==tok.vocab_size
    stream=TokenStream(tmp_path/"out"/"train.bin",tok.vocab_size)
    x,y=stream.sample(1,1,torch.device("cpu"))
    assert x.shape==(1,1) and y.shape==(1,1)

def test_bpe_rejects_invalid_vocab_size():
    with pytest.raises(ValueError):
        BPETokenizer.train(["x"],vocab_size=70000)
