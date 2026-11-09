import json
import torch
import pytest
from fastapi.testclient import TestClient
from artemide import ArtemideLM, ModelConfig, ByteTokenizer
from artemide.data import prepare_corpus, TokenStream
from artemide.training import save_checkpoint, load_checkpoint
from artemide.trainer import build_scheduler, validate
from artemide.inference import generate_text, stream_text
from artemide.api import create_app

def tiny():
    torch.manual_seed(3)
    return ArtemideLM(ModelConfig(vocab_size=259,d_model=32,n_layers=1,n_heads=4,max_seq_len=32)).eval()

def test_tokenizer_unicode_round_trip():
    tokenizer=ByteTokenizer()
    text="Artemide नमस्ते 🌻"
    ids=tokenizer.encode(text,add_bos=True,add_eos=True)
    assert tokenizer.decode(ids)==text
    assert ids[0]==1 and ids[-1]==2

def test_model_forward_and_cached_generation():
    model=tiny()
    prompt=torch.tensor([[1,ord("h")+3,ord("i")+3]])
    logits=model(prompt)
    assert logits.shape==(1,3,259)
    cached=model.generate(prompt,4,temperature=0,use_cache=True)
    plain=model.generate(prompt,4,temperature=0,use_cache=False)
    assert torch.equal(cached,plain)
    assert cached.shape==(1,7)

def test_streaming_matches_non_streaming_greedy_and_is_utf8_safe():
    model=tiny()
    tokenizer=ByteTokenizer()
    prompt="Artemide stream test"
    expected=generate_text(model,tokenizer,prompt,device=torch.device("cpu"),max_new_tokens=8,temperature=0)
    actual="".join(stream_text(model,tokenizer,prompt,device=torch.device("cpu"),max_new_tokens=8,temperature=0))
    assert actual==expected

def test_corpus_prepare_and_stream(tmp_path):
    source=tmp_path/"src"; source.mkdir()
    (source/"a.txt").write_text("alpha",encoding="utf-8")
    (source/"b.txt").write_text("beta",encoding="utf-8")
    meta=prepare_corpus(source,tmp_path/"data",validation_fraction=0.5,seed=2,license_name="test fixture")
    assert meta["vocab_size"]==259
    stream=TokenStream(tmp_path/"data"/"train.bin",259)
    x,y=stream.sample(2,2,torch.device("cpu"),torch.Generator().manual_seed(1))
    assert x.shape==(2,2) and torch.equal(x[:,1:],y[:,:1])

def test_checkpoint_round_trip(tmp_path):
    model=tiny(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    path=tmp_path/"checkpoint.pt"
    save_checkpoint(path,model,optimizer,step=9,metadata={"tag":"test"})
    restored=load_checkpoint(path,model,optimizer)
    assert restored["step"]==9 and restored["metadata"]["tag"]=="test"

def test_scheduler_decays_to_floor():
    p=torch.nn.Parameter(torch.tensor(1.0)); opt=torch.optim.SGD([p],lr=1.0)
    scheduler=build_scheduler(opt,total_steps=8,warmup_steps=2,min_lr_ratio=0.2)
    initial=opt.param_groups[0]["lr"]
    for _ in range(8):
        opt.step(); scheduler.step()
    assert initial==pytest.approx(0.5)
    assert opt.param_groups[0]["lr"]==pytest.approx(0.2)

def test_validation_restores_training_mode():
    model=tiny().train()
    class FakeData:
        path="fake"
        def sample(self,batch_size,sequence_length,device,generator=None):
            ids=torch.randint(0,259,(batch_size,sequence_length),device=device)
            return ids,ids
    metrics=validate(model,FakeData(),batches=2,batch_size=2,sequence_length=4,device=torch.device("cpu"),precision="fp32")
    assert model.training and metrics["val_loss"]>0

def test_openai_compatible_models_and_chat_completion():
    client=TestClient(create_app(tiny(),ByteTokenizer(),torch.device("cpu"),api_key="secret"))
    headers={"Authorization":"Bearer secret"}
    models=client.get("/v1/models",headers=headers)
    assert models.status_code==200 and models.json()["data"][0]["id"]=="artemide-core-1.9"
    response=client.post("/v1/chat/completions",headers=headers,json={
        "messages":[{"role":"user","content":"Hello"}],"max_tokens":3,"temperature":0
    })
    assert response.status_code==200
    body=response.json()
    assert body["object"]=="chat.completion"
    assert body["choices"][0]["message"]["role"]=="assistant"
    assert "content" in body["choices"][0]["message"]

def test_openai_compatible_stream_ends_with_done():
    client=TestClient(create_app(tiny(),ByteTokenizer(),torch.device("cpu"),api_key="secret"))
    response=client.post("/v1/chat/completions",headers={"Authorization":"Bearer secret"},json={
        "messages":[{"role":"user","content":"Hello"}],"max_tokens":3,"temperature":0,"stream":True
    })
    assert response.status_code==200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "chat.completion.chunk" in response.text
    assert "data: [DONE]" in response.text

def test_api_rejects_wrong_key():
    client=TestClient(create_app(tiny(),ByteTokenizer(),torch.device("cpu"),api_key="secret"))
    response=client.get("/v1/models",headers={"Authorization":"Bearer wrong"})
    assert response.status_code==401
