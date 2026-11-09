import json, sys, torch
from artemide.cli import main

def test_init_command_writes_random_checkpoint(tmp_path, monkeypatch):
    config=tmp_path/"small.json"
    config.write_text(json.dumps({"vocab_size":259,"d_model":32,"n_layers":1,
        "n_heads":4,"max_seq_len":32,"dropout":0.0,"ff_mult":4}),encoding="utf-8")
    output=tmp_path/"initial.pt"
    monkeypatch.setattr(sys,"argv",["artemide","init","--config",str(config),"--output",str(output),"--seed","7"])
    main()
    state=torch.load(output,map_location="cpu",weights_only=False)
    assert state["step"]==0
    assert state["metadata"]=={"stage":"random-initialization","seed":7}
    assert "model" in state
