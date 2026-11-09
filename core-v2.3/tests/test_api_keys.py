import pytest
from artemide.auth import APIKeyStore, RateLimitExceeded

def test_create_authenticate_revoke_and_usage(tmp_path):
    store=APIKeyStore(tmp_path/"keys.sqlite3")
    created=store.create_key("test-app",requests_per_minute=10,monthly_token_limit=1000)
    identity=store.authenticate(created["api_key"])
    assert identity["key_id"]==created["id"]
    store.record_usage(identity,12,8)
    usage=store.usage(created["id"])
    assert usage["total_tokens"]==20
    assert store.list_keys()[0]["tokens_used"]==20
    assert store.revoke(created["id"])
    assert store.authenticate(created["api_key"]) is None

def test_rate_limit_is_enforced(tmp_path):
    store=APIKeyStore(tmp_path/"keys.sqlite3")
    created=store.create_key("limited",requests_per_minute=1)
    store.authenticate(created["api_key"])
    with pytest.raises(RateLimitExceeded):
        store.authenticate(created["api_key"])

def test_admin_key_lifecycle_and_usage_api(tmp_path):
    import torch
    from fastapi.testclient import TestClient
    from artemide import ArtemideLM, ModelConfig, ByteTokenizer
    from artemide.api import create_app

    torch.manual_seed(1)
    model=ArtemideLM(ModelConfig(vocab_size=259,d_model=32,n_layers=1,n_heads=4,max_seq_len=32)).eval()
    client=TestClient(create_app(model,ByteTokenizer(),torch.device("cpu"),
        admin_api_key="admin-secret",key_db_path=tmp_path/"integration.sqlite3"))
    created=client.post("/v1/admin/api-keys",headers={"X-Admin-Key":"admin-secret"},
        json={"label":"integration-test","requests_per_minute":20,"monthly_token_limit":1000})
    assert created.status_code==200
    data=created.json()
    assert data["api_key"].startswith("art_")
    headers={"Authorization":"Bearer "+data["api_key"]}
    assert client.get("/v1/models",headers=headers).status_code==200
    chat=client.post("/v1/chat/completions",headers=headers,json={
        "model":"artemide-core-2.0","messages":[{"role":"user","content":"hi"}],
        "max_tokens":2,"temperature":0})
    assert chat.status_code==200
    usage=client.get("/v1/usage",headers=headers)
    assert usage.status_code==200 and usage.json()["tracked"] is True
    revoked=client.delete("/v1/admin/api-keys/"+data["id"],headers={"X-Admin-Key":"admin-secret"})
    assert revoked.status_code==200
    assert client.get("/v1/models",headers=headers).status_code==401
