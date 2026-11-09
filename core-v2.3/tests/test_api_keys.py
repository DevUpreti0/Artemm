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
