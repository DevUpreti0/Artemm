import io, sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).parents[1] / "sdks" / "python"))
from artemide_client import ArtemideAPIError, ArtemideClient

class FakeResponse(io.BytesIO):
    pass

def test_sdk_models_and_auth_header():
    response = FakeResponse(b'{"object":"list","data":[]}')
    with patch("artemide_client.urlopen", return_value=response) as mocked:
        result = ArtemideClient("http://localhost:8000", "test-key").models()
    assert result["object"] == "list"
    assert mocked.call_args.args[0].get_header("Authorization") == "Bearer test-key"
    assert response.closed

def test_sdk_stream_chat_parses_sse():
    response = FakeResponse(
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        b'data: [DONE]\n\n')
    with patch("artemide_client.urlopen", return_value=response):
        parts = list(ArtemideClient("http://localhost:8000").stream_chat([{"role":"user","content":"hi"}]))
    assert parts == ["Hello", " world"]
    assert response.closed

def test_sdk_translates_http_errors():
    from urllib.error import HTTPError
    err = HTTPError("http://localhost", 401, "Unauthorized", {}, io.BytesIO(b'{"detail":"bad key"}'))
    with patch("artemide_client.urlopen", side_effect=err):
        try: ArtemideClient("http://localhost:8000").models()
        except ArtemideAPIError as exc:
            assert exc.status_code == 401 and "bad key" in str(exc)
        else: raise AssertionError("expected API error")
