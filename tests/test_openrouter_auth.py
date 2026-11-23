"""Regression tests for the OpenRouter authentication failure.

The original failure was caused by ``docker-compose.yml`` declaring
``OPENROUTER_API_KEY=${sk-or-v1-...}``. Docker Compose reads that as the
shell-style ``${sk-<default>}`` expression, so the container received
``or-v1-...`` (the ``sk-`` prefix stripped) and OpenRouter replied
``401 Missing Authentication header``.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.agents.nodes as nodes

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "sk-or-v1-" + "a" * 64


def test_docker_compose_forwards_the_env_var_without_mangling():
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert compose.count("OPENROUTER_API_KEY=${OPENROUTER_API_KEY}") == 2
    assert "OPENROUTER_API_KEY=${sk-or-v1-" not in compose
    # No real credential may be hard-coded in the compose file.
    assert re.search(r"sk-or-v1-[0-9a-f]{32,}", compose) is None


def test_get_api_key_rejects_a_key_without_sk_prefix(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-v1-" + "a" * 64)

    with pytest.raises(RuntimeError) as excinfo:
        nodes._get_api_key()

    message = str(excinfo.value)
    assert "sk-or-v1-" in message
    assert "docker-compose.yml" in message


def test_get_api_key_never_echoes_the_configured_value(monkeypatch):
    secret_value = "or-v1-do-not-log-this-value"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret_value)

    with pytest.raises(RuntimeError) as excinfo:
        nodes._get_api_key()

    assert secret_value not in str(excinfo.value)


def test_get_api_key_strips_surrounding_quotes(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", f'"{FAKE_KEY}"')

    assert nodes._get_api_key() == FAKE_KEY


def test_get_api_key_reads_dotenv_from_project_root(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        f"OPENROUTER_API_KEY={FAKE_KEY}\n", encoding="utf-8"
    )
    monkeypatch.setattr(nodes, "PROJECT_ROOT", tmp_path)

    assert nodes._get_api_key() == FAKE_KEY


def test_missing_key_raises_an_actionable_error(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(nodes, "PROJECT_ROOT", tmp_path / "does-not-exist")

    with pytest.raises(RuntimeError, match="Missing OPENROUTER_API_KEY"):
        nodes._get_api_key()


def test_get_client_uses_the_validated_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)

    client, api_key = nodes._get_client()

    assert api_key == FAKE_KEY
    assert client.api_key == FAKE_KEY


class FakeAuthError(Exception):
    status_code = 401


class FakeModelGoneError(Exception):
    status_code = 404


class FakeRateLimitError(Exception):
    status_code = 429


def _completion_response(model: str, content: str = '{"findings": []}'):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


def _patch_llm(monkeypatch, completions, models):
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(nodes, "_get_client", lambda: (fake_client, FAKE_KEY))
    monkeypatch.setattr(nodes, "_get_live_free_models", lambda key: list(models))
    monkeypatch.setattr(nodes.time, "sleep", lambda delay: None)


def test_authentication_error_fails_fast_instead_of_trying_every_model(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs["model"])
            raise FakeAuthError(
                "Error code: 401 - {'error': {'message': 'Missing Authentication "
                "header', 'code': 401}}"
            )

    _patch_llm(monkeypatch, FakeCompletions(), ["a/free", "b/free", "c/free"])

    with pytest.raises(RuntimeError) as excinfo:
        nodes._call_free_llm_with_retry("prompt")

    assert calls == ["a/free"]
    message = str(excinfo.value)
    assert "authentication failed" in message.lower()
    assert "OPENROUTER_API_KEY" in message
    assert "Missing Authentication header" in message


def test_model_not_found_moves_to_the_next_model_without_retrying(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs["model"])
            if len(calls) == 1:
                raise FakeModelGoneError("404 model unavailable")
            return _completion_response("b/free")

    _patch_llm(monkeypatch, FakeCompletions(), ["a/free", "b/free"])

    parsed, used_model = nodes._call_free_llm_with_retry("prompt")

    assert calls == ["a/free", "b/free"]
    assert parsed == {"findings": []}
    assert used_model == "b/free"


def test_rate_limited_model_is_retried_before_succeeding(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs["model"])
            if len(calls) == 1:
                raise FakeRateLimitError("429 rate limited")
            return _completion_response("a/free")

    _patch_llm(monkeypatch, FakeCompletions(), ["a/free"])

    parsed, used_model = nodes._call_free_llm_with_retry("prompt")

    assert calls == ["a/free", "a/free"]
    assert used_model == "a/free"
    assert parsed == {"findings": []}


def test_daily_free_model_quota_exhaustion_fails_without_retrying(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs["model"])
            raise FakeRateLimitError(
                "Rate limit exceeded: free-models-per-day. Add credits to unlock "
                "more free model requests."
            )

    _patch_llm(monkeypatch, FakeCompletions(), ["a/free", "b/free"])

    with pytest.raises(
        RuntimeError, match="daily free-model request quota is exhausted"
    ) as excinfo:
        nodes._call_free_llm_with_retry("prompt")

    assert calls == ["a/free"]
    assert "Wait for the quota to reset" in str(excinfo.value)
    assert "add credits" in str(excinfo.value)


def test_fallback_models_are_valid_openrouter_slugs():
    assert nodes.FALLBACK_MODELS
    for model in nodes.FALLBACK_MODELS:
        assert model.endswith(":free"), model
        assert "/" in model, model
