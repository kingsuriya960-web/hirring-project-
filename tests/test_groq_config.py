from pathlib import Path
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app.agents.nodes as nodes


def test_model_not_found_skips_retries_and_tries_the_next_model(monkeypatch):
    calls = []

    class ModelNotFoundError(Exception):
        status_code = 404

    class FakeCompletions:
        def create(self, **kwargs):
            model = kwargs["model"]
            calls.append(model)
            if len(calls) == 1:
                raise ModelNotFoundError("model not found")
            return SimpleNamespace(
                model=model,
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"findings": []}')
                    )
                ],
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(nodes, "_get_client", lambda: client)
    monkeypatch.setattr(nodes.time, "sleep", lambda _: None)

    parsed, used_model = nodes._call_groq_with_retry("prompt")

    assert calls == nodes.GROQ_MODELS[:2]
    assert parsed == {"findings": []}
    assert used_model == f"groq/{nodes.GROQ_MODELS[1]}"
