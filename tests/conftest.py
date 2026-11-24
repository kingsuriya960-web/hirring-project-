import pytest

import app.agents.nodes as nodes


@pytest.fixture(autouse=True)
def disable_live_groq(monkeypatch):
    """Keep the suite deterministic even when a developer has a .env Groq key."""
    monkeypatch.setattr(nodes, "get_groq_api_key", lambda required=False: None)
