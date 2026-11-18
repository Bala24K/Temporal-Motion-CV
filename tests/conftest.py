import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.config import settings
from backend.state import Metrics, state
from backend.security import login_limiter


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(settings, "admin_emails", frozenset({"admin@example.com"}))
    login_limiter.reset()
    state.metrics = Metrics()          # fresh counters per test: metrics are process-global
    main.init_app_state("sqlite:///:memory:")
    with TestClient(main.app) as c:
        yield c


def register(client, email="a@example.com", pw="password123"):
    r = client.post("/api/auth/register", json={"email": email, "password": pw, "name": "Test"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


@pytest.fixture()
def auth(client):
    return register(client)


@pytest.fixture()
def admin(client):
    return register(client, "admin@example.com")
