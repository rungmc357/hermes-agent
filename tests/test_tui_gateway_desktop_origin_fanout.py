"""Desktop continuation fanout uses persisted origin metadata, never RPC input."""

from contextlib import contextmanager

import pytest

from gateway import delivery_ledger as dl
from tui_gateway import server


class _Db:
    def __init__(self, row):
        self.row = row
        self.requested = None

    def get_session(self, session_id):
        self.requested = session_id
        return self.row


@contextmanager
def _db_context(db):
    yield db


@pytest.mark.parametrize("platform", ["telegram", "discord"])
def test_desktop_reply_queues_persisted_messaging_origin(monkeypatch, platform):
    db = _Db({
        "source": platform,
        "chat_id": "native-chat",
        "thread_id": "native-thread",
    })
    captured = {}
    monkeypatch.setattr(server, "_session_db", lambda _session: _db_context(db))
    monkeypatch.setattr(dl, "ledger_enabled", lambda: True)
    monkeypatch.setattr(dl, "compute_obligation_id", lambda *args: "desktop-obligation")
    monkeypatch.setattr(
        dl,
        "record_external_obligation",
        lambda **kwargs: captured.update(kwargs),
    )
    session = {
        "source": "desktop",
        "session_key": "stored-session",
        "profile_home": "/tmp/.hermes/profiles/work",
    }

    assert server._queue_desktop_origin_fanout(session, "final answer") is True
    assert db.requested == "stored-session"
    assert captured == {
        "obligation_id": "desktop-obligation",
        "session_key": "stored-session",
        "platform": platform,
        "chat_id": "native-chat",
        "thread_id": "native-thread",
        "content": "final answer",
        "profile": "work",
        "db_path": server._hermes_home / "state.db",
    }


@pytest.mark.parametrize(
    ("runtime_source", "stored_source", "chat_id"),
    [
        ("tui", "telegram", "42"),
        ("desktop", "desktop", "42"),
        ("desktop", "telegram", ""),
        ("desktop", "slack", "42"),
    ],
)
def test_non_desktop_or_non_target_origin_never_queues(
    monkeypatch, runtime_source, stored_source, chat_id
):
    db = _Db({"source": stored_source, "chat_id": chat_id, "thread_id": None})
    calls = []
    monkeypatch.setattr(server, "_session_db", lambda _session: _db_context(db))
    monkeypatch.setattr(dl, "ledger_enabled", lambda: True)
    monkeypatch.setattr(dl, "record_external_obligation", lambda **kwargs: calls.append(kwargs))

    queued = server._queue_desktop_origin_fanout(
        {"source": runtime_source, "session_key": "stored-session"},
        "answer",
    )

    assert queued is False
    assert calls == []


def test_queue_failure_is_best_effort(monkeypatch):
    db = _Db({"source": "telegram", "chat_id": "42", "thread_id": None})
    monkeypatch.setattr(server, "_session_db", lambda _session: _db_context(db))
    monkeypatch.setattr(dl, "ledger_enabled", lambda: True)

    def _fail(**_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(dl, "record_external_obligation", _fail)

    assert server._queue_desktop_origin_fanout(
        {"source": "desktop", "session_key": "stored-session"},
        "answer",
    ) is False


def test_default_home_uses_active_profile_name(monkeypatch):
    db = _Db({"source": "discord", "chat_id": "42", "thread_id": None})
    captured = {}
    monkeypatch.setattr(server, "_session_db", lambda _session: _db_context(db))
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    monkeypatch.setattr(dl, "ledger_enabled", lambda: True)
    monkeypatch.setattr(dl, "record_external_obligation", lambda **kwargs: captured.update(kwargs))

    assert server._queue_desktop_origin_fanout(
        {
            "source": "desktop",
            "session_key": "stored-session",
            "profile_home": "/tmp/.hermes",
        },
        "answer",
    ) is True
    assert captured["profile"] == "default"
