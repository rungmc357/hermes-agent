"""Desktop continuation fanout uses persisted origin metadata, never RPC input."""

from contextlib import contextmanager
import json

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


def _stored_origin(platform, *, transport_profile="default", **kwargs):
    row = {
        "source": platform,
        "chat_id": "native-chat",
        "thread_id": "native-thread",
        "origin_json": json.dumps(
            {
                "platform": platform,
                "transport": {
                    "platform": platform,
                    "profile": transport_profile,
                },
            }
        ),
    }
    row.update(kwargs)
    return row


@pytest.mark.parametrize("platform", ["telegram", "discord"])
def test_desktop_reply_queues_persisted_messaging_origin(monkeypatch, platform):
    db = _Db(_stored_origin(platform, transport_profile="default"))
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
        "profile": "default",
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
    db = _Db(
        _stored_origin(
            stored_source,
            chat_id=chat_id,
            thread_id=None,
        )
    )
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
    db = _Db(
        _stored_origin("telegram", chat_id="42", thread_id=None)
    )
    monkeypatch.setattr(server, "_session_db", lambda _session: _db_context(db))
    monkeypatch.setattr(dl, "ledger_enabled", lambda: True)

    def _fail(**_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(dl, "record_external_obligation", _fail)

    assert server._queue_desktop_origin_fanout(
        {"source": "desktop", "session_key": "stored-session"},
        "answer",
    ) is False


def test_persisted_transport_profile_is_used_not_runtime_profile(monkeypatch):
    db = _Db(
        _stored_origin(
            "discord",
            transport_profile="default",
            chat_id="42",
            thread_id=None,
        )
    )
    captured = {}
    monkeypatch.setattr(server, "_session_db", lambda _session: _db_context(db))

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


def test_historical_row_without_transport_provenance_fails_closed(monkeypatch):
    db = _Db({"source": "telegram", "chat_id": "42", "thread_id": None})
    calls = []
    monkeypatch.setattr(server, "_session_db", lambda _session: _db_context(db))
    monkeypatch.setattr(dl, "ledger_enabled", lambda: True)
    monkeypatch.setattr(dl, "record_external_obligation", lambda **kwargs: calls.append(kwargs))

    assert server._queue_desktop_origin_fanout(
        {"source": "desktop", "session_key": "stored-session"},
        "answer",
    ) is False
    assert calls == []
