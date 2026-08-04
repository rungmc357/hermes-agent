"""Durable transport provenance stays separate from routed session profile."""

import json
from typing import Any, cast

from gateway.config import Platform
from gateway.session import SessionSource, SessionStore


class _Recorder:
    def __init__(self):
        self.kwargs = None

    def record_gateway_session_peer(self, _session_id, **kwargs):
        self.kwargs = kwargs


def test_peer_record_snapshots_trusted_transport_owner_not_runtime_profile():
    store = object.__new__(SessionStore)
    recorder = _Recorder()
    store._db = cast(Any, recorder)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="42",
        profile="work",  # profile_routes runtime destination
    )
    source._transport_platform = "telegram"
    source._transport_profile = "default"  # bot credential that received it

    store._record_gateway_session_peer("session-1", "agent:work:telegram:dm:42", source)

    assert recorder.kwargs is not None
    origin = json.loads(recorder.kwargs["origin_json"])
    assert origin["profile"] == "work"
    assert origin["transport"] == {
        "platform": "telegram",
        "profile": "default",
    }
    assert "transport" not in source.to_dict()


def test_peer_record_omits_untrusted_missing_transport_owner():
    store = object.__new__(SessionStore)
    recorder = _Recorder()
    store._db = cast(Any, recorder)
    source = SessionSource(platform=Platform.DISCORD, chat_id="123", profile="work")

    store._record_gateway_session_peer("session-2", "agent:work:discord:dm:123", source)

    assert recorder.kwargs is not None
    origin = json.loads(recorder.kwargs["origin_json"])
    assert "transport" not in origin
