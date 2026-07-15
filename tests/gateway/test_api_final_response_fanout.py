"""Integration coverage for canonical API final-response fanout."""

import asyncio
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import gateway.platforms.api_server as api_server_module
from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _IdempotencyCache,
    _make_request_fingerprint,
    _scoped_idempotency_key,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_state import SessionDB


@pytest.fixture
def fanout_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        """gateway:
  session_key_aliases:
    mobile-telegram:
      platform: telegram
      chat_id: canonical-chat
      chat_type: dm
      thread_id: canonical-thread
      user_id: canonical-user
    other-discord:
      platform: discord
      chat_id: other-chat
      chat_type: dm
      user_id: other-user
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(api_server_module, "_idem_cache", _IdempotencyCache())
    db = SessionDB(tmp_path / "state.db")
    db.create_session("session-one", "api_server")
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "sk-test"})
    )
    adapter._session_db = db
    try:
        yield adapter
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


def _source(profile: str | None = None) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="canonical-chat",
        chat_type="dm",
        thread_id="canonical-thread",
        user_id="canonical-user",
        profile=profile,
    )


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post(
        "/api/sessions/{session_id}/chat", adapter._handle_session_chat
    )
    app.router.add_post(
        "/api/sessions/{session_id}/chat/stream",
        adapter._handle_session_chat_stream,
    )
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    return app


def _headers(**extra) -> dict[str, str]:
    return {
        "Authorization": "Bearer sk-test",
        "X-Hermes-Session-Key": "mobile-telegram",
        **extra,
    }


async def _drain_background(adapter: APIServerAdapter) -> None:
    for _ in range(10):
        await asyncio.sleep(0)
        tasks = [
            task
            for task in list(adapter._background_tasks)
            if not task.done()
        ]
        if not tasks:
            return
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_all_api_turn_surfaces_fanout_one_terminal_response(fanout_adapter):
    calls = []

    async def fanout(**kwargs):
        calls.append(kwargs)

    fanout_adapter.set_final_response_fanout_handler(fanout)
    fanout_adapter._run_agent = AsyncMock(
        return_value=(
            {
                "final_response": "final answer",
                "completed": True,
                "session_id": "session-one",
            },
            {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        )
    )

    class FakeAgent:
        session_prompt_tokens = 1
        session_completion_tokens = 2
        session_total_tokens = 3
        session_id = "run-session"

        def run_conversation(self, **_kwargs):
            return {"final_response": "final answer", "completed": True}

    fanout_adapter._create_agent = MagicMock(return_value=FakeAgent())
    cases = [
        ("/api/sessions/session-one/chat", {"message": "hello"}),
        ("/api/sessions/session-one/chat/stream", {"message": "hello"}),
        (
            "/v1/chat/completions",
            {"model": "test", "messages": [{"role": "user", "content": "hello"}]},
        ),
        (
            "/v1/chat/completions",
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        ),
        ("/v1/responses", {"model": "test", "input": "hello"}),
        (
            "/v1/responses",
            {"model": "test", "input": "hello", "stream": True},
        ),
        ("/v1/runs", {"input": "hello"}),
    ]

    async with TestClient(TestServer(_app(fanout_adapter))) as client:
        for path, payload in cases:
            response = await client.post(path, json=payload, headers=_headers())
            assert response.status in {200, 202}
            await response.read()
            await _drain_background(fanout_adapter)

    assert Counter(call["surface"] for call in calls) == {
        "session_chat": 1,
        "session_chat_stream": 1,
        "chat_completions": 2,
        "responses": 2,
        "runs": 1,
    }
    assert all(call["session_source"] == _source() for call in calls)
    assert all(call["content"] == "final answer" for call in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/sessions/session-one/chat", {"message": "hello"}),
        ("/api/sessions/session-one/chat/stream", {"message": "hello"}),
        (
            "/v1/chat/completions",
            {"model": "test", "messages": [{"role": "user", "content": "hello"}]},
        ),
    ],
)
async def test_native_fanout_preserves_raw_media_directive(
    fanout_adapter, tmp_path, path, payload
):
    media_path = tmp_path / "reply.png"
    media_path.write_bytes(b"not-a-real-png-but-valid-test-bytes")
    raw_response = f"MEDIA:{media_path}"
    calls = []
    fanout_adapter.set_final_response_fanout_handler(
        lambda **kwargs: calls.append(kwargs)
    )
    fanout_adapter._run_agent = AsyncMock(
        return_value=(
            {"final_response": raw_response, "completed": True},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
    )

    async with TestClient(TestServer(_app(fanout_adapter))) as client:
        response = await client.post(path, json=payload, headers=_headers())
        assert response.status == 200
        response_body = await response.read()
        await _drain_background(fanout_adapter)

    assert b"data:image/png;base64" in response_body
    assert [call["content"] for call in calls] == [raw_response]


@pytest.mark.asyncio
async def test_fanout_survives_real_gateway_teardown_and_drains_before_disconnect(
    fanout_adapter,
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked(**_kwargs):
        started.set()
        await release.wait()

    fanout_adapter.set_final_response_fanout_handler(blocked)
    await fanout_adapter._fanout_completed_api_turn(
        session_source=_source(),
        response_text="deliver before shutdown",
        surface="responses",
    )
    await started.wait()

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.API_SERVER: fanout_adapter}
    runner._adapter_disconnect_timeout_secs = lambda: 1.0
    assert fanout_adapter.active_agent_work_count() == 1
    assert runner._active_api_run_count() == 1

    teardown_task = asyncio.create_task(
        runner._bounded_adapter_teardown(
            fanout_adapter,
            Platform.API_SERVER,
        )
    )
    await asyncio.sleep(0)
    assert not teardown_task.done()
    assert not next(iter(fanout_adapter._fanout_tasks)).cancelled()

    release.set()
    await teardown_task
    assert fanout_adapter.active_agent_work_count() == 0


@pytest.mark.asyncio
async def test_runner_timeout_cancels_and_reaps_fanout_without_loop_errors(
    fanout_adapter,
):
    started = asyncio.Event()
    cancelled = asyncio.Event()
    never_release = asyncio.Event()

    async def blocked_fanout(**_kwargs):
        started.set()
        try:
            await never_release.wait()
        finally:
            cancelled.set()

    fanout_adapter.set_final_response_fanout_handler(blocked_fanout)
    await fanout_adapter._fanout_completed_api_turn(
        session_source=_source(),
        response_text="terminal",
        surface="session_chat",
    )
    await started.wait()

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._adapter_disconnect_timeout_secs = lambda: 0.01
    loop = asyncio.get_running_loop()
    loop_errors = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    try:
        await runner._bounded_adapter_teardown(
            fanout_adapter,
            Platform.API_SERVER,
        )
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert cancelled.is_set()
    assert not [task for task in fanout_adapter._fanout_tasks if not task.done()]
    assert not [task for task in fanout_adapter._background_tasks if not task.done()]
    assert loop_errors == []


@pytest.mark.asyncio
async def test_legacy_adapter_without_optional_fanout_state_tears_down_safely(
    fanout_adapter,
):
    del fanout_adapter._fanout_tasks
    del fanout_adapter._preserve_fanout_during_cancel

    await fanout_adapter.cancel_background_tasks()

    assert fanout_adapter._fanout_tasks == set()
    assert fanout_adapter._preserve_fanout_during_cancel is False


@pytest.mark.asyncio
async def test_fanout_self_initializes_optional_tracking_state(fanout_adapter):
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked(**_kwargs):
        started.set()
        await release.wait()

    del fanout_adapter._fanout_tasks
    del fanout_adapter._preserve_fanout_during_cancel
    fanout_adapter.set_final_response_fanout_handler(blocked)
    await fanout_adapter._fanout_completed_api_turn(
        session_source=_source(),
        response_text="terminal",
        surface="responses",
    )
    await started.wait()

    assert len(fanout_adapter._fanout_tasks) == 1
    assert fanout_adapter.active_agent_work_count() == 1
    release.set()
    await fanout_adapter._drain_final_response_fanout()
    assert fanout_adapter.active_agent_work_count() == 0


@pytest.mark.asyncio
async def test_runner_timeout_has_one_cancellation_owner_and_keeps_budget(
    fanout_adapter,
):
    started = asyncio.Event()
    release_after_cancel = asyncio.Event()
    finished = asyncio.Event()
    cancellations = 0

    async def cancellation_resistant(**_kwargs):
        nonlocal cancellations
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellations += 1
            await release_after_cancel.wait()
        finally:
            finished.set()

    fanout_adapter.set_final_response_fanout_handler(cancellation_resistant)
    await fanout_adapter._fanout_completed_api_turn(
        session_source=_source(),
        response_text="terminal",
        surface="session_chat",
    )
    await started.wait()

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._adapter_disconnect_timeout_secs = lambda: 0.01
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    await runner._bounded_adapter_teardown(fanout_adapter, Platform.API_SERVER)
    elapsed = loop.time() - started_at

    assert elapsed < 0.1
    assert cancellations == 1
    assert not finished.is_set()

    release_after_cancel.set()
    await asyncio.wait_for(finished.wait(), timeout=1)
    await asyncio.sleep(0)
    assert cancellations == 1
    assert not [task for task in fanout_adapter._fanout_tasks if not task.done()]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "payload", "expected_surface"),
    [
        (
            "/api/sessions/session-one/chat",
            {"message": "hello"},
            "session_chat",
        ),
        (
            "/api/sessions/session-one/chat/stream",
            {"message": "hello"},
            "session_chat_stream",
        ),
        ("/v1/runs", {"input": "hello"}, "runs"),
    ],
)
@pytest.mark.parametrize("concurrent", [False, True])
async def test_remaining_surfaces_claim_terminal_fanout_once(
    fanout_adapter,
    path,
    payload,
    expected_surface,
    concurrent,
):
    calls = []
    fanout_adapter.set_final_response_fanout_handler(
        lambda **kwargs: calls.append(kwargs)
    )
    fanout_adapter._run_agent = AsyncMock(
        return_value=(
            {"final_response": "once", "completed": True},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
    )

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "run-session"

        def run_conversation(self, **_kwargs):
            return {"final_response": "once", "completed": True}

    fanout_adapter._create_agent = MagicMock(side_effect=lambda **_kwargs: FakeAgent())
    headers = _headers(**{"Idempotency-Key": f"{expected_surface}-once"})

    async with TestClient(TestServer(_app(fanout_adapter))) as client:
        if concurrent:
            responses = await asyncio.gather(
                client.post(path, json=payload, headers=headers),
                client.post(path, json=payload, headers=headers),
            )
        else:
            responses = []
            for _ in range(2):
                responses.append(await client.post(path, json=payload, headers=headers))
        for response in responses:
            assert response.status in {200, 202}
            await response.read()
        await _drain_background(fanout_adapter)

    assert [call["surface"] for call in calls] == [expected_surface]


@pytest.mark.asyncio
async def test_session_sync_and_stream_share_terminal_claim_namespace(fanout_adapter):
    calls = []
    fanout_adapter.set_final_response_fanout_handler(
        lambda **kwargs: calls.append(kwargs)
    )
    fanout_adapter._run_agent = AsyncMock(
        return_value=(
            {"final_response": "once", "completed": True},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
    )
    headers = _headers(**{"Idempotency-Key": "session-cross-mode"})

    async with TestClient(TestServer(_app(fanout_adapter))) as client:
        for path in (
            "/api/sessions/session-one/chat",
            "/api/sessions/session-one/chat/stream",
        ):
            response = await client.post(
                path,
                json={"message": "hello"},
                headers=headers,
            )
            assert response.status == 200
            await response.read()
        await _drain_background(fanout_adapter)

    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/sessions/session-one/chat", {"message": "hello"}),
        ("/api/sessions/session-one/chat/stream", {"message": "hello"}),
        ("/v1/runs", {"input": "hello"}),
    ],
)
async def test_new_terminal_claims_are_isolated_by_canonical_identity(
    fanout_adapter,
    path,
    payload,
):
    calls = []
    fanout_adapter.set_final_response_fanout_handler(
        lambda **kwargs: calls.append(kwargs)
    )
    fanout_adapter._run_agent = AsyncMock(
        return_value=(
            {"final_response": "identity result", "completed": True},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
    )

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "run-session"

        def run_conversation(self, **_kwargs):
            return {"final_response": "identity result", "completed": True}

    fanout_adapter._create_agent = MagicMock(side_effect=lambda **_kwargs: FakeAgent())

    async with TestClient(TestServer(_app(fanout_adapter))) as client:
        for alias in ("mobile-telegram", "other-discord"):
            response = await client.post(
                path,
                json=payload,
                headers=_headers(
                    **{
                        "Idempotency-Key": "shared-new-surface-key",
                        "X-Hermes-Session-Key": alias,
                    }
                ),
            )
            assert response.status in {200, 202}
            await response.read()
        await _drain_background(fanout_adapter)

    assert [call["session_source"].platform for call in calls] == [
        Platform.TELEGRAM,
        Platform.DISCORD,
    ]


def test_request_fingerprint_canonicalizes_nested_json_object_order():
    keys = ["input"]
    role_first = {"input": [{"role": "user", "content": "hello"}]}
    content_first = {"input": [{"content": "hello", "role": "user"}]}
    changed = {"input": [{"content": "different", "role": "user"}]}

    assert _make_request_fingerprint(role_first, keys) == _make_request_fingerprint(
        content_first,
        keys,
    )
    assert _make_request_fingerprint(role_first, keys) != _make_request_fingerprint(
        changed,
        keys,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/sessions/session-one/chat", {"message": "hello"}),
        ("/api/sessions/session-one/chat/stream", {"message": "hello"}),
        ("/v1/runs", {"input": "hello"}),
    ],
)
async def test_empty_terminal_result_does_not_poison_retry_claim(
    fanout_adapter,
    path,
    payload,
):
    calls = []
    fanout_adapter.set_final_response_fanout_handler(
        lambda **kwargs: calls.append(kwargs)
    )
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    fanout_adapter._run_agent = AsyncMock(
        side_effect=[
            ({"final_response": "", "completed": True}, usage),
            ({"final_response": "retry result", "completed": True}, usage),
        ]
    )
    run_results = iter(
        [
            {"final_response": "", "completed": True},
            {"final_response": "retry result", "completed": True},
        ]
    )

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "run-session"

        def __init__(self, result):
            self.result = result

        def run_conversation(self, **_kwargs):
            return self.result

    fanout_adapter._create_agent = MagicMock(
        side_effect=lambda **_kwargs: FakeAgent(next(run_results))
    )
    headers = _headers(**{"Idempotency-Key": "empty-then-real"})

    async with TestClient(TestServer(_app(fanout_adapter))) as client:
        for _ in range(2):
            response = await client.post(path, json=payload, headers=headers)
            assert response.status in {200, 202}
            await response.read()
            await _drain_background(fanout_adapter)

    assert [call["content"] for call in calls] == ["retry result"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "payload", "key"),
    [
        (
            "/v1/chat/completions",
            {"model": "test", "messages": [{"role": "user", "content": "hello"}]},
            "chat-replay",
        ),
        ("/v1/responses", {"model": "test", "input": "hello"}, "responses-replay"),
    ],
)
async def test_idempotency_replay_does_not_duplicate_fanout(
    fanout_adapter, path, payload, key
):
    calls = []
    fanout_adapter.set_final_response_fanout_handler(
        lambda **kwargs: calls.append(kwargs)
    )
    fanout_adapter._run_agent = AsyncMock(
        return_value=(
            {"final_response": "once", "completed": True},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
    )
    headers = _headers(**{"Idempotency-Key": key})

    async with TestClient(TestServer(_app(fanout_adapter))) as client:
        for _ in range(2):
            response = await client.post(path, json=payload, headers=headers)
            assert response.status == 200
            await response.read()
            await _drain_background(fanout_adapter)

    assert fanout_adapter._run_agent.await_count == 1
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "payload", "key"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            "chat-stream-replay",
        ),
        (
            "/v1/responses",
            {"model": "test", "input": "hello", "stream": True},
            "responses-stream-replay",
        ),
    ],
)
async def test_streaming_idempotency_replay_fans_out_once(
    fanout_adapter, path, payload, key
):
    calls = []
    fanout_adapter.set_final_response_fanout_handler(
        lambda **kwargs: calls.append(kwargs)
    )
    fanout_adapter._run_agent = AsyncMock(
        return_value=(
            {"final_response": "once", "completed": True},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
    )
    headers = _headers(**{"Idempotency-Key": key})

    async with TestClient(TestServer(_app(fanout_adapter))) as client:
        responses = await asyncio.gather(
            client.post(path, json=payload, headers=headers),
            client.post(path, json=payload, headers=headers),
        )
        for response in responses:
            assert response.status == 200
            await response.read()
        await _drain_background(fanout_adapter)

    assert fanout_adapter._run_agent.await_count == 2
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/chat/completions",
            {"model": "test", "messages": [{"role": "user", "content": "hello"}]},
        ),
        (
            "/v1/chat/completions",
            {
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        ),
        ("/v1/responses", {"model": "test", "input": "hello"}),
        (
            "/v1/responses",
            {"model": "test", "input": "hello", "stream": True},
        ),
    ],
)
async def test_idempotency_isolated_between_canonical_aliases(
    fanout_adapter, path, payload
):
    calls = []
    fanout_adapter.set_final_response_fanout_handler(
        lambda **kwargs: calls.append(kwargs)
    )
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    fanout_adapter._run_agent = AsyncMock(
        side_effect=[
            ({"final_response": "first identity", "completed": True}, usage),
            ({"final_response": "second identity", "completed": True}, usage),
        ]
    )
    common_headers = {"Idempotency-Key": "shared-cross-identity-key"}

    async with TestClient(TestServer(_app(fanout_adapter))) as client:
        for alias in ("mobile-telegram", "other-discord"):
            response = await client.post(
                path,
                json=payload,
                headers=_headers(
                    **common_headers,
                    **{"X-Hermes-Session-Key": alias},
                ),
            )
            assert response.status == 200
            await response.read()
            await _drain_background(fanout_adapter)

    assert fanout_adapter._run_agent.await_count == 2
    assert [call["session_source"].platform for call in calls] == [
        Platform.TELEGRAM,
        Platform.DISCORD,
    ]
    assert [call["content"] for call in calls] == [
        "first identity",
        "second identity",
    ]


def test_idempotency_scope_includes_profile_without_native_source():
    token = api_server_module._api_request_profile.set("default")
    try:
        default_key = _scoped_idempotency_key(
            surface="responses",
            key="same-key",
            gateway_session_key="legacy-key",
            session_source=None,
        )
        api_server_module._api_request_profile.set("work")
        work_key = _scoped_idempotency_key(
            surface="responses",
            key="same-key",
            gateway_session_key="legacy-key",
            session_source=None,
        )
    finally:
        api_server_module._api_request_profile.reset(token)

    assert default_key != work_key


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_response", ["", "   \n"])
async def test_nonstream_responses_empty_terminal_result_never_fans_out(
    fanout_adapter, empty_response
):
    calls = []
    fanout_adapter.set_final_response_fanout_handler(
        lambda **kwargs: calls.append(kwargs)
    )
    fanout_adapter._run_agent = AsyncMock(
        return_value=(
            {"final_response": empty_response, "completed": True},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
    )

    async with TestClient(TestServer(_app(fanout_adapter))) as client:
        response = await client.post(
            "/v1/responses",
            json={"model": "test", "input": "hello"},
            headers=_headers(),
        )
        assert response.status == 200
        await response.read()
        await _drain_background(fanout_adapter)

    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        {"final_response": "partial", "completed": False, "partial": True},
        {"final_response": "failed", "completed": False, "failed": True},
        {"final_response": "incomplete", "completed": False},
    ],
)
async def test_nonterminal_results_never_fanout(fanout_adapter, result):
    calls = []
    fanout_adapter.set_final_response_fanout_handler(
        lambda **kwargs: calls.append(kwargs)
    )
    fanout_adapter._run_agent = AsyncMock(
        return_value=(
            result,
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
    )

    async with TestClient(TestServer(_app(fanout_adapter))) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            headers=_headers(),
        )
        assert response.status == 200
        await response.read()
        await _drain_background(fanout_adapter)

    assert calls == []


@pytest.mark.asyncio
async def test_fanout_does_not_reset_pending_request_accounting(fanout_adapter):
    async def deliver(**_kwargs):
        return None

    fanout_adapter._pending_agent_requests = 3
    fanout_adapter.set_final_response_fanout_handler(deliver)
    await fanout_adapter._fanout_completed_api_turn(
        session_source=_source(),
        response_text="completed",
        surface="responses",
    )
    await asyncio.sleep(0)

    assert fanout_adapter._pending_agent_requests == 3


@pytest.mark.asyncio
async def test_fanout_is_nonblocking_and_preserves_exact_text(fanout_adapter):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def blocked(**kwargs):
        calls.append(kwargs)
        started.set()
        await release.wait()

    fanout_adapter.set_final_response_fanout_handler(blocked)
    await fanout_adapter._fanout_completed_api_turn(
        session_source=_source(),
        response_text="  exact text\n",
        surface="responses",
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    assert calls[0]["content"] == "  exact text\n"
    release.set()
    await _drain_background(fanout_adapter)


@pytest.mark.asyncio
async def test_runner_delivers_to_canonical_platform_chat_and_thread():
    sent = []

    class TargetAdapter:
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            sent.append((chat_id, content, reply_to, metadata))
            return SimpleNamespace(success=True)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.__dict__["adapters"] = {Platform.TELEGRAM: TargetAdapter()}
    await runner._deliver_api_final_response(
        session_source=_source(), content="done", surface="responses"
    )

    assert sent == [
        (
            "canonical-chat",
            "done",
            None,
            {"thread_id": "canonical-thread"},
        )
    ]


@pytest.mark.asyncio
async def test_runner_uses_profile_specific_adapter_and_fails_closed():
    default_sent = []
    secondary_sent = []

    class TargetAdapter:
        def __init__(self, sent):
            self.sent = sent

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.sent.append((chat_id, content))
            return SimpleNamespace(success=True)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.__dict__["adapters"] = {
        Platform.TELEGRAM: TargetAdapter(default_sent)
    }
    runner.__dict__["_profile_adapters"] = {
        "work": {Platform.TELEGRAM: TargetAdapter(secondary_sent)}
    }

    await runner._deliver_api_final_response(
        session_source=_source(), content="default", surface="responses"
    )
    await runner._deliver_api_final_response(
        session_source=_source("work"), content="secondary", surface="responses"
    )
    await runner._deliver_api_final_response(
        session_source=_source("missing"), content="drop", surface="responses"
    )

    assert default_sent == [("canonical-chat", "default")]
    assert secondary_sent == [("canonical-chat", "secondary")]


@pytest.mark.asyncio
async def test_fanout_failure_never_changes_api_result(fanout_adapter):
    async def broken(**_kwargs):
        raise RuntimeError("delivery unavailable")

    fanout_adapter.set_final_response_fanout_handler(broken)
    fanout_adapter._run_agent = AsyncMock(
        return_value=(
            {"final_response": "api still succeeds", "completed": True},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
    )

    async with TestClient(TestServer(_app(fanout_adapter))) as client:
        response = await client.post(
            "/v1/responses",
            json={"input": "hello"},
            headers=_headers(),
        )
        assert response.status == 200
        payload = await response.json()
        assert payload["status"] == "completed"
        await _drain_background(fanout_adapter)
