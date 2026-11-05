"""Local terminal results tell the agent when the host is saturated or the Hermes disk is nearly
full, at most once per interval. Regression context: 2026-11-05, agent build/test lanes drove a
14-core host to load ~400 with its disk at 98%; the agent kept launching work because nothing in
its tool results said the host was drowning."""

import json
from types import SimpleNamespace

import pytest

from tools import host_pressure, terminal_tool


@pytest.fixture
def overloaded(monkeypatch):
    monkeypatch.setattr(host_pressure, "_last_note_at", None)
    monkeypatch.setattr(host_pressure.os, "getloadavg", lambda: (400.0, 380.0, 300.0))
    monkeypatch.setattr(host_pressure.os, "cpu_count", lambda: 14)


def _terminal_result(monkeypatch, tmp_path, env_type="local"):
    class Env:
        env = {}
        cwd = str(tmp_path)

        def execute(self, command, **kwargs):
            return {"output": "ok", "returncode": 0}

    config = {"env_type": env_type, "cwd": str(tmp_path), "timeout": 5, "host_cwd": None,
              "modal_mode": "auto", "docker_image": "", "modal_image": "", "daytona_image": ""}
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": Env()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {"default": 0})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_resolve_container_task_id", lambda value: value or "default")
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda command, env_type, **kw: {"approved": True})
    return json.loads(terminal_tool.terminal_tool(command="echo ok"))


def test_saturated_host_note_reaches_the_agent_once_per_interval(overloaded, monkeypatch, tmp_path):
    first = _terminal_result(monkeypatch, tmp_path)
    assert first["output"] == "ok"
    assert "load average 400 on 14 cores" in first["host_pressure"]

    assert "host_pressure" not in _terminal_result(monkeypatch, tmp_path)


def test_note_repeats_after_the_interval_while_pressure_lasts(overloaded, tmp_path):
    assert host_pressure.host_pressure_note(now=1000.0, home=tmp_path)
    assert host_pressure.host_pressure_note(now=1000.0 + host_pressure._NOTE_INTERVAL_S - 1, home=tmp_path) is None
    assert host_pressure.host_pressure_note(now=1000.0 + host_pressure._NOTE_INTERVAL_S, home=tmp_path)


def test_nearly_full_disk_is_named(monkeypatch, tmp_path):
    from gateway import disk_status
    monkeypatch.setattr(host_pressure, "_last_note_at", None)
    monkeypatch.setattr(host_pressure.os, "getloadavg", lambda: (1.0, 1.0, 1.0))
    usage = SimpleNamespace(total=926 * 1024**3, used=int(925.8 * 1024**3), free=200 * 1024**2)
    monkeypatch.setattr(disk_status.shutil, "disk_usage", lambda path: usage)

    note = host_pressure.host_pressure_note(home=tmp_path)

    assert note and "Hermes data disk" in note and "200 MB free" in note


def test_healthy_host_and_remote_backends_get_no_note(monkeypatch, tmp_path):
    monkeypatch.setattr(host_pressure, "_last_note_at", None)
    monkeypatch.setattr(host_pressure.os, "getloadavg", lambda: (2.0, 2.0, 2.0))
    monkeypatch.setattr(host_pressure.os, "cpu_count", lambda: 14)
    assert "host_pressure" not in _terminal_result(monkeypatch, tmp_path)

    monkeypatch.setattr(host_pressure.os, "getloadavg", lambda: (400.0, 400.0, 400.0))
    assert "host_pressure" not in _terminal_result(monkeypatch, tmp_path, env_type="docker")
