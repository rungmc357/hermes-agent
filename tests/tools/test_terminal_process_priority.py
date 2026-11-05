"""terminal.process_priority runs host agent work (terminal commands, the execute_code kernel)
below Hermes' own priority, so a build/test swarm cannot starve the gateway that answers every
profile. Regression context: six agent build/test lanes at normal priority drove a 14-core macOS
host to load ~400; the gateway's event loop stalled for over a minute and state.db writes timed out."""

import json
import os
import shutil
import time
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from tests.waiting import wait_until
from tools.environments.local import LocalEnvironment
from tools.environments.local_priority import PROCESS_PRIORITY_LEVELS, priority_prefix_for
from tools.process_registry import ProcessRegistry

_LINUX_NICE = "cat /proc/self/stat | awk '{print $19}'"  # the shell's own niceness, no ps needed
_LINUX_IO_CLASS = "ionice -p $$"
_MACOS_PRI = "ps -o pri= -p $$"  # PRIO_DARWIN_BG reports scheduling priority 4 (normal: 20 at any nice)
_DARWIN_BG_PRI = 4


def _write_priority(home, value):
    cfg = home / "config.yaml"
    cfg.write_text(f"terminal:\n  process_priority: {value}\n", encoding="utf-8")
    bump = time.time() + 5 * (1 + PROCESS_PRIORITY_LEVELS.index(value))
    os.utime(cfg, (bump, bump))  # distinct mtime so the cached config read sees the rewrite


def _foreground(command, tmp_path, monkeypatch):
    env = LocalEnvironment(cwd=str(tmp_path), timeout=20)
    try:
        return env.execute(command)["output"].strip().splitlines()[-1].strip()
    finally:
        env.cleanup()


def _background(command, tmp_path, monkeypatch):
    monkeypatch.setattr(ProcessRegistry, "_write_checkpoint", lambda self: None)
    session = ProcessRegistry().spawn_local(command, cwd=str(tmp_path))
    out = wait_until(lambda: session.output_buffer.strip(), timeout=20, what="background priority output")
    return out.splitlines()[-1].strip()


_SPAWNS = pytest.mark.parametrize("spawn", [_foreground, _background], ids=["foreground", "background"])


@pytest.mark.platforms("linux")
@_SPAWNS
def test_low_priority_lowers_terminal_niceness_and_normal_keeps_it(tmp_path, monkeypatch, spawn):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    base = os.nice(0)

    _write_priority(tmp_path, "normal")
    assert int(spawn(_LINUX_NICE, tmp_path, monkeypatch)) == base

    _write_priority(tmp_path, "low")
    assert int(spawn(_LINUX_NICE, tmp_path, monkeypatch)) == min(19, base + 10)


@pytest.mark.platforms("linux")
@_SPAWNS
def test_background_priority_is_nice_19_and_idle_io_on_linux(tmp_path, monkeypatch, spawn):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_priority(tmp_path, "background")

    assert int(spawn(_LINUX_NICE, tmp_path, monkeypatch)) == 19
    if shutil.which("ionice"):
        assert spawn(_LINUX_IO_CLASS, tmp_path, monkeypatch) == "idle"


@pytest.mark.platforms("macos")
@_SPAWNS
def test_background_priority_puts_terminal_tree_in_darwin_background(tmp_path, monkeypatch, spawn):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    _write_priority(tmp_path, "normal")
    assert int(spawn(_MACOS_PRI, tmp_path, monkeypatch)) != _DARWIN_BG_PRI

    _write_priority(tmp_path, "background")
    # A grandchild, not just the shell: what the build tool forks must inherit the state.
    assert int(spawn(f"/bin/sh -c '{_MACOS_PRI}'", tmp_path, monkeypatch)) == _DARWIN_BG_PRI


@contextmanager
def _strict_kernel():
    from tools.code_kernel import shutdown_all_kernels
    shutdown_all_kernels()
    try:
        with patch("tools.code_execution_tool._load_config", return_value={"mode": "strict", "timeout": 30}):
            yield
    finally:
        shutdown_all_kernels()


_KERNEL_PRI = ("import subprocess, os; print(subprocess.run(['ps', '-o', 'pri=', '-p', str(os.getpid())],"
               " capture_output=True, text=True).stdout.strip())")


@pytest.mark.platforms("macos")
def test_background_priority_applies_to_the_execute_code_kernel(tmp_path, monkeypatch):
    from tools.code_execution_tool import execute_code
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    _write_priority(tmp_path, "background")

    with _strict_kernel():
        result = json.loads(execute_code(_KERNEL_PRI, task_id="priority-test"))

    assert result["status"] == "success", result
    assert int(result["output"].strip().splitlines()[-1]) == _DARWIN_BG_PRI


@pytest.mark.parametrize("level", ["normal", "", None, "realtime", "LOW-ish"])
def test_unknown_or_normal_levels_never_prefix(level):
    assert priority_prefix_for(level, "linux") == []


@pytest.mark.parametrize("level", ["low", "background"])
def test_windows_is_a_documented_no_op(level):
    assert priority_prefix_for(level, "win32") == []
