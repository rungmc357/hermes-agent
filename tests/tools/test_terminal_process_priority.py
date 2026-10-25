"""terminal.process_priority: low runs host terminal commands below Hermes' own CPU priority."""

import os
import time

import pytest

from tools.environments.local import LocalEnvironment
from tools.process_registry import ProcessRegistry

_NICE = "cat /proc/self/stat | awk '{print $19}'"  # the shell's own niceness, no ps needed


def _write_priority(home, value):
    (home / "config.yaml").write_text(f"terminal:\n  process_priority: {value}\n", encoding="utf-8")


def _foreground_nice(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path), timeout=20)
    try:
        return int(env.execute(_NICE)["output"].strip().splitlines()[-1])
    finally:
        env.cleanup()


def _background_nice(tmp_path, monkeypatch):
    monkeypatch.setattr(ProcessRegistry, "_write_checkpoint", lambda self: None)
    session = ProcessRegistry().spawn_local(_NICE, cwd=str(tmp_path))
    session.process.wait(timeout=20)
    for _ in range(50):
        if session.output_buffer.strip():
            break
        time.sleep(0.1)
    return int(session.output_buffer.strip().splitlines()[-1])


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("spawn", [_foreground_nice, _background_nice], ids=["foreground", "background"])
def test_low_priority_lowers_terminal_niceness_and_normal_keeps_it(tmp_path, monkeypatch, spawn):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    base = os.nice(0)
    args = (tmp_path,) if spawn is _foreground_nice else (tmp_path, monkeypatch)

    _write_priority(tmp_path, "normal")
    assert spawn(*args) == base

    _write_priority(tmp_path, "low")
    os.utime(tmp_path / "config.yaml", (time.time() + 5, time.time() + 5))  # defeat the mtime cache
    assert spawn(*args) == min(19, base + 10)
