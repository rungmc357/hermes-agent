"""Host-pressure note for local terminal results.

When the host is saturated (run-queue far above the core count) or the Hermes data volume is
nearly full, more agent-launched work makes it worse: the gateway's event loop and its SQLite
writer starve first, and a full disk fails state writes outright. The agent cannot see either from
inside a command's output, so the next local terminal result carries one short note; it repeats
at most every ``_NOTE_INTERVAL_S`` per process while the pressure lasts.

Two cheap syscalls (``getloadavg`` + ``statvfs``) per local terminal result; any failure reads as
"no note", never as an error in the tool result.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

# Runnable threads per core. A busy build sits around 1-2; the 2026-11-05 incident that starved a
# gateway ran at ~28 (load 400 on 14 cores).
_LOAD_PER_CORE_ALERT = 4.0
_NOTE_INTERVAL_S = 600.0

_lock = threading.Lock()
_last_note_at: float | None = None


def _load_reason() -> str | None:
    try:
        load1 = os.getloadavg()[0]
    except (AttributeError, OSError):  # Windows has no getloadavg
        return None
    cores = os.cpu_count() or 1
    if load1 / cores < _LOAD_PER_CORE_ALERT:
        return None
    return f"load average {load1:.0f} on {cores} cores ({load1 / cores:.1f} per core)"


def _disk_reason(home: Path | None) -> str | None:
    from gateway.disk_status import collect_disk_status
    disk = collect_disk_status(home)
    if disk["pressure"] not in ("elevated", "critical"):
        return None
    return f"Hermes data disk {disk['used_percent']}% used ({disk['free_mb']} MB free)"


def host_pressure_reasons(home: Path | None = None) -> list[str]:
    """Human-readable pressure conditions on this host, worst-first; empty when healthy."""
    return [r for r in (_disk_reason(home), _load_reason()) if r]


def host_pressure_note(*, now: float | None = None, home: Path | None = None) -> str | None:
    """The note to attach to this local terminal result, or None (healthy, or already said
    within the interval)."""
    global _last_note_at
    now = time.monotonic() if now is None else now
    with _lock:
        if _last_note_at is not None and now - _last_note_at < _NOTE_INTERVAL_S:
            return None
        reasons = host_pressure_reasons(home)
        if not reasons:
            return None
        _last_note_at = now
    return ("Host under resource pressure: " + "; ".join(reasons) + ". Hermes itself may become "
            "unresponsive. Before starting more builds or tests: stop runaway background work, run "
            "heavy jobs one at a time (or set terminal.process_priority: background), and free disk "
            "space (build caches, simulators, derived data) if it is nearly full.")

