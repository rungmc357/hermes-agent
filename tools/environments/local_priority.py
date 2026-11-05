"""Host scheduling priority for agent-launched local processes (``terminal.process_priority``).

Agent tool work (``make -j``, ``xcodebuild``, a pytest swarm) shares the host with the Hermes
process that answers every profile. At the same priority a few build lanes starve the gateway's
event loop and its SQLite writer. The prefix returned here runs the child below Hermes' own
priority; niceness, the macOS QoS clamp / darwin-background state and the Linux I/O class are all
inherited by everything the child starts, so the whole tree is covered.

An argv prefix rather than a ``preexec_fn``: ``preexec_fn`` is unsafe in the threaded gateway, and a
macOS QoS clamp set there does not survive ``exec``. Every prefix tool ``exec``s its target, so the
spawned pid/pgid is the target's and kill/timeout tracking is unchanged.
"""

from __future__ import annotations

# level -> host family -> ordered launchers; a launcher missing on the host is skipped, never fatal.
# darwin: nice is mostly ignored by the macOS scheduler, so QoS is the lever. ``-b`` is
# PRIO_DARWIN_BG: lowest CPU tier plus throttled disk and network I/O (what a build/simulator
# swarm hits hardest). posix: nice 19 + the idle I/O class (unprivileged on Linux; absent elsewhere).
_LAUNCHERS: dict[str, dict[str, tuple[tuple[str, tuple[str, ...]], ...]]] = {
    "low": {
        "darwin": (("/usr/sbin/taskpolicy", ("-c", "utility")),),
        "posix": (("nice", ("-n", "10")),),
    },
    "background": {
        "darwin": (("/usr/sbin/taskpolicy", ("-b",)),),
        "posix": (("nice", ("-n", "19")), ("ionice", ("-c", "3"))),
    },
}
PROCESS_PRIORITY_LEVELS = ("normal", *_LAUNCHERS)


def priority_prefix_for(level: str | None, family: str) -> list[str]:
    """argv prefix for ``level`` on host ``family`` (``darwin``/``linux``/``win32``/...).
    Unknown levels and Windows are normal priority; a darwin host without taskpolicy falls back to
    the POSIX launchers."""
    from hermes_platform.resolver import locate_command
    table = _LAUNCHERS.get(str(level or "").strip().lower())
    if not table or family == "win32":
        return []
    prefix: list[str] = []
    rows = table["darwin"] if family == "darwin" else ()
    if not any(locate_command(name).command for name, _ in rows):
        rows = table["posix"]
    for name, args in rows:
        if command := locate_command(name).command:
            prefix += [*command, *args]
    return prefix


def process_priority_prefix() -> list[str]:
    """Prefix for the configured ``terminal.process_priority`` on this host. An unavailable config
    home (unmounted, unreadable) means normal priority rather than a failed spawn."""
    from hermes_cli.config import load_config_readonly
    from hermes_cli.config_home import HomeInitializationError
    from hermes_platform.host.facts import os_family
    try:
        terminal_cfg = load_config_readonly().get("terminal")
    except HomeInitializationError:
        return []
    level = terminal_cfg.get("process_priority") if isinstance(terminal_cfg, dict) else None
    return priority_prefix_for(level, os_family())
