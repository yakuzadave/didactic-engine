"""Utilities for running external CLI tools.

This module centralizes subprocess invocation for external tools (Demucs, Basic Pitch, etc.)
so we can provide consistent:
- timeout handling
- stdout/stderr capture
- error messages that include the invoked command

These helpers intentionally keep dependencies limited to the Python stdlib.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class CommandResult:
    """Captured result for a command execution."""

    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str


def format_cmd(cmd: Sequence[str]) -> str:
    """Format a command for logs/errors."""
    return " ".join(shlex.quote(str(c)) for c in cmd)


def run_checked(
    cmd: Sequence[str],
    *,
    timeout_s: Optional[float] = None,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    tool_name: str = "command",
) -> CommandResult:
    """Run a command and raise RuntimeError with helpful context on failure.

    Args:
        cmd: Command tokens, e.g. ["demucs", "-n", "htdemucs", ...].
        timeout_s: Optional timeout in seconds.
        cwd: Optional working directory.
        env: Optional environment variables.
        tool_name: Friendly tool name used in error messages.

    Returns:
        CommandResult containing return code and captured output.

    Raises:
        RuntimeError: If the command fails or times out.
    """
    cmd_list = [str(c) for c in cmd]

    try:
        cp = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
            cwd=cwd,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{tool_name} timed out after {timeout_s}s\n"
            f"Command: {format_cmd(cmd_list)}"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{tool_name} command not found. Is it installed and on PATH?\n"
            f"Command: {format_cmd(cmd_list)}"
        ) from exc

    result = CommandResult(
        cmd=cmd_list,
        returncode=int(cp.returncode),
        stdout=cp.stdout or "",
        stderr=cp.stderr or "",
    )

    if result.returncode != 0:
        # Include both stdout and stderr; some tools report progress on stdout.
        raise RuntimeError(
            f"{tool_name} failed with exit code {result.returncode}\n"
            f"Command: {format_cmd(cmd_list)}\n"
            f"--- stdout ---\n{result.stdout.strip()}\n"
            f"--- stderr ---\n{result.stderr.strip()}\n"
        )

    return result
