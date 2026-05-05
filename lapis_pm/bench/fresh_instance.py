"""Stripped Claude Code instance spawner.

Design choice (load-bearing open question #1):
  We use option (a) — temp HOME entirely.  A fresh temporary directory is
  created per-call and set as HOME for the child process.  This guarantees:
    - No chub-inject hook fires  (settings.json in temp HOME has no hooks)
    - No MEMORY.md is injected   (temp HOME has only a stub MEMORY.md)
    - No subagents are loaded    (settings.json has empty mcpServers)
  The user's real ~/.claude/ is never touched.

Design choice (load-bearing open question #2):
  The stub MEMORY.md contains a single minimal line:
      "You have no persisted memory for this session."
  Early trial runs showed that a completely empty file causes the baseline
  to sometimes speculate about missing config; the minimal stub keeps it
  from generating noise about a missing file while still constituting a
  clean baseline.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

CAPTURE_SCHEMA_VERSION = 1

# Stub MEMORY.md written into the temp HOME.  Kept minimal to avoid priming
# the model with content — the single line just prevents "file not found"
# speculation without providing substantive context.
_STUB_MEMORY = "You have no persisted memory for this session.\n"

# Minimal settings.json: no hooks, no MCP servers, no agents.
_STUB_SETTINGS: dict = {
    "version": 1,
    "hooks": {},
    "mcpServers": {},
}


# Path to the operator-installed Synapse per-prompt hook.
# Operator copies synapse/hooks/synapse-inject.py here per INSTALL.md.
SYNAPSE_HOOK_PATH: str = os.environ.get(
    "SYNAPSE_HOOK_PATH",
    "/home/user/.claude/hooks/synapse-inject.py",
)

# Settings.json used when synapse=True: strips chub-inject + MCP servers but
# enables the Synapse UserPromptSubmit hook (synchronous, no async:true).
def _synapse_settings() -> dict:
    return {
        "version": 1,
        "hooks": {
            "UserPromptSubmit": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": SYNAPSE_HOOK_PATH,
                            "statusMessage": (
                                "synapse-inject (load-bearing for Synapse Phase 2"
                                " — do not remove)"
                            ),
                        }
                    ]
                }
            ]
        },
        "mcpServers": {},
    }


def spawn_stripped(prompt: str, timeout: int = 180, synapse: bool = False) -> dict:
    """Run a fresh, stripped Claude Code instance against *prompt*.

    Strips:
      - MEMORY.md injection: temp HOME contains only a stub MEMORY.md;
        because the hooks section is empty, chub-inject never fires.
      - chub-inject SessionStart hook: temp HOME's settings.json has no
        hooks key, so no user-level hooks are registered.
      - Subagents / MCP servers: settings.json has an empty mcpServers map.

    When *synapse* is True:
      - settings.json in the temp HOME is populated with ONLY the Synapse
        UserPromptSubmit hook entry (chub-inject still suppressed, MCP servers
        still empty).  The hook script path is SYNAPSE_HOOK_PATH (default
        ``/home/user/.claude/hooks/synapse-inject.py``; override via
        ``SYNAPSE_HOOK_PATH`` env var).
      - The return dict gains a ``synapse_state`` key with the hook path and
        settings that were used, so a reviewer can reproduce the run.

    The temp HOME is cleaned up after the subprocess exits.  The stripped
    state (settings.json contents, stub MEMORY.md text, temp HOME path) is
    recorded in the return value so a reviewer can reproduce the run.

    Returns::

        {
            "schema_version": int,
            "prompt":         str,
            "response":       str,
            "model":          str | None,   # extracted from stderr if present
            "duration_s":     float,
            "exit_code":      int,          # -1 on timeout
            "stderr":         str,
            "stripped_state": {
                "settings_json": dict,
                "memory_md":     str,
            },
            "timed_out":      bool,
            # Only present when synapse=True:
            "synapse_state":  {
                "hook_path":     str,
                "settings_json": dict,
            },
        }
    """
    with tempfile.TemporaryDirectory(prefix="lapis-bench-") as tmp_home:
        tmp_home_path = Path(tmp_home)

        # .claude/ directory
        claude_dir = tmp_home_path / ".claude"
        claude_dir.mkdir()

        # Mirror the real HOME's Claude credentials so the stripped
        # instance can authenticate.  Without this every `claude -p`
        # exits with "Not logged in · Please run /login" and the
        # bench captures only the error string.  The temp HOME is
        # torn down on subprocess exit so the mirror is short-lived.
        real_creds = Path.home() / ".claude" / ".credentials.json"
        if real_creds.is_file():
            tmp_creds = claude_dir / ".credentials.json"
            tmp_creds.write_bytes(real_creds.read_bytes())
            tmp_creds.chmod(0o600)

        # Select the settings.json to write based on the synapse flag.
        active_settings = _synapse_settings() if synapse else _STUB_SETTINGS
        settings_path = claude_dir / "settings.json"
        settings_path.write_text(json.dumps(active_settings, indent=2))

        # Write a stub MEMORY.md.  The path mirrors what chub-inject.py
        # would look for under a projects/ sub-dir; since the hook never
        # fires (no hooks in settings), the exact path doesn't matter, but
        # we create it so any fallback lookup finds the stub.
        stub_memory_dir = claude_dir / "projects" / "bench-stub" / "memory"
        stub_memory_dir.mkdir(parents=True)
        stub_memory_path = stub_memory_dir / "MEMORY.md"
        stub_memory_path.write_text(_STUB_MEMORY)

        stripped_state = {
            "settings_json": active_settings,
            "memory_md": _STUB_MEMORY,
        }

        env = os.environ.copy()
        env["HOME"] = str(tmp_home_path)
        # Sentinel env var — useful if any lapis hook ever wants to detect
        # that it's running inside a bench harness and opt out gracefully.
        env["LAPIS_BENCH_STRIP_MEMORY"] = "1"

        start = time.monotonic()
        try:
            result = subprocess.run(
                ["claude", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            duration_s = time.monotonic() - start
            capture = {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "prompt": prompt,
                "response": result.stdout,
                "model": _extract_model(result.stdout, result.stderr),
                "duration_s": round(duration_s, 3),
                "exit_code": result.returncode,
                "stderr": result.stderr,
                "stripped_state": stripped_state,
                "timed_out": False,
            }
            if synapse:
                capture["synapse_state"] = {
                    "hook_path": SYNAPSE_HOOK_PATH,
                    "settings_json": active_settings,
                }
            return capture
        except subprocess.TimeoutExpired:
            duration_s = time.monotonic() - start
            capture = {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "prompt": prompt,
                "response": "",
                "model": None,
                "duration_s": round(duration_s, 3),
                "exit_code": -1,
                "stderr": f"Timeout after {timeout}s",
                "stripped_state": stripped_state,
                "timed_out": True,
            }
            if synapse:
                capture["synapse_state"] = {
                    "hook_path": SYNAPSE_HOOK_PATH,
                    "settings_json": active_settings,
                }
            return capture


def _extract_model(stdout: str, stderr: str) -> str | None:
    """Try to extract model identifier from claude output."""
    combined = stderr + "\n" + stdout
    for line in combined.splitlines():
        lo = line.lower()
        if "model" in lo and any(k in lo for k in ("claude", "sonnet", "opus", "haiku")):
            return line.strip()
    return None
