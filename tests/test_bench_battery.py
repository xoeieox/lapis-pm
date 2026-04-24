"""Unit tests for lapis_pm.bench — LLM-free (subprocess mocked).

Coverage:
  - Battery YAML with valid pairs parses correctly.
  - Battery with malformed pairs raises a clear error.
  - spawn_stripped constructs expected command + env (subprocess mocked).
  - spawn_stripped timeout maps to a captured failure entry, not an exception.
  - compare over two identical captures returns empty deltas.
  - compare over captures with differing responses returns non-empty deltas.
  - CLI entry point parses --battery, --out, --timeout and dispatches correctly.
"""

from __future__ import annotations

import io
import json
import subprocess
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm.bench.__main__ import main
from lapis_pm.bench.battery import BatteryError, parse_battery, load_battery
from lapis_pm.bench.compare import compare
from lapis_pm.bench.fresh_instance import spawn_stripped


# ---------------------------------------------------------------------------
# Battery YAML: valid parse
# ---------------------------------------------------------------------------

VALID_YAML = """\
name: test_battery
description: A test battery
pairs:
  - id: pair-1
    loud:
      topic: "What is Python?"
      expected_awareness: "A programming language"
    quiet:
      topic: "What changed in our stack last week?"
      expected_awareness: "Nothing the baseline would know"
  - id: pair-2
    loud:
      topic: "What is JSON?"
      expected_awareness: "JavaScript Object Notation"
    quiet:
      topic: "Which serialiser do we use now?"
      expected_awareness: "TOML in our stack"
"""


def test_parse_battery_valid():
    battery = parse_battery(VALID_YAML)
    assert battery["name"] == "test_battery"
    assert len(battery["pairs"]) == 2
    assert battery["pairs"][0]["id"] == "pair-1"
    assert battery["pairs"][0]["loud"]["topic"] == "What is Python?"
    assert battery["pairs"][1]["quiet"]["topic"] == "Which serialiser do we use now?"


def test_parse_battery_preserves_expected_awareness():
    battery = parse_battery(VALID_YAML)
    assert battery["pairs"][0]["loud"]["expected_awareness"] == "A programming language"


# ---------------------------------------------------------------------------
# Battery YAML: malformed → BatteryError
# ---------------------------------------------------------------------------

def test_parse_battery_missing_name():
    yaml_text = "pairs:\n  - id: x\n    loud: {topic: a}\n    quiet: {topic: b}\n"
    with pytest.raises(BatteryError, match="name"):
        parse_battery(yaml_text)


def test_parse_battery_missing_pairs_key():
    with pytest.raises(BatteryError, match="pairs"):
        parse_battery("name: foo\n")


def test_parse_battery_empty_pairs_list():
    with pytest.raises(BatteryError, match="pairs"):
        parse_battery("name: foo\npairs: []\n")


def test_parse_battery_not_a_mapping():
    with pytest.raises(BatteryError):
        parse_battery("- item1\n- item2\n")


def test_parse_battery_pair_missing_id():
    yaml_text = "name: foo\npairs:\n  - loud: {topic: a}\n    quiet: {topic: b}\n"
    with pytest.raises(BatteryError, match="id"):
        parse_battery(yaml_text)


def test_parse_battery_pair_missing_loud():
    yaml_text = "name: foo\npairs:\n  - id: x\n    quiet: {topic: b}\n"
    with pytest.raises(BatteryError, match="loud"):
        parse_battery(yaml_text)


def test_parse_battery_pair_missing_quiet():
    yaml_text = "name: foo\npairs:\n  - id: x\n    loud: {topic: a}\n"
    with pytest.raises(BatteryError, match="quiet"):
        parse_battery(yaml_text)


def test_parse_battery_pair_missing_topic_in_loud():
    yaml_text = (
        "name: foo\npairs:\n"
        "  - id: x\n    loud: {expected_awareness: blah}\n    quiet: {topic: b}\n"
    )
    with pytest.raises(BatteryError, match="topic"):
        parse_battery(yaml_text)


def test_parse_battery_pair_not_a_mapping():
    yaml_text = "name: foo\npairs:\n  - just a string\n"
    with pytest.raises(BatteryError):
        parse_battery(yaml_text)


# ---------------------------------------------------------------------------
# load_battery: bundled loud_quiet.yaml
# ---------------------------------------------------------------------------

def test_load_loud_quiet_battery():
    """The bundled loud_quiet battery loads and has >= 5 pairs."""
    battery = load_battery("loud_quiet")
    assert battery["name"] == "loud_quiet"
    assert len(battery["pairs"]) >= 5
    for pair in battery["pairs"]:
        assert "id" in pair
        assert "topic" in pair["loud"]
        assert "topic" in pair["quiet"]


def test_load_battery_not_found():
    with pytest.raises(FileNotFoundError):
        load_battery("nonexistent_battery_xyz")


# ---------------------------------------------------------------------------
# spawn_stripped: subprocess mocked — command construction
# ---------------------------------------------------------------------------

def _make_proc(stdout="response", returncode=0, stderr=""):
    proc = MagicMock()
    proc.stdout = stdout
    proc.returncode = returncode
    proc.stderr = stderr
    return proc


def test_spawn_stripped_calls_claude_p():
    with patch("subprocess.run", return_value=_make_proc()) as mock_run:
        spawn_stripped("Hello world", timeout=30)

    assert mock_run.called
    cmd = mock_run.call_args.args[0]
    assert cmd[0] == "claude"
    assert cmd[1] == "-p"
    assert cmd[2] == "Hello world"


def test_spawn_stripped_uses_temp_home():
    """HOME in child env must differ from the real HOME."""
    import os
    real_home = os.environ.get("HOME", "")
    captured_env: dict = {}

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env", {}))
        return _make_proc()

    with patch("subprocess.run", side_effect=fake_run):
        spawn_stripped("test")

    assert captured_env.get("HOME", real_home) != real_home


def test_spawn_stripped_sets_strip_memory_env():
    captured_env: dict = {}

    def fake_run(cmd, **kwargs):
        captured_env.update(kwargs.get("env", {}))
        return _make_proc()

    with patch("subprocess.run", side_effect=fake_run):
        spawn_stripped("test")

    assert captured_env.get("LAPIS_BENCH_STRIP_MEMORY") == "1"


def test_spawn_stripped_settings_json_has_no_hooks():
    """Temp HOME's settings.json must have an empty hooks dict."""
    settings_seen: dict = {}

    def fake_run(cmd, **kwargs):
        home = kwargs["env"]["HOME"]
        p = Path(home) / ".claude" / "settings.json"
        if p.exists():
            settings_seen.update(json.loads(p.read_text()))
        return _make_proc()

    with patch("subprocess.run", side_effect=fake_run):
        spawn_stripped("check settings")

    assert "hooks" in settings_seen
    assert settings_seen["hooks"] == {}
    assert "mcpServers" in settings_seen
    assert settings_seen["mcpServers"] == {}


def test_spawn_stripped_returns_expected_keys():
    with patch("subprocess.run", return_value=_make_proc(stdout="my response")):
        result = spawn_stripped("prompt")

    for key in ("schema_version", "prompt", "response", "model",
                 "duration_s", "exit_code", "stderr", "stripped_state", "timed_out"):
        assert key in result, f"missing key: {key}"

    assert result["prompt"] == "prompt"
    assert result["response"] == "my response"
    assert result["exit_code"] == 0
    assert result["timed_out"] is False
    assert isinstance(result["duration_s"], float)


def test_spawn_stripped_captures_exit_code():
    with patch("subprocess.run", return_value=_make_proc(returncode=1)):
        result = spawn_stripped("failing prompt")
    assert result["exit_code"] == 1


def test_spawn_stripped_records_stripped_state():
    with patch("subprocess.run", return_value=_make_proc()):
        result = spawn_stripped("check state")
    state = result["stripped_state"]
    assert "settings_json" in state
    assert "memory_md" in state
    assert state["settings_json"]["hooks"] == {}


# ---------------------------------------------------------------------------
# spawn_stripped: timeout → captured failure, no exception
# ---------------------------------------------------------------------------

def test_spawn_stripped_timeout_returns_failure_not_exception():
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["claude"], 30)):
        result = spawn_stripped("slow prompt", timeout=30)

    assert result["timed_out"] is True
    assert result["exit_code"] == -1
    assert result["response"] == ""
    assert "30" in result["stderr"]   # timeout value noted in stderr


def test_spawn_stripped_timeout_preserves_prompt():
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["claude"], 5)):
        result = spawn_stripped("my prompt", timeout=5)
    assert result["prompt"] == "my prompt"


# ---------------------------------------------------------------------------
# compare: identical captures → empty deltas
# ---------------------------------------------------------------------------

def _write_capture(path: Path, results: list[dict]) -> None:
    path.write_text(json.dumps({
        "schema_version": 1,
        "battery_name": "test",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "results": results,
    }))


def _result(pair_id: str, loud_resp: str, quiet_resp: str) -> dict:
    def side(resp: str) -> dict:
        return {
            "schema_version": 1,
            "prompt": "q",
            "response": resp,
            "model": None,
            "duration_s": 1.0,
            "exit_code": 0,
            "stderr": "",
            "stripped_state": {},
            "timed_out": False,
        }
    return {"pair_id": pair_id, "loud": side(loud_resp), "quiet": side(quiet_resp)}


def test_compare_identical_captures_all_empty_deltas(tmp_path):
    results = [
        _result("pair-a", "loud A", "quiet A"),
        _result("pair-b", "loud B", "quiet B"),
    ]
    p = tmp_path / "cap.json"
    _write_capture(p, results)

    deltas = compare(p, p)

    assert set(deltas.keys()) == {"pair-a", "pair-b"}
    for pid, d in deltas.items():
        assert d["loud_delta"] == "", f"{pid} loud_delta not empty"
        assert d["quiet_delta"] == "", f"{pid} quiet_delta not empty"


# ---------------------------------------------------------------------------
# compare: differing captures → non-empty deltas keyed by pair_id
# ---------------------------------------------------------------------------

def test_compare_differing_responses_nonempty(tmp_path):
    baseline = tmp_path / "baseline.json"
    labeled = tmp_path / "labeled.json"
    _write_capture(baseline, [_result("pair-x", "old loud", "old quiet")])
    _write_capture(labeled, [_result("pair-x", "new loud", "new quiet")])

    deltas = compare(baseline, labeled)

    assert "pair-x" in deltas
    assert deltas["pair-x"]["loud_delta"] != ""
    assert deltas["pair-x"]["quiet_delta"] != ""


def test_compare_partial_loud_difference(tmp_path):
    """Only loud differs — quiet delta should be empty."""
    baseline = tmp_path / "b.json"
    labeled = tmp_path / "l.json"
    _write_capture(baseline, [_result("p", "old loud", "same quiet")])
    _write_capture(labeled, [_result("p", "new loud", "same quiet")])

    deltas = compare(baseline, labeled)

    assert deltas["p"]["loud_delta"] != ""
    assert deltas["p"]["quiet_delta"] == ""


def test_compare_pair_missing_in_labeled(tmp_path):
    baseline = tmp_path / "b.json"
    labeled = tmp_path / "l.json"
    _write_capture(baseline, [_result("only-in-baseline", "a", "b")])
    _write_capture(labeled, [])

    deltas = compare(baseline, labeled)

    assert "only-in-baseline" in deltas
    assert "missing" in deltas["only-in-baseline"]["loud_delta"].lower()


def test_compare_pair_missing_in_baseline(tmp_path):
    baseline = tmp_path / "b.json"
    labeled = tmp_path / "l.json"
    _write_capture(baseline, [])
    _write_capture(labeled, [_result("only-in-labeled", "a", "b")])

    deltas = compare(baseline, labeled)

    assert "only-in-labeled" in deltas
    assert "missing" in deltas["only-in-labeled"]["loud_delta"].lower()


# ---------------------------------------------------------------------------
# CLI: entry point dispatch
# ---------------------------------------------------------------------------

def test_cli_no_args_nonzero():
    rc = main([])
    assert rc != 0


def test_cli_run_top_level_flags(tmp_path):
    """--battery + --out (no subcommand) runs battery and writes JSON."""
    cap = {"schema_version": 1, "battery_name": "loud_quiet",
           "timestamp": "t", "results": []}
    out = tmp_path / "out.json"

    with (
        patch("lapis_pm.bench.__main__.load_battery",
              return_value={"name": "loud_quiet", "pairs": []}) as mock_load,
        patch("lapis_pm.bench.__main__.run_battery", return_value=cap) as mock_run,
    ):
        rc = main(["--battery", "loud_quiet", "--out", str(out)])

    assert rc == 0
    mock_load.assert_called_once_with("loud_quiet")
    mock_run.assert_called_once()
    assert out.exists()
    assert json.loads(out.read_text())["battery_name"] == "loud_quiet"


def test_cli_run_subcommand(tmp_path):
    cap = {"schema_version": 1, "battery_name": "x", "timestamp": "t", "results": []}
    out = tmp_path / "out.json"

    with (
        patch("lapis_pm.bench.__main__.load_battery", return_value={"name": "x", "pairs": []}),
        patch("lapis_pm.bench.__main__.run_battery", return_value=cap),
    ):
        rc = main(["run", "--battery", "x", "--out", str(out)])

    assert rc == 0


def test_cli_timeout_arg_forwarded(tmp_path):
    """--timeout value must reach run_battery."""
    cap = {"schema_version": 1, "battery_name": "x", "timestamp": "t", "results": []}
    out = tmp_path / "out.json"
    recorded: list[int] = []

    def fake_run(battery, timeout=180):
        recorded.append(timeout)
        return cap

    with (
        patch("lapis_pm.bench.__main__.load_battery", return_value={"name": "x", "pairs": []}),
        patch("lapis_pm.bench.__main__.run_battery", side_effect=fake_run),
    ):
        main(["--battery", "x", "--out", str(out), "--timeout", "60"])

    assert recorded == [60]


def test_cli_compare_subcommand_self(tmp_path):
    """compare subcommand on identical files prints JSON and empty-delta note."""
    results = [_result("p1", "same", "same")]
    p = tmp_path / "cap.json"
    _write_capture(p, results)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["compare", str(p), str(p)])

    assert rc == 0
    output = buf.getvalue()
    assert "p1" in output
    # Self-compare should note identical
    assert "identical" in output.lower() or "no deltas" in output.lower()


def test_cli_out_creates_parent_dirs(tmp_path):
    """Output path is created even if parent directories don't exist."""
    cap = {"schema_version": 1, "battery_name": "x", "timestamp": "t", "results": []}
    out = tmp_path / "nested" / "deep" / "out.json"

    with (
        patch("lapis_pm.bench.__main__.load_battery", return_value={"name": "x", "pairs": []}),
        patch("lapis_pm.bench.__main__.run_battery", return_value=cap),
    ):
        rc = main(["--battery", "x", "--out", str(out)])

    assert rc == 0
    assert out.exists()
