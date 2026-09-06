"""Tests for U2 - precedent actuator (lapis-pm-autonomy-actuator-v0).

Coverage (per spec AC list):
  AC3  shadow default             -> no merge, shadow record written, brief intact
  AC5  dial unknown value         -> fail-safe to shadow (no merge)
  AC5  malformed adjudication     -> no match (brief as today)
  AC5  lineage verification fail  -> no match (missing outcome / wrong target /
                                     outside 24h window)
  AC5  merge exception enforce    -> fall through to brief (never swallowed)
  AC5  hook exception             -> fall through to brief (never swallowed)
  U2.1 derive_fork_class          -> deterministic fork tuple
  U2.2 find_precedent             -> Design 5 verification chain
  U2.3 cmd_ratify adjudication    -> written alongside the outcome, idempotent
  U2.5 enforce match + promoted   -> merge + resolution record + class counter
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from lapis_pm import precedent as pc
from lapis_pm import verification_attestation as va


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mem(store: dict | None = None) -> MagicMock:
    """Mock MemoryStore with .set / .get / .search wired up."""
    data: dict[str, dict] = {}
    if store:
        data.update(store)

    mock = MagicMock()

    def _set(key: str, content: str, tags: list | None = None, source: str = "") -> bool:
        data[key] = {"key": key, "content": content, "tags": tags or []}
        return True

    def _get(key: str) -> dict | None:
        return data.get(key)

    def _search(query: str = "", tags: list | None = None, limit: int = 100) -> list[str]:
        # Real MemoryStore.search matches on key substring OR tag substring.
        out = []
        for k, v in data.items():
            if query and (query in k or any(query in t for t in (v.get("tags") or []))):
                out.append(k)
        return out[:limit]

    mock.set.side_effect = _set
    mock.get.side_effect = _get
    mock.search.side_effect = _search
    mock._data = data
    return mock


def _make_cls(
    changed_paths: list[str] | None = None,
    loc: int = 50,
    held_paths: list | None = None,
    issues: list | None = None,
) -> MagicMock:
    cls = MagicMock()
    cls.screen_verdict = "clean"
    cls.issues = issues if issues is not None else []
    cls.pr_number = 42
    cls.repo = "lapis-pm"
    cls.title = "feat: add thing"
    cls.html_url = "http://forgejo/pr/42"
    cls.diff = ""
    cls.diff_loc = loc
    cls.loc = loc
    cls.reasons = []
    cls.changed_paths = changed_paths if changed_paths is not None else ["lapis_pm/foo.py"]
    cls.held_paths = held_paths if held_paths is not None else []
    return cls


def _make_target(pm_authority: str = "advisory",
                 pm_verification: str = "pm-live-test") -> MagicMock:
    t = MagicMock()
    t.id = "my-target"
    t.pm_authority = pm_authority
    t.pm_repo = "lapis-pm"
    t.data = {"pm_verification": pm_verification}
    return t


def _adjudication_body(target_id: str = "my-target", pr: int = 42,
                       chosen: str = "confirm", repo: str = "lapis-pm",
                       fork: dict | None = None, ts: str = "2026-09-06T00:00:00Z",
                       citations: list | None = None,
                       source: str = "human") -> dict:
    if fork is None:
        fork = {
            "repo": repo, "verdict_class": "clean",
            "change_classes": ["code"], "loc_bucket": "lt100",
            "held_paths_class": "none",
        }
    if citations is None:
        citations = ["router/lapis-pm/ratification-outcomes/eid-001"]
    return {
        "schema": "adjudication/v1",
        "ts": ts,
        "repo": repo,
        "target_id": target_id,
        "fork": fork,
        "chosen": chosen,
        "intent": "test intent",
        "source": source,
        "pr": pr,
        "citations": citations,
    }


def _outcome_body(target_id: str = "my-target", outcome: str = "confirm",
                  created: str = "2026-09-06T00:00:00Z") -> dict:
    return {
        "ratification_outcome": outcome,
        "target_id": target_id,
        "created": created,
    }


def _seed_precedent(mem, target_id: str = "my-target", pr: int = 42,
                    chosen: str = "confirm", ts: str = "2026-09-06T00:00:00Z",
                    created: str = "2026-09-06T00:00:00Z",
                    outcome_target: str = "my-target",
                    outcome: str = "confirm",
                    source: str = "human") -> str:
    """Seed a well-formed adjudication record + its cited ratify outcome."""
    body = _adjudication_body(target_id=target_id, pr=pr, chosen=chosen,
                              ts=ts, source=source)
    key = f"decision/adjudication/{target_id}-{pr}-{chosen}"
    content = (
        f"<!-- adjudication record, schema adjudication/v1 -->\n"
        f"```json\n{json.dumps(body, ensure_ascii=False, sort_keys=True)}\n```\n"
    )
    mem.set(key, content, tags=["lapis-pm", "adjudication", f"target:{target_id}"])
    # Cited outcome.
    okey = body["citations"][0]
    mem.set(okey, json.dumps(_outcome_body(target_id=outcome_target,
                                            outcome=outcome, created=created)),
            tags=["lapis-pm", "router-portfolio"])
    return key


# ---------------------------------------------------------------------------
# U2.1 - derive_fork_class (deterministic fork tuple)
# ---------------------------------------------------------------------------

class TestDeriveForkClass:

    def test_code_only(self):
        cls = _make_cls(changed_paths=["lapis_pm/foo.py"], loc=50)
        fork = pc.derive_fork_class(cls, "clean", repo="lapis-pm")
        assert fork == {
            "repo": "lapis-pm",
            "verdict_class": "clean",
            "change_classes": ["code"],
            "loc_bucket": "lt100",
            "held_paths_class": "none",
        }

    def test_change_classes_buckets(self):
        cls = _make_cls(
            changed_paths=["systemd/x.timer", "tests/test_y.py",
                           "docs/readme.md", "lapis_pm/foo.py"],
            loc=200,
        )
        fork = pc.derive_fork_class(cls, "clean", repo="lapis-pm")
        assert set(fork["change_classes"]) == {"config", "test", "docs", "code"}
        assert fork["loc_bucket"] == "100_400"

    def test_deploy_bucket(self):
        cls = _make_cls(changed_paths=["deploy/unit.service"], loc=50)
        fork = pc.derive_fork_class(cls, "clean", repo="lapis-pm")
        # deploy/ is its own bucket; .service also hits config.
        assert "deploy" in fork["change_classes"]
        assert "config" in fork["change_classes"]

    def test_loc_bucket_gt400(self):
        cls = _make_cls(changed_paths=["lapis_pm/foo.py"], loc=500)
        fork = pc.derive_fork_class(cls, "clean", repo="lapis-pm")
        assert fork["loc_bucket"] == "gt400"

    def test_held_paths_class(self):
        cls = _make_cls(changed_paths=["lapis_pm/foo.py"],
                        held_paths=["lapis_pm/authority.py"])
        fork = pc.derive_fork_class(cls, "clean", repo="lapis-pm")
        assert fork["held_paths_class"] == "held"

    def test_fork_hash_stable(self):
        fork = pc.derive_fork_class(_make_cls(), "clean", repo="lapis-pm")
        assert pc.fork_hash(fork) == pc.fork_hash(dict(fork))


# ---------------------------------------------------------------------------
# U2.2 - find_precedent (Design 5 verification chain)
# ---------------------------------------------------------------------------

class TestFindPrecedent:

    def test_well_formed_match(self):
        mem = _make_mem()
        _seed_precedent(mem)
        fork = pc.derive_fork_class(_make_cls(), "clean", repo="lapis-pm")
        rec = pc.find_precedent(mem, "lapis-pm", fork)
        assert rec is not None
        assert rec["chosen"] == "confirm"
        assert rec.get("_key").startswith("decision/adjudication/")

    def test_malformed_record_skipped(self):
        # AC5: malformed adjudication record -> no match (brief as today).
        mem = _make_mem()
        mem.set("decision/adjudication/bad-1-confirm",
                "```json\n{not valid json\n```",
                tags=["lapis-pm", "adjudication", "target:bad"])
        fork = pc.derive_fork_class(_make_cls(), "clean", repo="lapis-pm")
        rec = pc.find_precedent(mem, "lapis-pm", fork)
        assert rec is None
        assert pc.find_precedent.last_skipped >= 1

    def test_wrong_source_no_match(self):
        # source: "precedent" is lineage-only, never a primary match.
        mem = _make_mem()
        _seed_precedent(mem, source="precedent")
        fork = pc.derive_fork_class(_make_cls(), "clean", repo="lapis-pm")
        rec = pc.find_precedent(mem, "lapis-pm", fork)
        assert rec is None

    def test_missing_cited_outcome_no_match(self):
        # AC5: lineage verification failure (missing outcome key).
        mem = _make_mem()
        body = _adjudication_body()
        key = "decision/adjudication/my-target-42-confirm"
        content = (
            f"```json\n{json.dumps(body, ensure_ascii=False, sort_keys=True)}\n```\n"
        )
        mem.set(key, content, tags=["lapis-pm", "adjudication", "target:my-target"])
        # Note: the cited outcome key is NOT seeded.
        fork = pc.derive_fork_class(_make_cls(), "clean", repo="lapis-pm")
        rec = pc.find_precedent(mem, "lapis-pm", fork)
        assert rec is None

    def test_wrong_target_id_no_match(self):
        # AC5: lineage verification failure (outcome target_id mismatch).
        mem = _make_mem()
        _seed_precedent(mem, outcome_target="other-target")
        fork = pc.derive_fork_class(_make_cls(), "clean", repo="lapis-pm")
        rec = pc.find_precedent(mem, "lapis-pm", fork)
        assert rec is None

    def test_outside_24h_window_no_match(self):
        # AC5: lineage verification failure (outside 24h window).
        mem = _make_mem()
        _seed_precedent(mem, ts="2026-09-06T00:00:00Z",
                        created="2026-08-01T00:00:00Z")
        fork = pc.derive_fork_class(_make_cls(), "clean", repo="lapis-pm")
        rec = pc.find_precedent(mem, "lapis-pm", fork)
        assert rec is None

    def test_non_confirm_choice_no_match(self):
        # Only confirm is merge-class; correct/override/redirect never match.
        mem = _make_mem()
        _seed_precedent(mem, chosen="correct")
        fork = pc.derive_fork_class(_make_cls(), "clean", repo="lapis-pm")
        rec = pc.find_precedent(mem, "lapis-pm", fork)
        assert rec is None

    def test_held_fork_no_match(self):
        mem = _make_mem()
        _seed_precedent(mem)
        fork = pc.derive_fork_class(_make_cls(held_paths=["x"]), "clean",
                                   repo="lapis-pm")
        rec = pc.find_precedent(mem, "lapis-pm", fork)
        assert rec is None


# ---------------------------------------------------------------------------
# AC3 - shadow default: no merge, shadow record written, brief intact
# ---------------------------------------------------------------------------

class TestShadowDefault:

    def test_shadow_no_merge_record_written(self, tmp_path):
        """With no env var set (default shadow), a matching precedent yields
        (a) no merge call, (b) a shadow record, (c) the brief is still emitted
        (the hook returns None)."""
        from lapis_pm import pm_core

        cls = _make_cls()
        payload = {"pr": {"head": {"sha": "abc123"}}, "classification": cls}
        mem = _make_mem()
        _seed_precedent(mem)

        with (
            patch.object(pm_core, "_mem", return_value=mem),
            patch.object(pc, "record_shadow_call") as mock_shadow,
            patch("lapis_pm.brief._act_merge_pr") as mock_merge,
            patch.dict("os.environ", {}, clear=True),
        ):
            action = pm_core._precedent_hook("my-target", cls, payload)

        # (a) no merge call
        assert not mock_merge.called
        # (b) a shadow record was written
        assert mock_shadow.called
        call_kwargs = mock_shadow.call_args.kwargs
        assert call_kwargs["dial"] == "shadow"
        assert call_kwargs["would_merge"] is True  # a match existed
        # (c) the brief is still emitted (hook returns None -> fall through)
        assert action is None

    def test_shadow_no_precedent(self, tmp_path):
        from lapis_pm import pm_core

        cls = _make_cls()
        payload = {"pr": {"head": {"sha": "abc123"}}, "classification": cls}
        mem = _make_mem()  # empty - no precedent

        with (
            patch.object(pm_core, "_mem", return_value=mem),
            patch.object(pc, "record_shadow_call") as mock_shadow,
            patch("lapis_pm.brief._act_merge_pr") as mock_merge,
            patch.dict("os.environ", {}, clear=True),
        ):
            action = pm_core._precedent_hook("my-target", cls, payload)

        assert not mock_merge.called
        assert mock_shadow.called
        assert mock_shadow.call_args.kwargs["would_merge"] is False
        assert action is None


# ---------------------------------------------------------------------------
# U2.5 - enforce: match + promoted class -> merge + resolution + counter
# ---------------------------------------------------------------------------

class TestEnforceMerge:

    def test_enforce_match_promoted_merges(self):
        from lapis_pm import pm_core

        cls = _make_cls()
        payload = {"pr": {"head": {"sha": "abc123"}}, "classification": cls}
        mem = _make_mem()
        prec_key = _seed_precedent(mem)

        # Promoted class record.
        fork = pc.derive_fork_class(cls, "clean", repo="lapis-pm")
        class_key = pc.class_record_key("lapis-pm", "clean",
                                        fork["change_classes"], fork["loc_bucket"])
        mem.set(class_key, json.dumps({"promoted": True,
                                       "auto_resolved_count": 0}),
                tags=["lapis-pm", "pm:autonomy-class"])

        with (
            patch.object(pm_core, "_mem", return_value=mem),
            patch.object(pc, "record_shadow_call") as mock_shadow,
            patch("lapis_pm.brief._act_merge_pr") as mock_merge,
            patch("lapis_pm.episodic.write_observation"),
            patch.dict("os.environ", {"PM_AUTONOMY_ACTUATOR": "enforce"}),
        ):
            action = pm_core._precedent_hook("my-target", cls, payload)

        # Merged via the proven path.
        assert mock_merge.called
        assert action is not None
        assert action.startswith("action:precedent_resolved:")
        assert f"pr={cls.pr_number}" in action
        # Resolution record written (source: precedent, precedent_of).
        res_key = pc.resolution_key("my-target", cls.pr_number, fork)
        assert res_key in mem._data
        # Class counter incremented.
        assert mem._data[class_key]["content"].count('"auto_resolved_count": 1') == 1

    def test_enforce_match_not_promoted_no_merge(self):
        # A class nobody has promoted never auto-resolves.
        from lapis_pm import pm_core

        cls = _make_cls()
        payload = {"pr": {"head": {"sha": "abc123"}}, "classification": cls}
        mem = _make_mem()
        _seed_precedent(mem)

        fork = pc.derive_fork_class(cls, "clean", repo="lapis-pm")
        class_key = pc.class_record_key("lapis-pm", "clean",
                                        fork["change_classes"], fork["loc_bucket"])
        mem.set(class_key, json.dumps({"promoted": False}),
                tags=["lapis-pm", "pm:autonomy-class"])

        with (
            patch.object(pm_core, "_mem", return_value=mem),
            patch.object(pc, "record_shadow_call") as mock_shadow,
            patch("lapis_pm.brief._act_merge_pr") as mock_merge,
            patch("lapis_pm.episodic.write_observation"),
            patch.dict("os.environ", {"PM_AUTONOMY_ACTUATOR": "enforce"}),
        ):
            action = pm_core._precedent_hook("my-target", cls, payload)

        assert not mock_merge.called
        assert action is None
        assert mock_shadow.call_args.kwargs["no_match_reason"] == "class_not_promoted"

    def test_enforce_merge_exception_falls_through(self):
        # AC5: merge exception in enforce mode -> brief as today (never swallowed).
        from lapis_pm import pm_core

        cls = _make_cls()
        payload = {"pr": {"head": {"sha": "abc123"}}, "classification": cls}
        mem = _make_mem()
        _seed_precedent(mem)

        fork = pc.derive_fork_class(cls, "clean", repo="lapis-pm")
        class_key = pc.class_record_key("lapis-pm", "clean",
                                        fork["change_classes"], fork["loc_bucket"])
        mem.set(class_key, json.dumps({"promoted": True}),
                tags=["lapis-pm", "pm:autonomy-class"])

        with (
            patch.object(pm_core, "_mem", return_value=mem),
            patch.object(pc, "record_shadow_call"),
            patch("lapis_pm.brief._act_merge_pr",
                   side_effect=RuntimeError("forgejo 4xx")),
            patch("lapis_pm.episodic.write_observation"),
            patch.dict("os.environ", {"PM_AUTONOMY_ACTUATOR": "enforce"}),
        ):
            action = pm_core._precedent_hook("my-target", cls, payload)

        # The merge raised; the hook must return None (fall through to brief).
        assert action is None


# ---------------------------------------------------------------------------
# AC5 - dial unknown value -> fail-safe to shadow (no merge)
# ---------------------------------------------------------------------------

class TestDialFailSafe:

    def test_unknown_dial_fails_safe_to_shadow(self):
        from lapis_pm import pm_core

        assert pm_core._autonomy_dial() == "shadow"  # default
        with patch.dict("os.environ", {"PM_AUTONOMY_ACTUATOR": "bogus"}):
            assert pm_core._autonomy_dial() == "shadow"  # unknown -> shadow
        with patch.dict("os.environ", {"PM_AUTONOMY_ACTUATOR": "enforce"}):
            assert pm_core._autonomy_dial() == "enforce"


# ---------------------------------------------------------------------------
# AC5 - hook exception -> fall through to brief (never swallowed)
# ---------------------------------------------------------------------------

class TestHookException:

    def test_hook_exception_returns_none(self):
        from lapis_pm import pm_core

        cls = _make_cls()
        payload = {"pr": {"head": {"sha": "abc123"}}, "classification": cls}
        with (
            patch.object(pm_core, "_mem", side_effect=RuntimeError("mem down")),
            patch.dict("os.environ", {"PM_AUTONOMY_ACTUATOR": "enforce"}),
        ):
            action = pm_core._precedent_hook("my-target", cls, payload)
        # A hook exception degrades to today's behavior (brief still emitted).
        assert action is None


# ---------------------------------------------------------------------------
# U2.3 - cmd_ratify writes the adjudication record (idempotent)
# ---------------------------------------------------------------------------

class TestRatifyAdjudication:

    def test_write_adjudication_idempotent(self):
        mem = _make_mem()
        fork = pc.derive_fork_class(_make_cls(), "clean", repo="lapis-pm")
        key1 = pc.write_adjudication(
            mem, target_id="my-target", pr_number=42, outcome="confirm",
            repo="lapis-pm", fork=fork, intent="first",
            citations=["router/lapis-pm/ratification-outcomes/eid-001"],
            invoked_interactive=True,
        )
        # Re-ratify the same (tid, pr, outcome): updates ts/intent, no 2nd key.
        key2 = pc.write_adjudication(
            mem, target_id="my-target", pr_number=42, outcome="confirm",
            repo="lapis-pm", fork=fork, intent="second",
            citations=["router/lapis-pm/ratification-outcomes/eid-001"],
            invoked_interactive=True,
        )
        assert key1 == key2
        assert key1 == "decision/adjudication/my-target-42-confirm"
        # Only one adjudication key for this triple.
        adj_keys = [k for k in mem._data if k.startswith("decision/adjudication/")]
        assert len(adj_keys) == 1
        # The intent was updated (idempotent update, not a second key).
        body = json.loads(
            mem._data[key1]["content"].split("```json\n", 1)[1].rsplit("\n```", 1)[0]
        )
        assert body["intent"] == "second"
        assert body["source"] == "human"
        assert body["invoked_interactive"] is True

    def test_write_adjudication_null_fork(self):
        mem = _make_mem()
        key = pc.write_adjudication(
            mem, target_id="my-target", pr_number=42, outcome="confirm",
            repo="lapis-pm", fork=None, intent="x",
            citations=["router/lapis-pm/ratification-outcomes/eid-001"],
        )
        body = json.loads(
            mem._data[key]["content"].split("```json\n", 1)[1].rsplit("\n```", 1)[0]
        )
        assert body["fork"] is None  # auditable but never matches
