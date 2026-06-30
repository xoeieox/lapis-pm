"""Prior-Art Scout brief composer.

Writes /srv/lapis/prior-art-scout/runs/YYYY-MM-DD/brief.md and run.yaml.
Appends top findings to the inertia-vault digest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schema import ScoutItem, ScoutRun

_VAULT_DIGEST = Path("/srv/git/inertia-vault-working/Lapis/Prior-Art-Scout-Digest.md")
_MAX_VAULT_LINES_PER_RUN = 10


def _sorted_yaml(obj: Any) -> str:
    return yaml.dump(obj, default_flow_style=False, sort_keys=True, allow_unicode=True)


def write_brief(run_dir: Path, items: list[ScoutItem], run: ScoutRun) -> None:
    """Write brief.md and run.yaml; append to vault digest."""
    run_dir.mkdir(parents=True, exist_ok=True)
    brief_md = _render_brief(items, run)
    (run_dir / "brief.md").write_text(brief_md)
    (run_dir / "run.yaml").write_text(_sorted_yaml(_run_to_dict(run)))
    _append_vault_digest(items, run)


def _run_to_dict(run: ScoutRun) -> dict:
    d = run.model_dump()
    return d


def _render_brief(items: list[ScoutItem], run: ScoutRun) -> str:
    lines: list[str] = []
    lines.append(f"# Prior-Art Scout Brief - {run.run_date}")
    lines.append("")

    # Coverage header
    p1_ns = "architecture/* + strategy/*"
    lines.append(
        f"Coverage: {p1_ns} in-flight ({run.priority1_count} items, exhaustive"
        + ("; namespaces under snapshot cap" if not run.saturated_namespaces else "")
        + ")."
    )
    lines.append(f"Decision batch: {run.priority2_batch}.")

    # Saturation declaration
    for ns in run.saturated_namespaces:
        lines.append(
            f"SATURATION: {ns}/* SATURATED at snapshot cap ({500} rows); "
            f"the tail beyond the cap is INVISIBLE and unscouted - "
            f"its size is unknown to the scout without an infra fix, so no count is asserted here. "
            f'"full cycle" does NOT mean full mem.db coverage.'
        )

    if run.skipped_hopeless:
        lines.append(f"Skipped (known-hopeless): {run.skipped_hopeless} items.")
    lines.append("Skipped namespaces: progress, project, chain.")

    if run.wall_budget_applied is not None:
        lines.append(f"NOTE: --wall-budget {run.wall_budget_applied} applied; run is capped.")

    if run.model_policy == "allow-escalation":
        policy_detail = "(sonnet deep-read + 122B critic)"
    else:
        policy_detail = "(QUEST read + 122B critic)"
    lines.append(f"Model policy: {run.model_policy} {policy_detail}.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Sourced findings
    sourced = [i for i in items if i.outcome == "sources-found"]
    for si in sourced:
        lines.append(f"## [{si.key}] - {si.lean}")
        lines.append(f"**Summary:** {si.summary}")
        if si.findings:
            lines.append(f"**Findings:** {si.findings}")
        for cit in si.citations:
            url = cit.get("url", "")
            excerpt = cit.get("excerpt", "")
            credibility = cit.get("credibility", "")
            cred_label = f"[credibility: {credibility}]" if credibility else ""
            if excerpt:
                lines.append(f'  > "{excerpt}"')
            lines.append(f"  Source: {url} {cred_label}".rstrip())
        lines.append("")

    # Honest-null log
    null_items = [i for i in items if i.outcome != "sources-found"]
    if null_items:
        lines.append(f"---")
        lines.append("")
        lines.append(f"## honest-null log ({len(null_items)} items)")
        for si in null_items:
            lines.append(f"- {si.key}: {si.outcome or 'unknown'}")
        lines.append("")

    return "\n".join(lines)


def _append_vault_digest(items: list[ScoutItem], run: ScoutRun) -> None:
    sourced = [i for i in items if i.outcome == "sources-found"]
    actionable = [i for i in sourced if i.lean in ("adopt-pattern", "adopt-tool")]
    if not actionable:
        return

    lines: list[str] = [f"\n## {run.run_date} Scout Run"]
    count = 0
    for si in actionable:
        if count >= _MAX_VAULT_LINES_PER_RUN:
            break
        lean_tag = f"[{si.lean}]"
        cit_url = si.citations[0].get("url", "") if si.citations else ""
        lines.append(f"- {lean_tag} **{si.key}** - {si.summary[:120]}")
        if cit_url:
            lines.append(f"  - {cit_url}")
        count += 1

    try:
        if _VAULT_DIGEST.exists():
            existing = _VAULT_DIGEST.read_text()
        else:
            existing = "# Prior-Art Scout Digest\n\nTop adopt-pattern + adopt-tool findings per scout run.\n"
        _VAULT_DIGEST.write_text(existing + "\n".join(lines) + "\n")
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "[prior_art_scout] vault digest append failed (fail-soft): %s", exc
        )
