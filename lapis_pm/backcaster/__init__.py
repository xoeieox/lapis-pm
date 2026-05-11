"""Backcaster — inverse operator for goal-state decomposition.

Given a goal-state, decomposes it into preconditions across 6 axes,
analyzes gaps vs current ecosystem state, derives required components
(across 7 category kinds), and synthesizes a roadmap.

Entry point: lapis_pm.backcaster.runner.run_backcaster()
CLI surface: lapis-pm backcaster <goal-file> [flags]
"""
