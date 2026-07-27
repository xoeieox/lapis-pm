"""Standalone contract shape: declaration -> elaboration -> observed -> delta.

D1 (arc-registry-contract-schema-v0): a contract does not assume its subject
is an arc. It carries a typed SubjectRef by reference. This module has no
import dependency on state_brief.py and never branches on SubjectRef.kind --
a consumer that needs subject-specific behavior does it in its own module.
state_brief.py (and arc_registry.py) import this module, never the reverse.

NEXT IS COMPUTED, not stored: compute_delta() is a module-level pure
function, never a property, so a delta can never be accidentally persisted.
Contract.to_dict() emits exactly subject/declaration/elaboration/observed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubjectRef:
    kind: str  # open string, NOT an enum. "arc" | "simulation-parameters" | ...
    id: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "id": self.id}

    @classmethod
    def from_dict(cls, d: dict) -> "SubjectRef":
        return cls(kind=d["kind"], id=d["id"])


@dataclass(frozen=True)
class Observation:
    source: str          # which exhaust source, e.g. "pm/landed" | "lapis_state"
    key: str              # which elaboration this observation bears on
    value: str | None     # None == observed-but-empty, distinct from not-observed
    ts: str                # iso8601

    def to_dict(self) -> dict:
        return {"source": self.source, "key": self.key, "value": self.value, "ts": self.ts}

    @classmethod
    def from_dict(cls, d: dict) -> "Observation":
        return cls(source=d["source"], key=d["key"], value=d.get("value"), ts=d["ts"])


@dataclass(frozen=True)
class Delta:
    key: str               # the elaboration key that diverged
    declared: str
    observed: str | None
    kind: str               # "contradicted" | "unobserved"


@dataclass(frozen=True)
class Contract:
    subject: SubjectRef
    declaration: str                 # Erah's language: "should do X / must not do Y"
    elaboration: dict[str, str]      # key -> checkable restatement
    observed: tuple[Observation, ...] = ()
    # NOTE: no `delta` field -- see compute_delta() below.

    def to_dict(self) -> dict:
        """Emits exactly subject, declaration, elaboration, observed -- never delta."""
        return {
            "subject": self.subject.to_dict(),
            "declaration": self.declaration,
            "elaboration": dict(self.elaboration),
            "observed": [o.to_dict() for o in self.observed],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Contract":
        return cls(
            subject=SubjectRef.from_dict(d["subject"]),
            declaration=d["declaration"],
            elaboration=dict(d["elaboration"]),
            observed=tuple(Observation.from_dict(o) for o in d.get("observed", ())),
        )


def _consistent(elaboration_text: str, observed_value: str) -> bool:
    """Deterministic, dumb consistency check for v0: case-folded
    substring/equality between the elaboration text and the observation
    value. Under-reports `contradicted` and over-reports `unobserved` on
    paraphrase -- that bias is deliberate (see spec Risks). No LLM."""
    e = elaboration_text.casefold().strip()
    v = observed_value.casefold().strip()
    if not e or not v:
        return e == v
    return e == v or e in v or v in e


def compute_delta(contract: Contract) -> tuple[Delta, ...]:
    """Module-level pure function -- never a property, so it can never be
    accidentally serialized (the schema-level expression of "NEXT IS
    COMPUTED, not stored").

    For each key in contract.elaboration:
    - no Observation carries that key at all -> Delta(kind="unobserved")
    - an Observation exists with value=None -> Delta(kind="contradicted")
      (an empty slot where we *did* look is a lie of omission, not silence)
    - an observation whose value contradicts the elaboration -> "contradicted"
    - an observation consistent with the elaboration -> no delta

    A contract whose elaborations have zero matching observations returns N
    "unobserved" deltas, never an empty tuple -- a zero-delta must never be
    indistinguishable from "we could not see anything".
    """
    latest_by_key: dict[str, Observation] = {}
    for obs in contract.observed:
        latest_by_key[obs.key] = obs

    deltas: list[Delta] = []
    for key, elaboration_text in contract.elaboration.items():
        obs = latest_by_key.get(key)
        if obs is None:
            deltas.append(Delta(key=key, declared=elaboration_text, observed=None, kind="unobserved"))
            continue
        if obs.value is None:
            deltas.append(Delta(key=key, declared=elaboration_text, observed=None, kind="contradicted"))
            continue
        if _consistent(elaboration_text, obs.value):
            continue
        deltas.append(Delta(key=key, declared=elaboration_text, observed=obs.value, kind="contradicted"))

    return tuple(deltas)
