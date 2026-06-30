"""Per-item intent + sub-intent construction from a roadmap snapshot item dict."""
from __future__ import annotations


def item_to_request(item: dict) -> dict:
    """Map a roadmap snapshot item (a dict from artifact["items"]) to a Dowser request dict."""
    summary = item["summary"]
    key = item["key"]
    ns = item["namespace"]
    intent = f"What existing open-source projects, libraries, or papers already solve: {summary}"
    sub_intents = [
        f"Who has built: {summary[:300]}",
        f"Open-source alternatives or prior art for: {summary[:300]}",
        f"Research papers or implementations related to: {key}",
    ]
    return {"intent": intent, "context": f"key: {key}\nnamespace: {ns}", "sub_intents": sub_intents}


def collision_to_request(record: dict) -> dict:
    """Frame a Dowser request for a collision survivor.

    record: dict with keys "idea" (emergent idea text), "hash" (collision_hash).
    """
    idea = record["idea"]
    col_hash = record["hash"]
    intent = (
        f"Has anyone explored, validated, or published prior thinking about this "
        f"speculative direction: {idea}"
    )
    sub_intents = [
        f"Prior work in the direction of: {idea[:300]}",
        f"Research, papers, or projects exploring: {idea[:300]}",
        f"Existing precedent or critique for the concept: {idea[:300]}",
    ]
    return {
        "intent": intent,
        "context": f"seed_type: collision\ncollision_hash: {col_hash}",
        "sub_intents": sub_intents,
    }
