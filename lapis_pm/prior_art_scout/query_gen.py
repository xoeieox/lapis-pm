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
