"""Helpers for mapping arbitrary corpus doc IDs onto valid Qdrant point IDs."""

from __future__ import annotations

import uuid
from typing import Any


POINT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "hybrid-vdb/qdrant-point-ids")


def to_qdrant_point_id(raw_id: Any) -> int | str:
    value = str(raw_id)
    if value.isdigit():
        return int(value)

    try:
        return str(uuid.UUID(value))
    except ValueError:
        return str(uuid.uuid5(POINT_ID_NAMESPACE, value))


def external_doc_id(point_id: Any, payload: dict[str, Any] | None) -> str:
    if payload and payload.get("doc_id") is not None:
        return str(payload["doc_id"])
    return str(point_id)
