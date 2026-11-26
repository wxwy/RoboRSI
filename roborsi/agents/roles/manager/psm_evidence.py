"""Public projections of W1/W4 receipts. Keep raw artifacts at the host."""

import hashlib
import json
import math
import re

FIELDS = ("scene_id", "episode_id", "source_sha256", "frame", "timestamp_ns",
          "clock_domain", "snapshot_sha256", "views_sha256")
DIGEST = r"[0-9a-f]{64}"
PUBLIC_ID = r"[A-Za-z0-9][A-Za-z0-9_.\-:/]{0,255}"


def public_id(value) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(PUBLIC_ID, value))


def publication(receipt: dict, candidate: dict) -> dict | None:
    if not isinstance(receipt, dict) or not isinstance(candidate, dict) or any(
        name not in receipt or name not in candidate or receipt[name] != candidate[name]
        for name in FIELDS
    ) or receipt.get("native_scene_id") != candidate.get("native_scene_id"):
        return None
    if any(type(candidate[key]) is not int or candidate[key] < 0 for key in ("frame", "timestamp_ns")):
        return None
    if not all(public_id(candidate[key]) for key in ("scene_id", "episode_id", "clock_domain")):
        return None
    for name in ("source_sha256", "snapshot_sha256", "views_sha256"):
        digest = candidate[name]
        if name == "views_sha256" and digest is None:
            continue
        if not isinstance(digest, str) or not re.fullmatch(DIGEST, digest):
            return None
    result = {key: candidate[key] for key in FIELDS}
    native = candidate.get("native_scene_id")
    if native is not None:
        if not public_id(native):
            return None
        result["native_scene_id"] = native
    return result


def scene_projection(raw, world: dict, frame: int) -> dict | None:
    if not isinstance(raw, dict) or any(raw.get(key) != value for key, value in world.items()):
        return None
    if type(raw.get("frame")) is not int or raw["frame"] != frame:
        return None
    roles = raw.get("entity_roles", {})
    geometry = raw.get("geometry", {})
    if not isinstance(roles, dict) or not isinstance(geometry, dict):
        return None
    value = {
        **world, "frame": frame,
        "entity_roles": {
            role: roles[role] for role in ("target", "reference", "source") if public_id(roles.get(role))
        },
    }
    result = {}
    rooms = geometry.get("room_membership")
    if isinstance(rooms, dict):
        result["room_membership"] = {
            entity: room for entity, room in list(rooms.items())[:16] if public_id(entity) and public_id(room)
        }
    box = geometry.get("bbox_m")
    if isinstance(box, list) and len(box) == 6 and all(
        type(n) in (int, float) and abs(n) < 1e6 and math.isfinite(n) for n in box
    ):
        result["bbox_m"] = box
    value["geometry"] = result
    return value


def projection_sha(value: dict) -> str:
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()
