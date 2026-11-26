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


# C4 has already projected the source. Apply a second allowlist for the
# specific operation, including the shape of each nested map and array.
_NUMBER = (int, float)
_VECTOR = [_NUMBER]
_ROOM = {"id": str, "uid": int, "name": str, "center_world": _VECTOR}
_ENTITY = {
    "id": str,
    "kind": str,
    "label": str,
    "source_instance_id": str,
    "source_part_id": str,
    "source_category_id": str,
    "room_uid": int,
    "island_id": str,
    "bbox_min": _VECTOR,
    "bbox_max": _VECTOR,
}
_ROW_COUNTS = {"match_count": int, "omitted_count": int}
_DETAIL_FIELDS = {
    "scene_overview": {
        "world_frame": str,
        "geometry_source": str,
        "rooms": [_ROOM],
        "room_count": int,
        "object_count": int,
        "robot_room_uid": int,
        "views_published": bool,
        "robot_pose": {"position_world": _VECTOR, "forward_world": _VECTOR},
    },
    "rooms": {"rows": [_ROOM], **_ROW_COUNTS},
    "entities": {"rows": [_ENTITY], **_ROW_COUNTS},
    "relation": {
        "source": str,
        "relation": str,
        "target": str,
        "basis": str,
        "bbox_gap_m": _NUMBER,
    },
    "list_views": {"view_ids": [str], "basis": str, **_ROW_COUNTS},
    "initial_context": {
        "basis": str,
        "role_candidates": [{
            "entity_id": str,
            "name": str,
            "role": str,
            "room_uid": int,
            "same_label_candidates": int,
        }],
        "omitted_candidates": int,
    },
    "view": {
        "view_id": str,
        "canonical_view_id": str,
        "variant": str,
        "is_floor_top": bool,
        "image_sha256": str,
        "pixel_sha256": str,
        "attachment_sha256": str,
        "owner_frame": int,
        "membership_version": str,
        "room_uid": int,
        "island_id": str,
        "projection": {
            "projection_id": str,
            "kind": str,
            "axis": str,
            "units": str,
            "right_world": _VECTOR,
            "up_world": _VECTOR,
            "forward_world": _VECTOR,
            "width_px": _NUMBER,
            "height_px": _NUMBER,
            "meters_per_pixel": _NUMBER,
            "pixel_center_offset": _NUMBER,
            "uv_min_room": _VECTOR,
            "rows_flipped": bool,
            "reference_frame": {"origin_world": _VECTOR, "axes_world": [_VECTOR]},
        },
    },
}


def _project(value, fields):
    if isinstance(fields, dict):
        if not isinstance(value, dict):
            return None
        return {key: _project(value[key], rule) for key, rule in fields.items() if key in value}
    if isinstance(fields, list):
        if not isinstance(value, (list, tuple)):
            return None
        return [_project(item, fields[0]) for item in value[:16]]
    types = fields if isinstance(fields, tuple) else (fields,)
    if type(value) not in types:
        return None
    if type(value) is str:
        return value[:256]
    if type(value) in (int, float) and not (-1e12 < value < 1e12):
        return None
    return value


def w4_details(operation: str, value) -> dict | None:
    fields = _DETAIL_FIELDS.get(operation)
    return _project(value, fields) if fields is not None else None
