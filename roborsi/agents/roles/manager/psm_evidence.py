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


# The W4 policy constructs its own first projection. The bridge admits only
# these fields, including at nested levels, before passing it to the Provider.
_DETAIL_FIELDS = frozenset("""
world_frame geometry_source rooms room_count object_count robot_room_uid views_published
robot_pose position_world forward_world rows match_count omitted_count id name uid room_uid
island_id kind source_category_id source_instance_id source_part_id category bbox_m bbox
bbox_world min max center center_world source relation target basis distance_m bbox_gap_m
view_id view_ids availability owner_frame image_sha256 pixel_sha256 projection_id
attachment_sha256 projection width height resolution meters_per_pixel m_per_pixel uv
plane right_world up_world origin_world binding_id task_binding_id role_candidates role
entity_id same_label_candidates text label
""".split())


def w4_details(value, depth: int = 0):
    if depth >= 7:
        return None
    if isinstance(value, dict):
        return {key: w4_details(item, depth + 1) for key, item in value.items() if key in _DETAIL_FIELDS}
    if isinstance(value, (list, tuple)):
        return [w4_details(item, depth + 1) for item in value[:16]]
    if isinstance(value, str):
        return value[:256]
    if type(value) in (int, float) and abs(value) < 1e12 and math.isfinite(value):
        return value
    return value if type(value) is bool else None
