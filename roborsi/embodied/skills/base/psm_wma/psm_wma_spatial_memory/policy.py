"""RoboRSI Skill entry point for the host's role-scoped read service."""

from collections.abc import Mapping


def _deny(answer):
    return {
        "reply": {
            "kind": "spatial_reply",
            "outcome": "forbidden",
            "publication": None,
            "entity_ids": [],
            "evidence": [],
            "answer": answer,
        },
        "details": {},
    }


def run(
    *, reader=None, operation="scene_overview", parameters=None, include_native=False
):
    """Return the projected JSON for one query with a host-injected reader."""
    if not callable(getattr(reader, "query", None)) or not isinstance(operation, str):
        return _deny("This Skill requires the host's authorized read binding.")
    if parameters is not None and not isinstance(parameters, Mapping):
        return _deny("Expected structured read-only query parameters.")
    return reader.query(operation, **(parameters or {})).for_agent(
        include_native=include_native
    )


def resolve_projected_evidence(
    reader, artifact, /, *, operation, parameters=None, include_native=False
):
    """Host entry for the workspace/evidence reader using the shared read boundary."""
    if not callable(getattr(reader, "resolve_evidence", None)) or not isinstance(
        operation, str
    ):
        return _deny("This evidence read requires the host's authorized binding.")
    if parameters is not None and not isinstance(parameters, Mapping):
        return _deny("Expected structured read-only query parameters.")
    return reader.resolve_evidence(artifact, operation, **(parameters or {})).for_agent(
        include_native=include_native
    )
