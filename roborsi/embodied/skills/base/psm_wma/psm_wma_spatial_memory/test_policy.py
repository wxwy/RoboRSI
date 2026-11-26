"""Exercise the shipped RoboRSI registry and W4 adapter on a host-bound Mock."""

import json

from g2.online.agent_queries import AgentQueries, ReadScope
from g2.tests.test_agent_queries import AUDIT, FUTURE, SCORE, _publish
from roborsi.embodied.skills import get_ns, run

from roborsi.embodied.skills.base.psm_wma.psm_wma_spatial_memory.policy import (
    resolve_projected_evidence,
)


NAME = "psm_wma_spatial_memory"


def test_discovered_skill_requires_authorized_binding(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORSI_HOME", str(tmp_path))
    assert get_ns(NAME, "psm_wma").name == NAME
    assert run(NAME)["reply"]["outcome"] == "forbidden"

    def denied(*_):
        raise AssertionError("Skill explanation contacted the spatial/runtime host")

    knowledge = AgentQueries(
        scope=None,
        capabilities={"skill.read"},
        current_scope=denied,
        publication_at=denied,
    )
    reply = run(
        NAME, reader=knowledge, operation="describe_skill", parameters={"name": NAME}
    )
    assert reply["reply"]["outcome"] == "ok"
    for operation in (
        "run.submit",
        "scan",
        "navigate",
        "step",
        "world.reset",
        "read_file",
    ):
        assert (
            run(NAME, reader=knowledge, operation=operation)["reply"]["outcome"]
            == "forbidden"
        )


def test_skill_and_workspace_evidence_use_same_projection(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBORSI_HOME", str(tmp_path / "roborsi"))
    scope = ReadScope("env-a", "epoch-a", "ep-a")
    files = _publish(tmp_path, scope)
    reader = AgentQueries(
        scope=scope,
        capabilities={"spatial.read"},
        current_scope=lambda: scope,
        publication_at=lambda _scope, frame: files if frame in {None, 0} else None,
    )
    answer = run(NAME, reader=reader, operation="initial_context")
    assert answer["reply"]["outcome"] == "ok"
    assert all(marker not in json.dumps(answer) for marker in (AUDIT, FUTURE, SCORE))
    projected = answer["reply"]["evidence"][0]
    assert (
        resolve_projected_evidence(reader, projected, operation="initial_context")
        == answer
    )
    raw = files.initial.artifact
    ref = dict(ref=str(raw.path), sha256=raw.sha256)
    assert (
        resolve_projected_evidence(reader, ref, operation="initial_context")["reply"][
            "outcome"
        ]
        == "forbidden"
    )
    assert (
        run(NAME, reader=reader, operation="read_file", parameters=ref)["reply"][
            "outcome"
        ]
        == "forbidden"
    )
