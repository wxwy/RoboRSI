"""PSM registry backend. W1 owns the world and W2 owns every action."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from importlib import import_module

from roborsi.embodied.agent_loop.env import Backend, BackendUnavailable
from roborsi.embodied.agent_loop.psm_config import PsmConfig

WORLD_FIELDS = ("environment_id", "scene_id", "episode_id")
TASK_FIELDS = ("task_id", "subtask_id", "revision")
RUN_FIELDS = (*WORLD_FIELDS, *TASK_FIELDS, "run_id")


@dataclass(frozen=True)
class Caller:
    """Identity from the trusted host/terminal, kept outside W0 wire records."""

    user_id: str
    conversation_id: str
    environment_id: str
    scene_id: str
    episode_id: str
    capabilities: frozenset[str] = field(default_factory=frozenset, compare=False, repr=False)

    def restrict(self, *allowed: str) -> Caller:
        return replace(self, capabilities=self.capabilities.intersection(allowed))

    @property
    def world(self) -> dict:
        return {name: getattr(self, name) for name in WORLD_FIELDS}


class PsmBackend(Backend):
    def configure(self, config: PsmConfig) -> None:
        module, member = config.host_factory.split(":", 1)
        factory = getattr(import_module(module), member)
        self.host = factory(**config.host_options)

    def current(self) -> dict:
        if not hasattr(self, "host"):
            raise BackendUnavailable("configure the PSM host before opening a Manager session")
        return self.host.current()

    def list_tasks(self) -> list[str]:
        return [self.current()["task_id"]]

    def make_env(self, task, config=None):
        raise BackendUnavailable("PSM cannot create an Env; execution uses the hosted W2 runner")
