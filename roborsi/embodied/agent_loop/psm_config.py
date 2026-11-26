"""Frozen PSM configuration. The action deployment belongs to W2."""

from __future__ import annotations

import hashlib
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from roborsi.agents.roles.role_skills import ROLES
from roborsi.embodied.skills import PSM_NAMESPACE

CONFIG_ENV = "ROBORSI_PSM_CONFIG"
STATE_ENV = "PSM_AGENT_STATE_DIR"
LOOPBACK = {"localhost", "127.0.0.1"}


@dataclass(frozen=True)
class AgentModel:
    kind: str
    model: str
    url: str = ""
    key_env: str = ""

    @classmethod
    def from_config(cls, spec: dict) -> AgentModel:
        model = cls(**spec)
        if not isinstance(model.model, str) or not model.model.strip():
            raise ValueError("a role model identifier is required")
        if model.kind == "openai_compatible":
            url = urlsplit(model.url)
            if (
                url.scheme not in ("http", "https") or not url.hostname
                or url.username or url.password or url.query or url.fragment
                or not url.path.rstrip("/").endswith("/v1")
            ):
                raise ValueError("provide an OpenAI-compatible /v1 base URL without credentials")
            if url.hostname not in LOOPBACK and (url.scheme != "https" or not model.key_env):
                raise ValueError("remote providers require HTTPS and a credential environment variable")
            if model.key_env and not re.fullmatch(r"[A-Z][A-Z0-9_]*", model.key_env):
                raise ValueError("invalid credential environment variable name")
        elif model.kind != "mock" or model.url or model.key_env:
            raise ValueError("unknown or invalid Agent provider configuration")
        return model


@dataclass(frozen=True)
class PsmConfig:
    state: Path
    host_factory: str
    host_options: dict
    models: dict[str, AgentModel]
    coding_backend: str
    identity: str

    @classmethod
    def load(cls) -> PsmConfig:
        path = Path(os.environ[CONFIG_ENV]).resolve()
        raw = path.read_bytes()
        config = tomllib.loads(raw.decode("utf-8"))
        root = config["psm"]
        if root["backend"] != PSM_NAMESPACE or root["mode"] != "frozen":
            raise ValueError("W3 requires the psm_wma backend and frozen mode")
        state = Path(os.environ.get(STATE_ENV, root["state_dir"]))
        if not state.is_absolute():
            state = path.parent / state
        state = state.resolve()
        models = {name: AgentModel.from_config(value) for name, value in config["providers"].items()}
        selection = config["models"]
        if selection["mode"] == "shared" and set(selection) == {"mode", "shared"}:
            roles = dict.fromkeys(ROLES, models[selection["shared"]])
        elif selection["mode"] == "per_role" and set(selection) == {"mode", *ROLES}:
            roles = {role: models[selection[role]] for role in ROLES}
        else:
            raise ValueError("configure one shared model or explicitly configure every role")
        coding = config["coding"]["backend"]
        if coding not in {"disabled", "claude", "codex", "copilot"}:
            raise ValueError("unknown RSI coding backend")
        identity = hashlib.sha256(raw + b"\0" + os.fsencode(path) + b"\0" + os.fsencode(state)).hexdigest()
        return cls(state, root["host_factory"], config.get("host", {}), roles, coding, identity)
