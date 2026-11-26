"""Scoped API turns for Frozen roles, driven by persistent_agent.run_role."""

from __future__ import annotations

import asyncio
import base64
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from roborsi.embodied.agent_loop.psm_config import AgentModel
from roborsi.utils.helpers import detect_image_mime


class ProviderError(RuntimeError):
    """The bridge reports the failure without serializing a provider response."""


class ReadAuthorizationExpired(PermissionError):
    """The Host invalidated the read snapshot while a turn was in progress."""


@dataclass(frozen=True)
class ApprovedImage:
    mime: str
    content: bytes
    sha256: str

    def data_url(self) -> str:
        if (
            self.mime not in ("image/png", "image/jpeg")
            or not isinstance(self.content, bytes) or len(self.content) > 2_000_000
            or detect_image_mime(self.content) != self.mime
            or hashlib.sha256(self.content).hexdigest() != self.sha256
        ):
            raise ProviderError("Invalid approved image")
        return f"data:{self.mime};base64,{base64.b64encode(self.content).decode('ascii')}"


@dataclass(frozen=True)
class ReadTool:
    spec: dict
    read: Callable[[dict], tuple[dict, ApprovedImage | None]]


@dataclass
class ScopedRole:
    role: str
    workspace: Path
    model: AgentModel
    tools: dict[str, ReadTool]
    context: dict
    mock_provider: Callable | None = None
    is_authorized: Callable[[], bool] | None = None

    def _require_authorization(self) -> None:
        if self.is_authorized is not None and not self.is_authorized():
            raise ReadAuthorizationExpired("Agent read authorization expired")

    def turn(self, system: str, user: str) -> str:
        from roborsi.agents.sessions.persistent_agent import _read_json, _write_json

        self.workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
        history = self.workspace / "history.json"
        with (self.workspace / "turn.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._require_authorization()
            turns = _read_json(history).get("turns", [])
            initial_history = "same_episode" if turns else "empty"
            messages = [{"role": "system", "content": system}, *turns, {"role": "user", "content": user}]
            answer = asyncio.run(self._complete(messages))
            self._require_authorization()
            # Tool payloads, raw evidence, images, and provider reasoning never seed another turn.
            turns += [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
            _write_json(history, {"turns": turns[-12:]})
            _write_json(self.workspace / "context.json", {
                **self.context, "initial_history": initial_history, "history_ref": "history.json",
            })
        self._require_authorization()
        return answer

    async def _complete(self, messages: list[dict]) -> str:
        if self.model.kind == "mock":
            if self.mock_provider is None:
                raise ProviderError("The host does not provide a CPU mock model")
            return await self._chat(self.mock_provider(self.role), messages)

        from roborsi.providers.custom_provider import CustomProvider

        key = os.environ.get(self.model.key_env) if self.model.key_env else "local-agent"
        if not key:
            raise ProviderError("The configured Agent provider credential is missing")
        provider = CustomProvider(api_key=key, api_base=self.model.url, default_model=self.model.model)
        provider._client = provider._client.with_options(timeout=30, max_retries=0)
        try:
            return await self._chat(provider, messages)
        finally:
            await provider._client.close()

    async def _chat(self, provider, messages: list[dict]) -> str:
        specs = [{"type": "function", "function": tool.spec} for tool in self.tools.values()]
        self._require_authorization()
        response = await provider.chat(messages, tools=specs, model=self.model.model, temperature=0.0)
        self._require_authorization()
        if response.finish_reason == "error":
            raise ProviderError("Agent provider request failed")
        if response.tool_calls:
            # One bounded round of approved reads. Never execute a partially valid batch.
            calls = response.tool_calls
            if len(calls) > 3 or len({call.id for call in calls}) != len(calls) or any(
                call.name not in self.tools or not isinstance(call.arguments, dict)
                for call in calls
            ):
                return "当前问答没有请求工具的权限。"
            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [call.to_openai_tool_call() for call in calls],
            })
            images = []
            for call in calls:
                self._require_authorization()
                data, image = self.tools[call.name].read(call.arguments)
                self._require_authorization()
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": json.dumps(data, ensure_ascii=False),
                })
                if image is not None:
                    images.append({"type": "image_url", "image_url": {"url": image.data_url()}})
            if images:
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": "Approved views for the cited committed publication."},
                    *images,
                ]})
            self._require_authorization()
            response = await provider.chat(messages, model=self.model.model, temperature=0.0)
        if response.tool_calls or response.finish_reason != "stop" or not response.content:
            raise ProviderError("Agent provider returned an incomplete answer")
        return response.content[:8_000]
