"""Local, authenticated PSM bridge for the native CLI."""

import argparse
import asyncio
import hmac
import os
import secrets
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, StringConstraints

from roborsi.agents.roles import manager_chat
from roborsi.agents.roles.manager.psm import PsmManager, Route, route
from roborsi.agents.sessions.psm import ProviderError, ReadAuthorizationExpired
from roborsi.embodied.agent_loop import get_backend
from roborsi.embodied.agent_loop.psm_config import LOOPBACK, PsmConfig
from roborsi.embodied.skills import PSM_NAMESPACE

Identifier = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
Utterance = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4096)]
_READ_ROUTES = {"skill", "space", "run_qa"}
_REPLIES = {
    "clarify_target": "请指定当前场景中已确认的单一任务和对象。",
    "clarify_run": "请提供准确的 run ID。",
    "negated": "收到，保持当前执行状态。",
    "compound": "请一次指定一个目标。",
    "unknown_command": "可用命令为 /run、/status、/result 和 /stop。",
}


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Utterance
    session: Identifier = "direct"
    submission_id: Identifier | None = None
    entity_id: Identifier | None = None


def _read_key(path: Path) -> str:
    key = path.read_text(encoding="ascii")
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("invalid bridge key file")
    return key


def bridge_key(config: PsmConfig, *, create: bool = False) -> str:
    path = config.state / ("bridge-" + config.identity + ".key")
    if not create:
        return _read_key(path)

    # Publish a full key atomically; concurrent bridge probes must not read a partial file.
    config.state.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = config.state / ("key-" + secrets.token_hex(16))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as output:
        output.write(secrets.token_hex(32))
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        return _read_key(path)
    finally:
        temporary.unlink()


def create_app(config: PsmConfig) -> FastAPI:
    backend = get_backend(PSM_NAMESPACE)
    backend.configure(config)
    manager = PsmManager(backend, config)
    key = bridge_key(config, create=True)

    @asynccontextmanager
    async def lifespan(app):
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="psm-qa") as executor:
            app.state.qa_workers = executor
            try:
                yield
            finally:
                close = getattr(backend.host, "close", None)
                if close:
                    close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        if not hmac.compare_digest((authorization or "").encode(), ("Bearer " + key).encode()):
            raise HTTPException(status_code=401, detail="Invalid local bridge credentials")

    secured = [Depends(authorize)]

    @app.get("/healthz", dependencies=secured)
    def health() -> dict:
        return {"backend": PSM_NAMESPACE, "config": config.identity}

    @app.get("/data.json", dependencies=secured)
    def monitor(session: Identifier = "direct") -> dict:
        caller, current = manager.context(session)
        return manager.monitor(caller, current)

    @app.get("/runs/{run_id}/status", dependencies=secured)
    def status(run_id: Identifier, session: Identifier = "direct") -> dict:
        return manager.operate(Route("status", run_id), *manager.context(session))

    @app.get("/runs/{run_id}/result", dependencies=secured)
    def result(run_id: Identifier, session: Identifier = "direct") -> dict:
        return manager.operate(Route("result", run_id), *manager.context(session))

    @app.post("/runs/{run_id}/stop", dependencies=secured)
    def stop(run_id: Identifier, session: Identifier = "direct") -> dict:
        return manager.operate(Route("stop", run_id), *manager.context(session))

    @app.get("/evidence/{digest}", dependencies=secured)
    async def evidence(digest: Identifier, session: Identifier = "direct") -> dict:
        caller, current = await run_in_threadpool(manager.context, session)
        value = await run_in_threadpool(manager.evidence, caller, current, digest)
        if value is None or not manager._read_authorization_is_current(caller):
            raise HTTPException(status_code=404, detail="Evidence is not available")
        return value

    @app.post("/message", dependencies=secured)
    async def message(body: Message) -> dict:
        try:
            caller, current = await run_in_threadpool(manager.context, body.session)
            action = route(body.text, current)
            if action.action != "run" and (body.entity_id is not None or body.submission_id is not None):
                raise HTTPException(status_code=400, detail="Entity and submission IDs require an execution request")
            if action.action in _READ_ROUTES:
                # QA workers can wait on the Provider without starving the control pool.
                read = partial(manager_chat.reply, body.text, body.session,
                               scoped=(manager, action, caller, current))
                answer = await asyncio.get_running_loop().run_in_executor(app.state.qa_workers, read)
                if not manager._read_authorization_is_current(caller):
                    raise ReadAuthorizationExpired("Agent read authorization expired")
                return {"route": action.action, "reply": answer}
            if action.action in _REPLIES:
                return {"route": action.action, "reply": _REPLIES[action.action]}
            response = await run_in_threadpool(manager.operate, action, caller, current, body.submission_id, body.entity_id)
            return {"route": action.action, "record": response}
        except HTTPException:
            raise
        except ReadAuthorizationExpired as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from None
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from None
        except Exception:
            raise HTTPException(status_code=503, detail="PSM host request failed") from None

    @app.exception_handler(Exception)
    async def host_failure(_request, _error):
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": "PSM host request failed"}, status_code=503)

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=sorted(LOOPBACK), default="127.0.0.1")
    parser.add_argument("--evo-port", type=int, default=8791)
    args = parser.parse_args()
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    uvicorn.run(create_app(PsmConfig.load()), host=args.host, port=args.evo_port, access_log=False)


if __name__ == "__main__":
    main()
