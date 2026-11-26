"""PSM terminal and monitor used by the native RoboRSI launcher."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from roborsi.embodied.agent_loop.psm_config import LOOPBACK, PsmConfig
from roborsi.embodied.skills import PSM_NAMESPACE


def terminal(host: str, port: int, no_bridge: bool, *, serve: bool, text: str | None,
             monitor: bool, watch: bool, session: str, submission_id: str | None, entity_id: str | None) -> None:
    if host not in LOOPBACK:
        raise SystemExit("The PSM bridge listens on a loopback address.")
    if sum((serve, text is not None, monitor, watch)) > 1:
        raise SystemExit("Select one of --serve, --message, --monitor or --watch.")

    if text is None and (entity_id is not None or submission_id is not None):
        raise SystemExit("--entity-id and --submission-id require --message.")
    from roborsi.channels.agent.psm_bridge import bridge_key
    from roborsi.cli.tui_launch import managed_bridge

    config = PsmConfig.load()
    module = "roborsi.channels.agent.psm_bridge"
    arguments = ["--host", host, "--evo-port", str(port)]
    if serve:
        if no_bridge:
            raise SystemExit("--serve cannot be combined with --no-bridge.")
        os.execv(sys.executable, [sys.executable, "-m", module, *arguments])

    def request(path: str, timeout: int | float, payload: dict | None = None) -> dict:
        headers = {"Authorization": "Bearer " + bridge_key(config)}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            f"http://{host}:{port}{path}", headers=headers,
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as response:
            return json.load(response)

    def ready() -> bool:
        try:
            return request("/healthz", 0.4) == {"backend": PSM_NAMESPACE, "config": config.identity}
        except (OSError, ValueError):
            return False

    def send(message: str) -> None:
        body = {"text": message, "session": session}
        if submission_id is not None:
            body["submission_id"] = submission_id
        if entity_id is not None:
            body["entity_id"] = entity_id
        result = request("/message", 75, body)
        print(json.dumps(result, ensure_ascii=False) if text is not None else
              result.get("reply") or json.dumps(result["record"], ensure_ascii=False), flush=True)

    try:
        with managed_bridge(ready, module, arguments, no_bridge):
            if monitor or watch:
                path = "/data.json?" + urllib.parse.urlencode({"session": session})
                last = None
                while True:
                    data = request(path, 3)
                    if data != last:
                        print(json.dumps(data, ensure_ascii=False), flush=True)
                        last = data
                    if not watch:
                        return
                    time.sleep(1)
            if text is not None:
                send(text)
                return
            print("PSM Frozen · /run <目标> · /status <run_id> · /result <run_id> · /stop <run_id>", flush=True)
            while True:
                try:
                    user = input("PSM> " if sys.stdin.isatty() else "")
                except EOFError:
                    return
                if user in ("/exit", "/quit"):
                    return
                if user.strip():
                    try:
                        send(user)
                    except urllib.error.HTTPError as exc:
                        print(f"PSM request failed (HTTP {exc.code}).", file=sys.stderr, flush=True)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"PSM request failed (HTTP {exc.code}).") from None
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"PSM bridge unavailable ({type(exc).__name__}).") from None
    except KeyboardInterrupt:
        return
