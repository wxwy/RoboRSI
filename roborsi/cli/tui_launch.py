"""Native RoboRSI cockpit launcher and bridge process lifecycle."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TUI_DIR = REPO / "roborsi" / "frontend" / "tui"


def _bridge_up(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/data.json", timeout=0.4):
            return True
    except OSError:
        return False


@contextmanager
def managed_bridge(ready, module: str, arguments: list[str], no_bridge: bool):
    bridge = None
    try:
        if not ready():
            if no_bridge:
                raise SystemExit("[roborsi] No matching bridge is available.")
            bridge = subprocess.Popen([sys.executable, "-m", module, *arguments], cwd=REPO,
                                      stdout=sys.stderr)
            until = time.monotonic() + 12
            while not ready():
                status = bridge.poll()
                if status is not None:
                    raise SystemExit(f"[roborsi] The bridge exited during startup ({status}).")
                if time.monotonic() >= until:
                    raise SystemExit("[roborsi] The bridge did not become ready.")
                time.sleep(0.1)
        yield
    finally:
        if bridge is not None and bridge.poll() is None:
            bridge.terminate()
            try:
                bridge.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bridge.kill()
                bridge.wait()


def launch(host: str = "127.0.0.1", port: int = 8791, no_bridge: bool = False, *,
           serve: bool = False, text: str | None = None, monitor: bool = False,
           watch: bool = False, session: str = "direct", submission_id: str | None = None,
           entity_id: str | None = None) -> None:
    if os.environ.get("ROBORSI_PSM_CONFIG"):
        from roborsi.cli.psm_client import terminal

        terminal(host, port, no_bridge, serve=serve, text=text, monitor=monitor, watch=watch,
                 session=session, submission_id=submission_id, entity_id=entity_id)
        return
    if serve or text is not None or monitor or watch or submission_id is not None or entity_id is not None or session != "direct":
        raise SystemExit("[roborsi] PSM terminal options require ROBORSI_PSM_CONFIG.")
    if shutil.which("node") is None:
        raise SystemExit("[roborsi] The Ink cockpit requires Node.js >=18.")
    if not (TUI_DIR / "node_modules").exists():
        subprocess.run(["npm", "install"], cwd=TUI_DIR, check=True)
    # board.web.server accepts --host and --evo-port; it does not start a second cockpit port.
    with managed_bridge(lambda: _bridge_up(host, port), "roborsi.embodied.board.web.server",
                        ["--host", host, "--evo-port", str(port)], no_bridge):
        subprocess.run(["node", str(TUI_DIR / "bin" / "roborsi-tui.mjs"),
                        "--host", host, "--port", str(port)], cwd=TUI_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="roborsi tui", description="Open the RoboRSI cockpit.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--no-bridge", action="store_true")
    args = parser.parse_args()
    launch(host=args.host, port=args.port, no_bridge=args.no_bridge)


if __name__ == "__main__":
    main()
