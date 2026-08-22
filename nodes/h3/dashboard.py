# APNext H3 Studio dashboard - server side
#
# The dashboard itself is a static page (web/h3_dashboard.html, served by
# ComfyUI at /extensions/comfyui_dagthomas/h3_dashboard.html). This module
# adds the one API it needs beyond ComfyUI's own endpoints:
#
#   GET /apnext/h3/api_workflows -> {"workflows": {name: <api-format prompt>}}
#
# read live from examples/h3/api/*.api.json, so regenerating the exports
# updates the dashboard without touching anything else. The sidebar button
# that opens the page lives in web/js/h3_dashboard_button.js.

import json
import os

try:
    from server import PromptServer
except Exception:
    PromptServer = None
try:
    from aiohttp import web
except Exception:
    web = None

_API_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples", "h3", "api",
)


async def _list_characters(request):
    """The character roster for the dashboard's cast picker."""
    from .characters import _ROWS
    return web.json_response({"characters": [
        {"label": r["label"], "character": r["character"],
         "actor": r["actor"], "franchise": r["franchise"]}
        for r in _ROWS
    ]})


async def _list_api_workflows(request):
    workflows = {}
    try:
        for file_name in sorted(os.listdir(_API_DIR)):
            if not file_name.endswith(".api.json"):
                continue
            try:
                with open(os.path.join(_API_DIR, file_name), encoding="utf-8") as f:
                    workflows[file_name[: -len(".api.json")]] = json.load(f)
            except (OSError, ValueError) as exc:
                print(f"⚠️ H3 Studio: could not read {file_name}: {exc}")
    except OSError:
        pass
    return web.json_response({"workflows": workflows})


if PromptServer is not None and web is not None and getattr(PromptServer, "instance", None) is not None:
    PromptServer.instance.routes.get("/apnext/h3/api_workflows")(_list_api_workflows)
    PromptServer.instance.routes.get("/apnext/h3/characters")(_list_characters)
