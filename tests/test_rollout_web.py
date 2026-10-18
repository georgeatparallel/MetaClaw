"""Exercise the packaged rollout without importing optional training dependencies.

The model is scripted; the real MCP SDK serializes requests over MockTransport.
No live model, search service, OpenClaw CLI, or training backend is required.
"""

import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "rollout_under_test", Path(__file__).resolve().parents[1] / "metaclaw" / "openclaw_env_rollout.py",
)
rollout = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rollout)
_CLIENT = httpx.AsyncClient
_URL = "https://docs.openclaw.ai/cli?section=agents#add"


def tool(name, **arguments):
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"


@pytest.fixture
def harness(monkeypatch):
    """Route the actual model and MCP clients through an observable HTTP boundary."""
    model_requests, web_requests = [], []
    replies = []
    behavior = {"mode": "success"}

    async def handler(request):
        payload = json.loads(request.content) if request.content else {}
        if request.url.host == "proxy.test":
            model_requests.append(payload)
            content = replies.pop(0) if replies else "Session closed."
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        web_requests.append(request)
        assert str(request.url) == "https://search.parallel.ai/mcp"
        assert request.headers["User-Agent"] == "MetaClaw (https://github.com/aiming-lab/MetaClaw)"
        assert "authorization" not in request.headers
        if behavior["mode"] == "http_error":
            return httpx.Response(503)
        if behavior["mode"] == "timeout":
            await asyncio.Event().wait()
        if behavior["mode"] == "cancel":
            await asyncio.Event().wait()
        if request.method != "POST":
            return httpx.Response(405)
        method = payload["method"]
        if "id" not in payload:
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": payload["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-search", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": [
                {"name": name, "inputSchema": {"type": "object"}}
                for name in (["web_fetch"] if behavior["mode"] == "missing" else ["web_search", "web_fetch"])
            ]}
        else:
            assert method == "tools/call"
            result = {
                "content": [{"type": "text", "text": f"OpenClaw documentation: {_URL}"}],
                "isError": behavior["mode"] == "tool_error",
            }
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    class Client(_CLIENT):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(rollout.httpx, "AsyncClient", Client)
    monkeypatch.setattr(rollout, "_get_rollout_system_prompt", lambda: "CLI agent instructions.")
    # Anonymous requests must remain anonymous even when credentials exist in the environment.
    for key in ["PARALLEL_API_KEY", "OPENAI_API_KEY", "TINKER_API_KEY"]:
        monkeypatch.setenv(key, "must-not-be-forwarded")
    return model_requests, web_requests, replies, behavior


def episode(**kwargs):
    return asyncio.run(rollout.run_task_episode(
        task_id="docs", task_instruction="Consult current CLI documentation", proxy_url="http://proxy.test",
        model_id="configured-model", max_steps=4, **kwargs,
    ))


def test_default_search_fetch_and_cli_results_are_consumed(harness, monkeypatch):
    model, web, replies, _ = harness
    replies.extend([
        tool("web_search", objective="OpenClaw CLI reference", search_queries=["OpenClaw agents CLI reference"]),
        tool("web_fetch", urls=[_URL], objective="Read agent setup"),
        tool("run_command", command="openclaw status"),
        tool("run_command", command="done"),
    ])
    commands = []

    async def run_command(command):
        commands.append(command)
        return "Gateway ready."

    monkeypatch.setattr(rollout, "_exec_command", run_command)
    episode()
    assert {t["function"]["name"] for t in model[0]["tools"]} == {"run_command", "web_search", "web_fetch"}
    assert "untrusted reference" in model[0]["messages"][0]["content"]
    assert _URL in model[1]["messages"][-1]["content"]
    assert _URL in model[2]["messages"][-1]["content"]
    assert "Gateway ready." in model[3]["messages"][-1]["content"]
    assert commands == ["openclaw status"]
    calls = [json.loads(r.content) for r in web if r.content]
    assert [c["method"] for c in calls].count("tools/list") == 2
    args = [c["params"]["arguments"] for c in calls if c["method"] == "tools/call"]
    assert args[0]["search_queries"] == ["OpenClaw agents CLI reference"]
    assert args[1]["urls"] == [_URL]
    assert args[0]["model_name"] == args[1]["model_name"] == "configured-model"
    assert args[0]["session_id"] == args[1]["session_id"]
    assert len(args[0]["session_id"]) == 32


def test_explicit_disable_hides_tools_and_refuses_model_request(harness):
    model, web, replies, _ = harness
    replies.extend([tool("web_search", objective="docs", search_queries=["docs"]), tool("run_command", command="done")])
    episode(web_enabled=False)
    assert all([t["function"]["name"] for t in request["tools"]] == ["run_command"] for request in model)
    assert "disabled" in model[1]["messages"][-1]["content"]
    assert not web


@pytest.mark.parametrize("name, arguments", [
    ("web_search", {"objective": "docs", "search_queries": []}),
    ("web_search", {"objective": 1, "search_queries": ["docs"]}),
    ("web_search", {"objective": "docs", "search_queries": ["docs"], "session_id": "spoofed"}),
    ("web_fetch", {"urls": ["file:///etc/passwd"]}),
    ("web_fetch", {"urls": ["https://[bad"]}),
    ("web_fetch", {"urls": [_URL], "objective": "x" * 201}),
    ("unknown", {"command": "touch should-not-run"}),
])
def test_bad_arguments_and_unknown_tools_never_execute(harness, monkeypatch, name, arguments):
    model, web, replies, _ = harness

    async def forbidden(*args):
        pytest.fail("invalid tool call reached shell")

    monkeypatch.setattr(rollout, "_exec_command", forbidden)
    replies.extend([tool(name, **arguments), tool("run_command", command="done")])
    episode()
    assert model[1]["messages"][-1]["content"].startswith("Output:\nError:")
    assert not web


@pytest.mark.parametrize("mode, expected", [
    ("http_error", "request failed"), ("tool_error", "Error: Parallel MCP:"),
    ("missing", "does not offer web_search"), ("timeout", "timed out"),
])
def test_web_failures_become_model_observations(harness, monkeypatch, mode, expected):
    model, _, replies, behavior = harness
    behavior["mode"] = mode
    monkeypatch.setattr(rollout, "_WEB_TIMEOUT", 0.05 if mode == "timeout" else 5)
    replies.extend([tool("web_search", objective="docs", search_queries=["docs"]), tool("run_command", command="done")])
    episode()
    assert expected in model[1]["messages"][-1]["content"]


def test_web_cancellation_stops_episode(harness):
    model, web, replies, behavior = harness
    behavior["mode"] = "cancel"
    replies.append(tool("web_search", objective="docs", search_queries=["docs"]))
    async def cancel_episode():
        task = asyncio.create_task(rollout.run_task_episode("cancel", "Read docs", "http://proxy.test"))
        while not web:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_episode())
    assert len(model) == 1


def test_rollout_loop_forwards_disable_and_can_be_cancelled(tmp_path, monkeypatch):
    (tmp_path / "train.jsonl").write_text('{"task_id":"one","instruction":"Check status"}\n')
    observed = []

    async def run_episode(**kwargs):
        observed.append(kwargs)
        raise asyncio.CancelledError()

    monkeypatch.setattr(rollout, "run_task_episode", run_episode)

    async def run_loop():
        task = asyncio.create_task(rollout.rollout_loop(
            "http://proxy.test", str(tmp_path), concurrency=1, web_enabled=False,
        ))
        while not observed:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_loop())
    assert observed[0]["web_enabled"] is False


@pytest.mark.parametrize("payload", [
    "[]", "null", '{"name":[],"arguments":{}}', '{"name":"web_search","arguments":null}',
])
def test_malformed_tool_call_is_ignored(payload):
    assert rollout._parse_tool_call(f"<tool_call>{payload}</tool_call>") is None
