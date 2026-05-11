# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic", "rich", "prompt_toolkit"]
# ///
"""
Python coding agent.

## Run

    export ANTHROPIC_API_KEY=...
    export ANTHROPIC_BASE_URL=http://...
    uv run agent.py "what is rust?"
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from anthropic import AsyncAnthropic
from rich.markup import escape


class Merge(Enum):
    REPLACE = "replace"
    ACCUMULATE = "accumulate"
    BLOCK = "block"
    CHAIN = "chain"


@dataclass(frozen=True)
class Return:
    key: str
    kind: Merge = Merge.REPLACE


BLOCK = Return("block", kind=Merge.BLOCK)
REASON = Return("reason")
INPUT = Return("input", kind=Merge.CHAIN)
CONTENT = Return("content", kind=Merge.CHAIN)
IS_ERROR = Return("is_error")
SYSTEM_PROMPT = Return("system_prompt")
PATH = Return("path")
ADDITIONAL_CONTEXT = Return("additional_context", kind=Merge.ACCUMULATE)
SYSTEM = Return("system", kind=Merge.CHAIN)
TOOLS = Return("tools", kind=Merge.CHAIN)
MESSAGES = Return("messages", kind=Merge.CHAIN)
MODEL = Return("model")
MAX_TURNS = Return("max_turns")
CLIENT = Return("client")


@dataclass(frozen=True)
class Event:
    name: str
    returns: tuple[Return, ...] = ()


class HookAPI:
    def __init__(self, runner: "HookRunner") -> None:
        self._runner = runner

    def register_event(self, name: str, *returns: Return) -> None:
        if name in self._runner.events:
            raise ValueError(f"event {name!r} already registered")
        self._runner.events[name] = Event(name, returns)
        self._runner.handlers[name] = []

    def on(self, event: str, handler: Callable[[dict, dict], Any]) -> None:
        if event not in self._runner.events:
            raise ValueError(
                f"unknown event {event!r}. Known: {', '.join(self._runner.events) or '(none yet)'}"
            )
        self._runner.handlers[event].append(handler)

    def register_tool(self, tool: "Tool") -> None:
        if any(t.name == tool.name for t in self._runner.tools):
            raise ValueError(f"tool {tool.name!r} already registered")
        self._runner.tools.append(tool)

    def register_prompter(self, fn: Callable[..., Any]) -> None:
        """Register the async callable that reads the initial user prompt."""
        self._runner.prompter = fn

    def register_history_loader(self, fn: Callable[[], list[str]]) -> None:
        """Register a callable that returns prior prompts for up/down navigation."""
        self._runner.history_loader = fn

    def register_flag(self, *args: Any, **kwargs: Any) -> None:
        self._runner.parser.add_argument(*args, **kwargs)

    @property
    def runner(self) -> "HookRunner":
        return self._runner


class HookRunner:
    def __init__(self) -> None:
        self.events: dict[str, Event] = {}
        self.handlers: dict[str, list[Callable[[dict, dict], Any]]] = {}
        self.tools: list[Tool] = []
        self.prompter: Callable[..., Any] | None = None
        self.history_loader: Callable[[], list[str]] | None = None
        self.parser = argparse.ArgumentParser(
            prog="agent.py", description="Single-file Python coding agent."
        )
        self.strict: bool = False
        self.api = HookAPI(self)
        self.load(lifecycle_hook)

    def load(self, hook: Callable[[HookAPI], None]) -> None:
        try:
            hook(self.api)
        except Exception as e:
            print(f"[hook {hook.__name__} failed to load] {e}", file=sys.stderr)

    def parse_args(self) -> argparse.Namespace:
        return self.parser.parse_args()

    def fire(self, event: str, payload: dict, ctx: dict | None = None) -> dict:
        result: dict = {}
        ctx = ctx or {}
        payload = {**payload}  # handlers see chained values; don't mutate caller's dict
        for handler in self.handlers[event]:
            for ret in self.events[event].returns:
                if ret.kind is Merge.CHAIN and ret.key in result:
                    payload[ret.key] = result[ret.key]
            try:
                r = handler(payload, ctx) or {}
            except Exception as e:
                if self.strict:
                    raise
                print(f"[hook error on {event}] {e}", file=sys.stderr)
                continue
            if not isinstance(r, dict) or not r:
                continue
            for ret in self.events[event].returns:
                if ret.key not in r:
                    continue
                if ret.kind is Merge.BLOCK and r[ret.key]:
                    return {
                        **result,
                        ret.key: r[ret.key],
                        "reason": r.get("reason", ""),
                    }
                if ret.kind is Merge.ACCUMULATE:
                    result.setdefault(ret.key, []).append(r[ret.key])
                else:
                    result[ret.key] = r[ret.key]
        return result

    def describe(self) -> str:
        return "\n".join(
            f"  {name:<22}  {len(self.handlers[name])} handler(s)"
            for name in self.events
        )


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    execute: Callable[[dict], tuple[str, bool]]
    # Optional co-located display. Both receive a per-call `state` dict
    # that carries data from call → result and is visible to other hooks.
    render_call: Callable[["Tool", dict, Any, dict], None] | None = None
    render_result: Callable[["Tool", dict, str, bool, Any, dict], None] | None = None

    def to_anthropic(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.schema,
        }


def _read_tool() -> Tool:
    def execute(args: dict) -> tuple[str, bool]:
        try:
            return Path(args["path"]).read_text(), False
        except Exception as e:
            return f"{type(e).__name__}: {e}", True

    return Tool(
        name="read",
        description="Read a file from disk. Returns the full text.",
        schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        execute=execute,
    )


def _write_tool() -> Tool:
    def execute(args: dict) -> tuple[str, bool]:
        try:
            Path(args["path"]).write_text(args["content"])
            return f"wrote {len(args['content'])} bytes to {args['path']}", False
        except Exception as e:
            return f"{type(e).__name__}: {e}", True

    return Tool(
        name="write",
        description="Overwrite a file with the given content. Creates it if missing.",
        schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        execute=execute,
    )


def _edit_tool() -> Tool:
    def execute(args: dict) -> tuple[str, bool]:
        try:
            path = Path(args["path"])
            text = path.read_text()
            occurrences = text.count(args["old"])
            if occurrences != 1:
                return (
                    f"edit failed: `old` must appear exactly once (found {occurrences})",
                    True,
                )
            path.write_text(text.replace(args["old"], args["new"]))
            return f"edited {args['path']}", False
        except Exception as e:
            return f"{type(e).__name__}: {e}", True

    def render_call(_tool: "Tool", args: dict, console: Any, state: dict) -> None:
        console.print(f"\n⏺ [bold]Update[/bold]({args.get('path', '')})")
        try:
            state["pre"] = Path(args["path"]).read_text()
        except OSError:
            pass

    def render_result(
        tool: "Tool",
        args: dict,
        content: str,
        is_error: bool,
        console: Any,
        state: dict,
    ) -> None:
        if is_error:
            _default_render_result(tool, args, content, is_error, console, state)
            return
        try:
            after = Path(args["path"]).read_text()
        except OSError:
            return
        _render_diff(console, state.get("pre", ""), after)

    return Tool(
        name="edit",
        description=(
            "Replace `old` with `new` in a file. `old` must appear exactly once. "
            "When deleting a whole line, include its trailing newline in `old` "
            "(e.g. old='foo\\n', new='') — otherwise a blank line is left behind."
        ),
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
            },
            "required": ["path", "old", "new"],
        },
        execute=execute,
        render_call=render_call,
        render_result=render_result,
    )


def _bash_tool() -> Tool:
    def execute(args: dict) -> tuple[str, bool]:
        try:
            r = subprocess.run(
                args["cmd"], shell=True, capture_output=True, text=True, timeout=60
            )
            return (r.stdout + r.stderr)[:20_000] or "(no output)", r.returncode != 0
        except subprocess.TimeoutExpired:
            return "bash timed out after 60s", True
        except Exception as e:
            return f"{type(e).__name__}: {e}", True

    return Tool(
        name="bash",
        description="Run a shell command. Returns combined stdout+stderr.",
        schema={
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
        execute=execute,
    )


@dataclass
class SessionEntry:
    id: str
    parent_id: str | None
    role: str
    content: Any


class SessionManager:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: list[SessionEntry] = []
        self._last_id: str | None = None
        if path.exists():
            self._load()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._write_header()

    def _write_header(self) -> None:
        header = {
            "type": "header",
            "version": 1,
            "id": uuid.uuid4().hex,
            "cwd": os.getcwd(),
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        self.path.write_text(json.dumps(header) + "\n")

    def _load(self) -> None:
        for line in self.path.read_text().splitlines():
            obj = json.loads(line)
            if obj.get("type") == "entry":
                e = SessionEntry(
                    obj["id"], obj["parentId"], obj["role"], obj["content"]
                )
                self.entries.append(e)
                self._last_id = e.id

    def append(self, role: str, content: Any) -> SessionEntry:
        entry = SessionEntry(uuid.uuid4().hex, self._last_id, role, content)
        self.entries.append(entry)
        self._last_id = entry.id
        with self.path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "type": "entry",
                        "id": entry.id,
                        "parentId": entry.parent_id,
                        "role": entry.role,
                        "content": entry.content,
                    },
                    default=str,
                )
                + "\n"
            )
        return entry

    def to_messages(self) -> list[dict]:
        out: list[dict] = []
        for e in self.entries:
            if e.role == "tool_result":
                out.append({"role": "user", "content": e.content})
            else:
                out.append({"role": e.role, "content": e.content})
        return out


async def agent_loop(
    client: AsyncAnthropic,
    model: str,
    cwd: str,
    tools: list[Tool],
    session: SessionManager,
    runner: HookRunner,
    pending_reminders: list[str],
    max_turns: int = 25,
) -> None:
    tool_map = {t.name: t for t in tools}
    schemas = [t.to_anthropic() for t in tools]
    ctx = {"cwd": cwd, "session": session}

    for _ in range(max_turns):
        runner.fire("turn_start", {}, ctx)

        for reminder in pending_reminders:
            session.append("user", reminder)
        pending_reminders.clear()

        result = runner.fire("build_system_prompt", {"cwd": cwd}, ctx)
        system = result.get("system_prompt", "")
        for extra in result.get("additional_context", []):
            system += f"\n\n{extra}"

        payload = {
            "system": system,
            "tools": schemas,
            "messages": session.to_messages(),
        }
        override = runner.fire("before_model_request", payload, ctx)
        final = {
            "system": override.get("system", payload["system"]),
            "tools": override.get("tools", payload["tools"]),
            "messages": override.get("messages", payload["messages"]),
        }
        runner.fire("model_request_prepared", final, ctx)

        async with client.messages.stream(
            model=model,
            max_tokens=4096,
            system=final["system"],
            tools=final["tools"],
            messages=final["messages"],
        ) as stream:
            had_text = False
            async for text in stream.text_stream:
                runner.fire("text_delta", {"text": text}, ctx)
                had_text = True
            if had_text:
                runner.fire("text_end", {}, ctx)
            res = await stream.get_final_message()

        dumped = res.model_dump(mode="json")
        assistant_content = dumped["content"]
        session.append("assistant", assistant_content)
        runner.fire(
            "message_end",
            {"message": assistant_content, "usage": res.usage.model_dump()},
            ctx,
        )

        tool_uses = [b for b in res.content if b.type == "tool_use"]
        if not tool_uses:
            runner.fire("stop", {}, ctx)
            return

        tool_results = []
        for tu in tool_uses:
            tool_state: dict = {}
            pre = runner.fire(
                "pre_tool_use",
                {
                    "id": tu.id,
                    "name": tu.name,
                    "input": tu.input,
                    "state": tool_state,
                },
                ctx,
            )
            pending_reminders.extend(pre.get("additional_context", []))

            effective_input = pre.get("input", tu.input)
            if pre.get("block"):
                content, is_error = pre.get("reason", "blocked by hook"), True
            else:
                tool = tool_map.get(tu.name)
                if tool is None:
                    content, is_error = f"unknown tool: {tu.name}", True
                else:
                    content, is_error = tool.execute(effective_input)

            post = runner.fire(
                "post_tool_use",
                {
                    "id": tu.id,
                    "name": tu.name,
                    "input": effective_input,
                    "content": content,
                    "is_error": is_error,
                    "state": tool_state,
                },
                ctx,
            )
            pending_reminders.extend(post.get("additional_context", []))
            content = post.get("content", content)
            is_error = post.get("is_error", is_error)

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": content,
                    "is_error": is_error,
                }
            )
        session.append("tool_result", tool_results)

    print("(max turns reached)", file=sys.stderr)


@dataclass
class AgentSession:
    client: AsyncAnthropic
    model: str
    session: SessionManager
    runner: HookRunner = field(default_factory=HookRunner)
    pending_reminders: list[str] = field(default_factory=list)
    max_turns: int = 25

    def start(self) -> None:
        ctx = {"cwd": os.getcwd(), "session": self.session}
        result = self.runner.fire("session_start", {"cwd": os.getcwd()}, ctx)
        self.pending_reminders.extend(result.get("additional_context", []))

    def end(self) -> None:
        ctx = {"cwd": os.getcwd(), "session": self.session}
        self.runner.fire("session_end", {}, ctx)

    async def prompt(self, text: str) -> None:
        ctx = {"cwd": os.getcwd(), "session": self.session}
        result = self.runner.fire("user_prompt_submit", {"prompt": text}, ctx)
        self.pending_reminders.extend(result.get("additional_context", []))
        if result.get("block"):
            print(
                f"(prompt blocked: {result.get('reason', 'no reason')})",
                file=sys.stderr,
            )
            return

        self.session.append("user", text)

        all_tools = self.runner.tools
        await agent_loop(
            self.client,
            self.model,
            os.getcwd(),
            all_tools,
            self.session,
            self.runner,
            self.pending_reminders,
            self.max_turns,
        )


def lifecycle_hook(api: HookAPI) -> None:
    api.register_event("before_session_load", PATH)
    api.register_event("args_parsed")
    api.register_event("build_session_config", MODEL, MAX_TURNS, CLIENT)
    api.register_event("session_start", ADDITIONAL_CONTEXT)
    api.register_event("user_prompt_submit", BLOCK, REASON, ADDITIONAL_CONTEXT)
    api.register_event("turn_start")
    api.register_event("build_system_prompt", SYSTEM_PROMPT, ADDITIONAL_CONTEXT)
    api.register_event("before_model_request", SYSTEM, TOOLS, MESSAGES)
    api.register_event("model_request_prepared")
    api.register_event("text_delta")
    api.register_event("text_end")
    api.register_event("message_end")
    api.register_event("pre_tool_use", BLOCK, REASON, INPUT, ADDITIONAL_CONTEXT)
    api.register_event("post_tool_use", CONTENT, IS_ERROR, ADDITIONAL_CONTEXT)
    api.register_event("stop")
    api.register_event("session_end")


def read_tool_hook(api: HookAPI) -> None:
    api.register_tool(_read_tool())


def write_tool_hook(api: HookAPI) -> None:
    api.register_tool(_write_tool())


def edit_tool_hook(api: HookAPI) -> None:
    api.register_tool(_edit_tool())


def bash_tool_hook(api: HookAPI) -> None:
    api.register_tool(_bash_tool())


def system_prompt_hook(api: HookAPI) -> None:
    # Closed over at load so the cached prefix stays stable across midnight.
    today = date.today().isoformat()

    def build(event: dict, _ctx: dict) -> dict:
        prompt = f"""You are a Python coding assistant. You have four tools: read, write, edit, bash.

Rules:
- Always `read` a file before you `write` or `edit` it.
- Prefer `edit` for small changes. Only `write` for new files or full rewrites.
- If a tool errors, read the error and try again.
- Keep replies short. Explain what you did, not what you're about to do.

Current date: {today}
Current working directory: {event['cwd']}
"""
        return {"system_prompt": prompt}

    api.on("build_system_prompt", build)


def prompt_arg_hook(api: HookAPI) -> None:
    api.register_flag("prompt", nargs="*", help="What to ask the agent.")


def debug_hooks_flag_hook(api: HookAPI) -> None:
    api.register_flag(
        "--debug-hooks",
        action="store_true",
        help="Print the lifecycle summary at startup.",
    )

    def maybe_print(_event: dict, ctx: dict) -> None:
        if ctx["args"].debug_hooks:
            print(ctx["runner"].describe(), file=sys.stderr)

    api.on("args_parsed", maybe_print)


def model_flag_hook(api: HookAPI) -> None:
    api.register_flag("--model", default="claude-sonnet-4-6")

    def provide(event: dict, _ctx: dict) -> dict:
        return {"model": event["args"].model}

    api.on("build_session_config", provide)


def max_turns_flag_hook(api: HookAPI) -> None:
    api.register_flag("--max-turns", type=int, default=25)

    def provide(event: dict, _ctx: dict) -> dict:
        return {"max_turns": event["args"].max_turns}

    api.on("build_session_config", provide)


def strict_hooks_flag_hook(api: HookAPI) -> None:
    api.register_flag(
        "--strict-hooks",
        action="store_true",
        help="Re-raise exceptions from hook handlers instead of logging them.",
    )

    def apply(event: dict, _ctx: dict) -> None:
        api.runner.strict = bool(event["args"].strict_hooks)

    api.on("args_parsed", apply)


def session_path_hook(api: HookAPI) -> None:
    def default(_event: dict, _ctx: dict) -> dict:
        session_dir = _session_dir(os.getcwd())
        path = (
            session_dir
            / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl"
        )
        return {"path": path}

    api.on("before_session_load", default)


def resume_hook(api: HookAPI) -> None:
    api.register_flag(
        "--new",
        action="store_true",
        help="Start a new session instead of continuing the most recent.",
    )
    api.register_flag(
        "--session",
        type=Path,
        default=None,
        help="Open a specific session file.",
    )

    def pick_session(event: dict, _ctx: dict) -> dict | None:
        args = event["args"]
        if args.session is not None:
            path = Path(args.session).expanduser()
            if not path.exists():
                print(f"session not found: {path}", file=sys.stderr)
                sys.exit(1)
            return {"path": path}
        if args.new:
            return None
        parent = _session_dir(os.getcwd())
        if not parent.exists():
            return None
        files = sorted(parent.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if files:
            return {"path": files[-1]}
        return None

    api.on("before_session_load", pick_session)


def _session_dir(cwd: str) -> Path:
    """Per-cwd session directory. Readable prefix + short hash to avoid collisions."""
    safe = re.sub(r"[/\\:\s]+", "-", cwd.lstrip("/\\"))
    digest = hashlib.sha1(cwd.encode()).hexdigest()[:8]
    return Path.home() / ".py-agent" / "sessions" / f"--{safe}-{digest}--"


def session_history_hook(api: HookAPI) -> None:
    """Provide prior user prompts from this cwd's sessions as history entries."""

    def load() -> list[str]:
        session_dir = _session_dir(os.getcwd())
        if not session_dir.exists():
            return []
        entries: list[str] = []
        for path in sorted(
            session_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime
        ):
            for line in path.read_text().splitlines():
                obj = json.loads(line)
                if (
                    obj.get("type") == "entry"
                    and obj.get("role") == "user"
                    and isinstance(obj.get("content"), str)
                    and obj["content"].strip()
                ):
                    entries.append(obj["content"].strip())
        return entries

    api.register_history_loader(load)


def prompt_toolkit_hook(api: HookAPI) -> None:
    """Register a prompt_toolkit prompter. Consumes `runner.history_loader` if present."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings

    runner = api.runner

    async def read(args: argparse.Namespace) -> str:
        if args.prompt:
            return " ".join(args.prompt)
        history = InMemoryHistory()
        if runner.history_loader:
            for entry in runner.history_loader():
                history.append_string(entry)
        kb = KeyBindings()
        kb.add("c-d")(lambda _: None)  # disable Ctrl-D — only Ctrl-C exits
        session = PromptSession(history=history, key_bindings=kb)
        try:
            return (await session.prompt_async("> ")).strip()
        except KeyboardInterrupt:
            return ""

    api.register_prompter(read)


def list_sessions_hook(api: HookAPI) -> None:
    api.register_flag(
        "--list-sessions",
        action="store_true",
        help="List prior sessions and exit.",
    )

    def summarize(path: Path) -> tuple[str, int, str]:
        created, entries, preview = "", 0, ""
        for line in path.read_text().splitlines():
            obj = json.loads(line)
            if obj.get("type") == "header":
                created = obj.get("createdAt", "")[:16].replace("T", " ")
            elif obj.get("type") == "entry":
                entries += 1
                if not preview and obj.get("role") == "user":
                    content = obj.get("content")
                    if isinstance(content, str):
                        preview = content[:60]
        return created, entries, preview

    def maybe_list(event: dict, _ctx: dict) -> None:
        if not event["args"].list_sessions:
            return
        session_dir = _session_dir(os.getcwd())
        files = (
            sorted(
                session_dir.glob("*.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if session_dir.exists()
            else []
        )
        if not files:
            print("no sessions yet")
            sys.exit(0)
        for path in files:
            created, entries, preview = summarize(path)
            print(f"{path.stem}  {created}  {entries} entries  {preview!r}")
        sys.exit(0)

    api.on("args_parsed", maybe_list)


def cache_debug_hook(api: HookAPI) -> None:
    """Dump the cache_control breakpoints in the outbound payload.

    Lets you verify whether the server is actually receiving the markers
    and roughly how many bytes sit in each cached prefix. Enable with
    --debug-cache.
    """

    api.register_flag("--debug-cache", action="store_true")
    enabled = {"v": False}

    def capture(event, _ctx):
        enabled["v"] = bool(getattr(event["args"], "debug_cache", False))

    def dump(event, _ctx):
        if not enabled["v"]:
            return
        system = event["system"]
        tools = event["tools"]
        messages = event["messages"]

        def has_cc(block):
            return isinstance(block, dict) and "cache_control" in block

        sys_marked = isinstance(system, list) and any(has_cc(b) for b in system)
        msgs_marked = False
        if messages:
            last = messages[-1].get("content")
            if isinstance(last, list):
                msgs_marked = any(has_cc(b) for b in last)

        sys_bytes = len(json.dumps(system))
        tools_bytes = len(json.dumps(tools))
        msgs_bytes = len(json.dumps(messages))
        print(
            f"[cache-debug] markers: system={sys_marked} "
            f"last_msg={msgs_marked} | bytes: system={sys_bytes} "
            f"tools={tools_bytes} messages={msgs_bytes}",
            file=sys.stderr,
        )

    api.on("args_parsed", capture)
    api.on("model_request_prepared", dump)


def anthropic_client_hook(api: HookAPI) -> None:
    def provide(_event: dict, _ctx: dict) -> dict:
        return {"client": AsyncAnthropic()}

    api.on("build_session_config", provide)


def anthropic_cache_hook(api: HookAPI) -> None:
    cc = {"type": "ephemeral"}

    def with_cache(content):
        if isinstance(content, str):
            return [{"type": "text", "text": content, "cache_control": cc}]
        if isinstance(content, list) and content:
            return [*content[:-1], {**content[-1], "cache_control": cc}]
        return content

    def mark(event, _ctx):
        system = event["system"]
        messages = event["messages"]
        return {
            "system": with_cache(system) if system else system,
            "messages": (
                [
                    *messages[:-1],
                    {**messages[-1], "content": with_cache(messages[-1]["content"])},
                ]
                if messages
                else messages
            ),
        }

    api.on("before_model_request", mark)


def _default_render_call(tool: Tool, args: dict, console: Any, _state: dict) -> None:
    args_repr = ", ".join(f"{k}={v!r}" for k, v in args.items())[:100]
    console.print(f"\n⏺ [bold]{tool.name.title()}[/bold]({escape(args_repr)})")


def _default_render_result(
    _tool: Tool,
    _args: dict,
    content: str,
    is_error: bool,
    console: Any,
    _state: dict,
) -> None:
    snippet = str(content).split("\n", 1)[0][:100] or "ok"
    color = "red" if is_error else "dim"
    console.print(f"  [{color}]⎿ {escape(snippet)}[/{color}]")


def _render_diff(console: Any, before: str, after: str) -> None:
    import difflib

    b, a = before.splitlines(), after.splitlines()
    ops = difflib.SequenceMatcher(None, b, a).get_opcodes()
    added = sum(j2 - j1 for op, _, _, j1, j2 in ops if op != "equal")
    removed = sum(i2 - i1 for op, i1, i2, _, _ in ops if op != "equal")
    if not added and not removed:
        return
    w = len(str(max(len(b), len(a), 1)))
    console.print(f"  [dim]⎿[/dim] [green]+{added}[/green] [red]-{removed}[/red]")
    for idx, (op, i1, i2, j1, j2) in enumerate(ops):
        if op != "equal":
            for k, line in enumerate(b[i1:i2]):
                console.print(f"    [red]{i1 + k + 1:>{w}} - {escape(line)}[/red]")
            for k, line in enumerate(a[j1:j2]):
                console.print(f"    [green]{j1 + k + 1:>{w}} + {escape(line)}[/green]")
            continue
        n = i2 - i1
        lead = 3 if idx > 0 else 0
        trail = 3 if idx < len(ops) - 1 else 0
        rng = range(n) if lead + trail >= n else [*range(lead), *range(n - trail, n)]
        for k in rng:
            console.print(f"    [dim]{i1 + k + 1:>{w}}[/dim]   {escape(b[i1 + k])}")


def ui_hook(api: HookAPI) -> None:
    """Single display extension — owns all terminal output in one visual style.

    Conventions:
      ⏺ bold name(args)   — agent actions (tool calls)
      ⎿ dim tree leaf     — nested results, cache, diff summary
      markdown             — assistant text (via rich.markdown)
      dim parenthetical    — session boundaries, ambient info
    Tools may override via Tool.render_call / render_result / capture_pre.
    """
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console(highlight=False)
    buf: list[str] = []

    def find(name: str) -> Tool | None:
        return next((t for t in api.runner.tools if t.name == name), None)

    def on_text_delta(event: dict, _ctx: dict) -> None:
        buf.append(event["text"])

    def on_text_end(_event: dict, _ctx: dict) -> None:
        if buf:
            console.print(Markdown("".join(buf)))
            buf.clear()

    def on_pre_tool(event: dict, _ctx: dict) -> None:
        tool = find(event["name"])
        if tool is None:
            return
        state = event.get("state", {})
        (tool.render_call or _default_render_call)(tool, event["input"], console, state)

    def on_post_tool(event: dict, _ctx: dict) -> None:
        tool = find(event["name"])
        if tool is None:
            return
        state = event.get("state", {})
        (tool.render_result or _default_render_result)(
            tool,
            event.get("input", {}),
            event.get("content", ""),
            bool(event.get("is_error")),
            console,
            state,
        )

    def on_message_end(event: dict, _ctx: dict) -> None:
        u = event.get("usage") or {}
        r = u.get("cache_read_input_tokens") or 0
        w = u.get("cache_creation_input_tokens") or 0
        i = u.get("input_tokens") or 0
        console.print(f"[dim]  ⎿ cache read={r} write={w} input={i}[/dim]")

    def on_session_start(_event: dict, ctx: dict) -> None:
        s = ctx["session"]
        if s.entries:
            console.print(f"[dim](resumed {s.path}, {len(s.entries)} entries)[/dim]")

    def on_session_end(_event: dict, ctx: dict) -> None:
        console.print(f"[dim](session saved to {ctx['session'].path})[/dim]")

    api.on("text_delta", on_text_delta)
    api.on("text_end", on_text_end)
    api.on("pre_tool_use", on_pre_tool)
    api.on("post_tool_use", on_post_tool)
    api.on("message_end", on_message_end)
    api.on("session_start", on_session_start)
    api.on("session_end", on_session_end)


HOOKS = (
    prompt_arg_hook,
    model_flag_hook,
    max_turns_flag_hook,
    strict_hooks_flag_hook,
    session_history_hook,
    prompt_toolkit_hook,
    debug_hooks_flag_hook,
    session_path_hook,
    resume_hook,
    list_sessions_hook,
    system_prompt_hook,
    read_tool_hook,
    write_tool_hook,
    edit_tool_hook,
    bash_tool_hook,
    anthropic_client_hook,
    anthropic_cache_hook,
    cache_debug_hook,
    ui_hook,
)


async def main() -> None:
    runner = HookRunner()
    for hook in HOOKS:
        runner.load(hook)

    args = runner.parse_args()
    ctx = {"args": args, "runner": runner}
    runner.fire("args_parsed", {"args": args}, ctx)

    config = runner.fire("build_session_config", {"args": args}, ctx)
    session_path = runner.fire("before_session_load", {"args": args}, ctx)["path"]

    prompt = await runner.prompter(args)
    if not prompt:
        return

    session = SessionManager(session_path)
    agent = AgentSession(session=session, runner=runner, **config)

    agent.start()
    try:
        await agent.prompt(prompt)
    finally:
        agent.end()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
