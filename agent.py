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


class Merge(Enum):
    REPLACE = "replace"
    ACCUMULATE = "accumulate"
    BLOCK = "block"


@dataclass(frozen=True)
class Return:
    key: str
    kind: Merge = Merge.REPLACE


BLOCK = Return("block", kind=Merge.BLOCK)
REASON = Return("reason")
INPUT = Return("input")
CONTENT = Return("content")
IS_ERROR = Return("is_error")
SYSTEM_PROMPT = Return("system_prompt")
PATH = Return("path")
ADDITIONAL_CONTEXT = Return("additional_context", kind=Merge.ACCUMULATE)
SYSTEM = Return("system")
TOOLS = Return("tools")
MESSAGES = Return("messages")


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
        for handler in self.handlers[event]:
            try:
                r = handler(payload, ctx) or {}
            except Exception as e:
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

    return Tool(
        name="edit",
        description="Replace `old` with `new` in a file. `old` must appear exactly once.",
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
            pre = runner.fire("pre_tool_use", {"name": tu.name, "input": tu.input}, ctx)
            pending_reminders.extend(pre.get("additional_context", []))

            if pre.get("block"):
                content, is_error = pre.get("reason", "blocked by hook"), True
            else:
                effective_input = pre.get("input", tu.input)
                tool = tool_map.get(tu.name)
                if tool is None:
                    content, is_error = f"unknown tool: {tu.name}", True
                else:
                    content, is_error = tool.execute(effective_input)

            post = runner.fire(
                "post_tool_use",
                {"name": tu.name, "content": content, "is_error": is_error},
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
        )


def lifecycle_hook(api: HookAPI) -> None:
    api.register_event("before_session_load", PATH)
    api.register_event("args_parsed")
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
    def build(event: dict, _ctx: dict) -> dict:
        prompt = f"""You are a Python coding assistant. You have four tools: read, write, edit, bash.

Rules:
- Always `read` a file before you `write` or `edit` it.
- Prefer `edit` for small changes. Only `write` for new files or full rewrites.
- If a tool errors, read the error and try again.
- Keep replies short. Explain what you did, not what you're about to do.

Current date: {date.today().isoformat()}
Current working directory: {event['cwd']}
"""
        return {"system_prompt": prompt}

    api.on("build_system_prompt", build)


def session_start_printer_hook(api: HookAPI) -> None:
    def on_start(_event: dict, ctx: dict) -> None:
        session = ctx["session"]
        if session.entries:
            print(
                f"(resumed {session.path}, {len(session.entries)} entries)",
                file=sys.stderr,
            )

    api.on("session_start", on_start)


def session_end_printer_hook(api: HookAPI) -> None:
    def on_end(_event: dict, ctx: dict) -> None:
        print(f"\n(session saved to {ctx['session'].path})", file=sys.stderr)

    api.on("session_end", on_end)


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
        parent = event["default_path"].parent
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


def cache_stats_hook(api: HookAPI) -> None:
    def on_msg(event, _ctx):
        u = event.get("usage") or {}
        read = u.get("cache_read_input_tokens") or 0
        write = u.get("cache_creation_input_tokens") or 0
        total = u.get("input_tokens") or 0
        print(
            f"[cache] read={read} write={write} input={total}",
            file=sys.stderr,
        )

    api.on("message_end", on_msg)


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


def tool_call_renderer_hook(api: HookAPI) -> None:
    def on_pre_tool_use(event: dict, _ctx: dict) -> None:
        args_repr = ", ".join(f"{k}={v!r}" for k, v in event["input"].items())[:120]
        print(f"\n  → {event['name']}({args_repr})")

    api.on("pre_tool_use", on_pre_tool_use)


def markdown_renderer_hook(api: HookAPI) -> None:
    """Buffer assistant text and render it as markdown on each turn end."""
    from rich.console import Console
    from rich.markdown import Markdown

    buf: list[str] = []
    console = Console()

    def on_delta(event: dict, _ctx: dict) -> None:
        buf.append(event["text"])

    def on_end(_event: dict, _ctx: dict) -> None:
        if buf:
            console.print(Markdown("".join(buf)))
            buf.clear()

    api.on("text_delta", on_delta)
    api.on("text_end", on_end)


async def main() -> None:
    runner = HookRunner()
    for hook in (
        prompt_arg_hook,
        session_history_hook,
        prompt_toolkit_hook,
        debug_hooks_flag_hook,
        resume_hook,
        list_sessions_hook,
        system_prompt_hook,
        read_tool_hook,
        write_tool_hook,
        edit_tool_hook,
        bash_tool_hook,
        anthropic_cache_hook,
        cache_stats_hook,
        cache_debug_hook,
        tool_call_renderer_hook,
        markdown_renderer_hook,
        session_start_printer_hook,
        session_end_printer_hook,
    ):
        runner.load(hook)

    args = runner.parse_args()
    ctx = {"args": args, "runner": runner}
    runner.fire("args_parsed", {"args": args}, ctx)

    session_dir = _session_dir(os.getcwd())
    default_path = (
        session_dir
        / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl"
    )
    override = runner.fire(
        "before_session_load",
        {"args": args, "default_path": default_path},
        ctx,
    )
    session_path = override.get("path", default_path)

    prompt = (
        await runner.prompter(args)
        if runner.prompter
        else " ".join(args.prompt) if args.prompt else ""
    )
    if not prompt:
        return

    session = SessionManager(session_path)

    agent = AgentSession(
        client=AsyncAnthropic(),
        model="claude-sonnet-4-6",
        session=session,
        runner=runner,
    )

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
