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

## Extension model

A *hook* is a function `(HookAPI) -> None` that runs once at startup
and wires its contributions into the runner. Through `HookAPI` a hook
can attach handlers to named events, register tools, add CLI flags,
or provide the prompter and history loader.

When an event fires it carries a typed `Payload` dataclass. Handlers
receive the payload and mutate it in place, in priority order (lower
first, insertion order on ties). Any handler can stop the chain for
that fire by setting `payload.blocked = True`.
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
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from anthropic import AsyncAnthropic
from rich.markup import escape

# ═══════════════════════════════════════════════════════════════════════════
# Hook primitives
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Payload:
    """Base class for every event payload. Handlers mutate subclasses in place.

    Setting `blocked = True` (optionally with a `reason`) stops further
    handlers for the current fire().
    """

    blocked: bool = False
    reason: str = ""


# --- Concrete payloads. One per event that carries data. ----------------


@dataclass(kw_only=True)
class ArgsParsed(Payload):
    args: argparse.Namespace


@dataclass(kw_only=True)
class SessionPath(Payload):
    args: argparse.Namespace
    path: Path | None = None


@dataclass(kw_only=True)
class SessionConfig(Payload):
    args: argparse.Namespace
    model: str | None = None
    max_turns: int = 25
    client: Any = None


@dataclass(kw_only=True)
class SessionStart(Payload):
    cwd: str
    additional_context: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class UserPrompt(Payload):
    prompt: str
    additional_context: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class SystemPrompt(Payload):
    cwd: str
    system_prompt: str = ""
    additional_context: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class ModelRequest(Payload):
    system: Any  # str or list of blocks (after cache decoration)
    tools: list
    messages: list
    extra: dict = field(
        default_factory=dict
    )  # becomes **kwargs to client.messages.stream


@dataclass(kw_only=True)
class TextDelta(Payload):
    text: str


@dataclass(kw_only=True)
class MessageEnd(Payload):
    message: Any
    usage: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class PreTool(Payload):
    id: str
    name: str
    input: dict
    state: dict = field(default_factory=dict)
    additional_context: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class PostTool(Payload):
    id: str
    name: str
    input: dict
    content: str
    is_error: bool
    state: dict = field(default_factory=dict)
    additional_context: list[str] = field(default_factory=list)


# --- Event + runner -----------------------------------------------------


@dataclass(frozen=True)
class Event:
    name: str
    payload_cls: type[Payload] | None = None  # None = bare signal, no payload


@dataclass
class _Handler:
    fn: Callable[[Any, dict], Any]
    priority: int = 50


class HookAPI:
    def __init__(self, runner: "HookRunner") -> None:
        self._runner = runner

    def register_event(
        self, name: str, payload_cls: type[Payload] | None = None
    ) -> None:
        if name in self._runner.events:
            raise ValueError(f"event {name!r} already registered")
        self._runner.events[name] = Event(name, payload_cls)
        self._runner.handlers[name] = []

    def on(
        self,
        event: str,
        fn: Callable[[Any, dict], Any] | None = None,
        *,
        priority: int = 50,
    ) -> Any:
        """Register a handler for `event`. Call directly or use as a decorator."""
        if event not in self._runner.events:
            raise ValueError(
                f"unknown event {event!r}. Known: {', '.join(self._runner.events) or '(none yet)'}"
            )

        def register(handler: Callable) -> Callable:
            self._runner.handlers[event].append(_Handler(handler, priority))
            return handler

        return register if fn is None else register(fn)

    def register_tool(self, tool: "Tool") -> None:
        if any(t.name == tool.name for t in self._runner.tools):
            raise ValueError(f"tool {tool.name!r} already registered")
        self._runner.tools.append(tool)

    def prompter(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register the async callable that reads the initial user prompt."""
        self._runner.prompter = fn
        return fn

    def history_loader(self, fn: Callable[[], list[str]]) -> Callable[[], list[str]]:
        """Register a callable that returns prior prompts for up/down navigation."""
        self._runner.history_loader = fn
        return fn

    def register_flag(self, *args: Any, **kwargs: Any) -> None:
        self._runner.parser.add_argument(*args, **kwargs)

    @property
    def runner(self) -> "HookRunner":
        return self._runner


class HookRunner:
    def __init__(self) -> None:
        self.events: dict[str, Event] = {}
        self.handlers: dict[str, list[_Handler]] = {}
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
        except Exception:
            print(f"[hook {hook.__name__} failed to load]", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    def parse_args(self) -> argparse.Namespace:
        return self.parser.parse_args()

    def fire(
        self, event: str, payload: Payload | None = None, ctx: dict | None = None
    ) -> Payload | None:
        """Run every handler registered on `event`, in priority order (lower first,
        insertion order on ties). Handlers mutate `payload`. If a handler sets
        `payload.blocked = True`, later handlers are skipped. Returns `payload`
        so callers can destructure directly."""
        if event not in self.events:
            raise ValueError(f"unknown event {event!r}")
        ctx = ctx or {}
        ordered = sorted(self.handlers[event], key=lambda h: h.priority)
        for h in ordered:
            try:
                h.fn(payload, ctx)
            except Exception:
                if self.strict:
                    raise
                print(f"[hook error on {event}]", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                continue
            if payload is not None and payload.blocked:
                break
        return payload

    def describe(self) -> str:
        return "\n".join(
            f"  {name:<22}  {len(self.handlers[name])} handler(s)"
            for name in self.events
        )


# ═══════════════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════════════


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


def _tool_fn(
    fn: Callable[[dict], tuple[str, bool]],
) -> Callable[[dict], tuple[str, bool]]:
    """Wrap a tool body so unexpected exceptions become `(error_text, True)`."""

    def execute(args: dict) -> tuple[str, bool]:
        try:
            return fn(args)
        except Exception as e:
            return f"{type(e).__name__}: {e}", True

    return execute


def _read_tool() -> Tool:
    MAX_BYTES = 50 * 1024  # pi-mono-style byte cap, whole-line boundary

    @_tool_fn
    def execute(args: dict) -> tuple[str, bool]:
        lines = Path(args["path"]).read_text().splitlines()
        offset = max(int(args.get("offset", 0)), 0)
        limit = max(int(args.get("limit", 500)), 1)
        chunk = lines[offset : offset + limit]
        if not chunk:
            return "(empty)", False
        # Byte-cap: keep whole lines whose cumulative size ≤ MAX_BYTES.
        kept: list[str] = []
        used = 0
        for line in chunk:
            size = len(line.encode("utf-8")) + (1 if kept else 0)  # +1 for \n
            if used + size > MAX_BYTES:
                break
            kept.append(line)
            used += size
        if not kept:
            return (
                f"(line {offset + 1} alone exceeds the {MAX_BYTES // 1024}KB byte cap; "
                f"try `bash` with `sed -n '{offset + 1}p' <file> | head -c {MAX_BYTES}`)",
                False,
            )
        width = len(str(offset + len(kept)))
        numbered = "\n".join(
            f"{offset + i + 1:>{width}}\t{line}" for i, line in enumerate(kept)
        )
        remaining = len(lines) - offset - len(kept)
        if remaining > 0:
            numbered += (
                f"\n... ({remaining} more lines; "
                f"pass offset={offset + len(kept)} to continue)"
            )
        return numbered, False

    return Tool(
        name="read",
        description=(
            "Read a file from disk. Returns lines prefixed with 1-based line numbers "
            "separated by a tab (e.g. '  12\\tfoo'). Optional `offset` (0-based line "
            "index, default 0) and `limit` (default 500) page through large files; "
            "output is additionally capped at 50KB on a whole-line boundary. "
            "The line-number prefix is display only — never include it in `edit`'s "
            "`old` or `write`'s `content`."
        ),
        schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
        },
        execute=execute,
    )


def _write_tool() -> Tool:
    @_tool_fn
    def execute(args: dict) -> tuple[str, bool]:
        Path(args["path"]).write_text(args["content"])
        return f"wrote {len(args['content'])} bytes to {args['path']}", False

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
    @_tool_fn
    def execute(args: dict) -> tuple[str, bool]:
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
            "Replace `old` with `new` in a file. `old` must appear exactly once and "
            "must match raw file bytes — do not include the '<n>\\t' line-number "
            "prefix that `read` adds for display. When deleting a whole line, include "
            "its trailing newline in `old` (e.g. old='foo\\n', new='') — otherwise a "
            "blank line is left behind."
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
    @_tool_fn
    def execute(args: dict) -> tuple[str, bool]:
        try:
            r = subprocess.run(
                args["cmd"], shell=True, capture_output=True, text=True, timeout=60
            )
        except subprocess.TimeoutExpired:
            return "bash timed out after 60s", True
        return (r.stdout + r.stderr)[:20_000] or "(no output)", r.returncode != 0

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


# ═══════════════════════════════════════════════════════════════════════════
# Session
# ═══════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════
# Agent loop
# ═══════════════════════════════════════════════════════════════════════════


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
        runner.fire("turn_start", None, ctx)

        sp = runner.fire("build_system_prompt", SystemPrompt(cwd=cwd), ctx)
        assert isinstance(sp, SystemPrompt)
        system: Any = sp.system_prompt
        for extra in sp.additional_context:
            system += f"\n\n{extra}"

        mr = ModelRequest(system=system, tools=schemas, messages=session.to_messages())
        runner.fire("before_model_request", mr, ctx)
        max_tokens = mr.extra.pop("max_tokens", 64000)
        runner.fire("model_request_prepared", mr, ctx)

        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=mr.system,
            tools=mr.tools,
            messages=mr.messages,
            **mr.extra,
        ) as stream:
            had_text = False
            async for text in stream.text_stream:
                runner.fire("text_delta", TextDelta(text=text), ctx)
                had_text = True
            if had_text:
                runner.fire("text_end", None, ctx)
            res = await stream.get_final_message()

        dumped = res.model_dump(mode="json")
        assistant_content = dumped["content"]
        session.append("assistant", assistant_content)
        runner.fire(
            "message_end",
            MessageEnd(message=assistant_content, usage=res.usage.model_dump()),
            ctx,
        )

        tool_uses = [b for b in res.content if b.type == "tool_use"]
        if not tool_uses:
            runner.fire("stop", None, ctx)
            return

        tool_results = []
        for tu in tool_uses:
            pre = PreTool(id=tu.id, name=tu.name, input=tu.input, state={})
            runner.fire("pre_tool_use", pre, ctx)
            pending_reminders.extend(pre.additional_context)

            if pre.blocked:
                content, is_error = pre.reason or "blocked by hook", True
            else:
                tool = tool_map.get(tu.name)
                if tool is None:
                    content, is_error = f"unknown tool: {tu.name}", True
                else:
                    content, is_error = tool.execute(pre.input)

            post = PostTool(
                id=tu.id,
                name=tu.name,
                input=pre.input,
                content=content,
                is_error=is_error,
                state=pre.state,
            )
            runner.fire("post_tool_use", post, ctx)
            pending_reminders.extend(post.additional_context)

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": post.content,
                    "is_error": post.is_error,
                }
            )
        # Attach pending reminders to the same user message as tool_results.
        tool_results.extend({"type": "text", "text": r} for r in pending_reminders)
        pending_reminders.clear()
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

    def _ctx(self) -> dict:
        return {"cwd": os.getcwd(), "session": self.session}

    def start(self) -> None:
        p = self.runner.fire(
            "session_start", SessionStart(cwd=os.getcwd()), self._ctx()
        )
        assert isinstance(p, SessionStart)
        self.pending_reminders.extend(p.additional_context)

    def end(self) -> None:
        self.runner.fire("session_end", None, self._ctx())

    async def prompt(self, text: str) -> None:
        p = self.runner.fire("user_prompt_submit", UserPrompt(prompt=text), self._ctx())
        assert isinstance(p, UserPrompt)
        self.pending_reminders.extend(p.additional_context)
        if p.blocked:
            print(
                f"(prompt blocked: {p.reason or 'no reason'})",
                file=sys.stderr,
            )
            return

        # User prompt + any pending reminders ride on the same user message.
        content = [{"type": "text", "text": text}]
        content.extend({"type": "text", "text": r} for r in self.pending_reminders)
        self.pending_reminders.clear()
        self.session.append("user", content)

        await agent_loop(
            self.client,
            self.model,
            os.getcwd(),
            self.runner.tools,
            self.session,
            self.runner,
            self.pending_reminders,
            self.max_turns,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Lifecycle
# ═══════════════════════════════════════════════════════════════════════════


def lifecycle_hook(api: HookAPI) -> None:
    api.register_event("before_session_load", SessionPath)
    api.register_event("args_parsed", ArgsParsed)
    api.register_event("build_session_config", SessionConfig)
    api.register_event("session_start", SessionStart)
    api.register_event("user_prompt_submit", UserPrompt)
    api.register_event("turn_start")
    api.register_event("build_system_prompt", SystemPrompt)
    api.register_event("before_model_request", ModelRequest)
    api.register_event("model_request_prepared", ModelRequest)
    api.register_event("text_delta", TextDelta)
    api.register_event("text_end")
    api.register_event("message_end", MessageEnd)
    api.register_event("pre_tool_use", PreTool)
    api.register_event("post_tool_use", PostTool)
    api.register_event("stop")
    api.register_event("session_end")


# ═══════════════════════════════════════════════════════════════════════════
# Built-in hooks
# ═══════════════════════════════════════════════════════════════════════════


def read_tool_hook(api: HookAPI) -> None:
    api.register_tool(_read_tool())


def write_tool_hook(api: HookAPI) -> None:
    api.register_tool(_write_tool())


def edit_tool_hook(api: HookAPI) -> None:
    api.register_tool(_edit_tool())


def bash_tool_hook(api: HookAPI) -> None:
    api.register_tool(_bash_tool())


def system_prompt_hook(api: HookAPI) -> None:
    today = date.today().isoformat()

    @api.on("build_system_prompt")
    def build(p: SystemPrompt, _ctx: dict) -> None:
        p.system_prompt = f"""You are a Python coding assistant. You have four tools: read, write, edit, bash.

Rules:
- Always `read` a file before you `write` or `edit` it.
- Prefer `edit` for small changes. Only `write` for new files or full rewrites.
- If a tool errors, read the error and try again.
- Verify results with tools before claiming done (re-read the file after editing, run the test, check the exit code).
- Keep replies short. Explain what you did, not what you're about to do.

Current date: {today}
Current working directory: {p.cwd}
"""


def skills_hook(api: HookAPI) -> None:
    """Surface Agent Skills (agentskills.io) so the model can load them on demand.

    Scans, first-wins: ~/.py-agent/skills (user) then <cwd>/.agent/skills (project).
    A skill is either a directory containing SKILL.md or a top-level .md file in a
    root. Each must start with YAML-style `---` frontmatter with a non-empty
    `description`; `name` defaults to the parent directory name. The model reads
    the file via the `read` tool when the description matches the task.
    """

    def parse(path: Path) -> dict | None:
        try:
            text = path.read_text()
        except OSError:
            return None
        if not text.startswith("---\n"):
            return None
        _, _, rest = text.partition("---\n")
        fm, sep, _ = rest.partition("\n---")
        if not sep:
            return None
        meta: dict[str, str] = {}
        for line in fm.splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip("\"'")
        desc = meta.get("description", "")
        if not desc:
            return None
        return {
            "name": meta.get("name") or path.parent.name,
            "description": desc,
            "path": str(path),
        }

    def discover(root: Path) -> list[dict]:
        if not root.is_dir():
            return []
        out: list[dict] = []
        for entry in sorted(root.iterdir()):
            if entry.name.startswith(".") or entry.name == "node_modules":
                continue
            if entry.is_dir():
                s = parse(entry / "SKILL.md")
            elif entry.is_file() and entry.suffix == ".md":
                s = parse(entry)
            else:
                s = None
            if s:
                out.append(s)
        return out

    @api.on("build_system_prompt")
    def inject(p: SystemPrompt, _ctx: dict) -> None:
        seen: dict[str, dict] = {}
        for root in (
            Path.home() / ".py-agent" / "skills",
            Path(p.cwd) / ".agent" / "skills",
        ):
            for s in discover(root):
                seen.setdefault(s["name"], s)
        if not seen:
            return
        lines = [
            "The following skills provide specialized instructions for specific tasks.",
            "Use the `read` tool to load a skill's file when the task matches its description.",
            "Resolve any relative paths inside a skill file against that file's directory.",
            "",
            "<available_skills>",
        ]
        for s in seen.values():
            lines += [
                "  <skill>",
                f"    <name>{s['name']}</name>",
                f"    <description>{s['description']}</description>",
                f"    <location>{s['path']}</location>",
                "  </skill>",
            ]
        lines.append("</available_skills>")
        p.additional_context.append("\n".join(lines))


def prompt_arg_hook(api: HookAPI) -> None:
    api.register_flag("prompt", nargs="*", help="What to ask the agent.")


def debug_hooks_flag_hook(api: HookAPI) -> None:
    api.register_flag(
        "--debug-hooks",
        action="store_true",
        help="Print the lifecycle summary at startup.",
    )

    @api.on("args_parsed")
    def maybe_print(p: ArgsParsed, ctx: dict) -> None:
        if p.args.debug_hooks:
            print(ctx["runner"].describe(), file=sys.stderr)


def model_flag_hook(api: HookAPI) -> None:
    api.register_flag("--model", default="claude-sonnet-4-6")

    @api.on("build_session_config")
    def provide(p: SessionConfig, _ctx: dict) -> None:
        p.model = p.args.model


def max_turns_flag_hook(api: HookAPI) -> None:
    api.register_flag("--max-turns", type=int, default=25)

    @api.on("build_session_config")
    def provide(p: SessionConfig, _ctx: dict) -> None:
        p.max_turns = p.args.max_turns


def _stream_extra_hook(
    api: HookAPI,
    build: Callable[[argparse.Namespace], dict | None],
    *flag_args: Any,
    **flag_kwargs: Any,
) -> None:
    """Shared shape: register a flag, read parsed args, contribute to `extra`.

    `build(args)` returns a dict to merge into the stream kwargs (e.g.
    `{"thinking": {...}}`), or a falsy value to skip.
    """
    api.register_flag(*flag_args, **flag_kwargs)
    cell: dict[str, argparse.Namespace] = {}

    @api.on("args_parsed")
    def capture(p: ArgsParsed, _ctx: dict) -> None:
        cell["args"] = p.args

    @api.on("before_model_request")
    def inject(p: ModelRequest, _ctx: dict) -> None:
        args = cell.get("args")
        if args is None:
            return
        extra = build(args)
        if extra:
            p.extra.update(extra)


def max_tokens_flag_hook(api: HookAPI) -> None:
    """--max-tokens N — caps the per-response output budget (default 64000)."""
    _stream_extra_hook(
        api,
        lambda a: {"max_tokens": a.max_tokens},
        "--max-tokens",
        type=int,
        default=64000,
    )


def thinking_hook(api: HookAPI) -> None:
    """--thinking-display VALUE — adaptive thinking, always on (default 'summarized')."""
    _stream_extra_hook(
        api,
        lambda a: {"thinking": {"type": "adaptive", "display": a.thinking_display}},
        "--thinking-display",
        default="summarized",
    )


def output_effort_hook(api: HookAPI) -> None:
    """--effort LEVEL — sets output_config.effort (default 'xhigh')."""
    _stream_extra_hook(
        api,
        lambda a: {"output_config": {"effort": a.effort}},
        "--effort",
        choices=("low", "medium", "high", "xhigh"),
        default="xhigh",
    )


def strict_hooks_flag_hook(api: HookAPI) -> None:
    api.register_flag(
        "--strict-hooks",
        action="store_true",
        help="Re-raise exceptions from hook handlers instead of logging them.",
    )

    @api.on("args_parsed")
    def apply(p: ArgsParsed, _ctx: dict) -> None:
        api.runner.strict = bool(p.args.strict_hooks)


def session_path_hook(api: HookAPI) -> None:
    def default(p: SessionPath, _ctx: dict) -> None:
        session_dir = _session_dir(os.getcwd())
        p.path = (
            session_dir
            / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl"
        )

    # Priority 90: run after resume_hook (priority 50) so --session / latest
    # selection wins; we only set a default when nothing else did.
    @api.on("before_session_load", priority=90)
    def default_if_unset(p: SessionPath, ctx: dict) -> None:
        if p.path is None:
            default(p, ctx)


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

    @api.on("before_session_load")
    def pick_session(p: SessionPath, _ctx: dict) -> None:
        args = p.args
        if args.session is not None:
            path = Path(args.session).expanduser()
            if not path.exists():
                print(f"session not found: {path}", file=sys.stderr)
                sys.exit(1)
            p.path = path
            return
        if args.new:
            return
        parent = _session_dir(os.getcwd())
        if not parent.exists():
            return
        files = sorted(parent.glob("*.jsonl"), key=lambda x: x.stat().st_mtime)
        if files:
            p.path = files[-1]


def _session_dir(cwd: str) -> Path:
    """Per-cwd session directory. Readable prefix + short hash to avoid collisions."""
    safe = re.sub(r"[/\\:\s]+", "-", cwd.lstrip("/\\"))
    digest = hashlib.sha1(cwd.encode()).hexdigest()[:8]
    return Path.home() / ".py-agent" / "sessions" / f"--{safe}-{digest}--"


def session_history_hook(api: HookAPI) -> None:
    """Provide prior user prompts from this cwd's sessions as history entries."""

    @api.history_loader
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
                if obj.get("type") != "entry" or obj.get("role") != "user":
                    continue
                content = obj.get("content")
                # New sessions store user content as a list whose first block
                # is the prompt text; old sessions store it as a bare string.
                if isinstance(content, list) and content:
                    first = content[0]
                    content = first.get("text", "") if isinstance(first, dict) else ""
                if isinstance(content, str) and content.strip():
                    entries.append(content.strip())
        return entries


def prompt_toolkit_hook(api: HookAPI) -> None:
    """Register a prompt_toolkit prompter. Consumes `runner.history_loader` if present."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings

    runner = api.runner

    @api.prompter
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

    @api.on("args_parsed")
    def maybe_list(p: ArgsParsed, _ctx: dict) -> None:
        if not p.args.list_sessions:
            return
        session_dir = _session_dir(os.getcwd())
        files = (
            sorted(
                session_dir.glob("*.jsonl"),
                key=lambda x: x.stat().st_mtime,
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


def cache_debug_hook(api: HookAPI) -> None:
    """Dump the cache_control breakpoints in the outbound payload.

    Lets you verify whether the server is actually receiving the markers
    and roughly how many bytes sit in each cached prefix. Enable with
    --debug-cache.
    """

    api.register_flag("--debug-cache", action="store_true")
    enabled = {"v": False}

    @api.on("args_parsed")
    def capture(p: ArgsParsed, _ctx: dict) -> None:
        enabled["v"] = bool(getattr(p.args, "debug_cache", False))

    @api.on("model_request_prepared")
    def dump(p: ModelRequest, _ctx: dict) -> None:
        if not enabled["v"]:
            return

        def has_cc(block: Any) -> bool:
            return isinstance(block, dict) and "cache_control" in block

        sys_marked = isinstance(p.system, list) and any(has_cc(b) for b in p.system)
        msgs_marked = False
        if p.messages:
            last = p.messages[-1].get("content")
            if isinstance(last, list):
                msgs_marked = any(has_cc(b) for b in last)

        sys_bytes = len(json.dumps(p.system))
        tools_bytes = len(json.dumps(p.tools))
        msgs_bytes = len(json.dumps(p.messages))
        print(
            f"[cache-debug] markers: system={sys_marked} "
            f"last_msg={msgs_marked} | bytes: system={sys_bytes} "
            f"tools={tools_bytes} messages={msgs_bytes}",
            file=sys.stderr,
        )


def anthropic_client_hook(api: HookAPI) -> None:
    @api.on("build_session_config")
    def provide(p: SessionConfig, _ctx: dict) -> None:
        p.client = AsyncAnthropic()


def anthropic_cache_hook(api: HookAPI) -> None:
    cc = {"type": "ephemeral"}

    def with_cache(content: Any) -> Any:
        if isinstance(content, str):
            return [{"type": "text", "text": content, "cache_control": cc}]
        if isinstance(content, list) and content:
            return [*content[:-1], {**content[-1], "cache_control": cc}]
        return content

    @api.on("before_model_request")
    def mark(p: ModelRequest, _ctx: dict) -> None:
        if p.system:
            p.system = with_cache(p.system)
        if p.messages:
            last = p.messages[-1]
            p.messages = [
                *p.messages[:-1],
                {**last, "content": with_cache(last["content"])},
            ]


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

    @api.on("text_delta")
    def on_text_delta(p: TextDelta, _ctx: dict) -> None:
        buf.append(p.text)

    @api.on("text_end")
    def on_text_end(_p: Any, _ctx: dict) -> None:
        if buf:
            console.print(Markdown("".join(buf)))
            buf.clear()

    @api.on("pre_tool_use")
    def on_pre_tool(p: PreTool, _ctx: dict) -> None:
        tool = find(p.name)
        if tool is None:
            return
        (tool.render_call or _default_render_call)(tool, p.input, console, p.state)

    @api.on("post_tool_use")
    def on_post_tool(p: PostTool, _ctx: dict) -> None:
        tool = find(p.name)
        if tool is None:
            return
        (tool.render_result or _default_render_result)(
            tool,
            p.input,
            p.content,
            p.is_error,
            console,
            p.state,
        )

    @api.on("message_end")
    def on_message_end(p: MessageEnd, _ctx: dict) -> None:
        u = p.usage or {}
        r = u.get("cache_read_input_tokens") or 0
        w = u.get("cache_creation_input_tokens") or 0
        i = u.get("input_tokens") or 0
        console.print(f"[dim]  ⎿ cache read={r} write={w} input={i}[/dim]")

    @api.on("session_start")
    def on_session_start(_p: SessionStart, ctx: dict) -> None:
        s = ctx["session"]
        if s.entries:
            console.print(f"[dim](resumed {s.path}, {len(s.entries)} entries)[/dim]")

    @api.on("session_end")
    def on_session_end(_p: Any, ctx: dict) -> None:
        console.print(f"[dim](session saved to {ctx['session'].path})[/dim]")


HOOKS = (
    prompt_arg_hook,
    model_flag_hook,
    max_turns_flag_hook,
    max_tokens_flag_hook,
    thinking_hook,
    output_effort_hook,
    strict_hooks_flag_hook,
    session_history_hook,
    prompt_toolkit_hook,
    debug_hooks_flag_hook,
    session_path_hook,
    resume_hook,
    list_sessions_hook,
    system_prompt_hook,
    skills_hook,
    read_tool_hook,
    write_tool_hook,
    edit_tool_hook,
    bash_tool_hook,
    anthropic_client_hook,
    anthropic_cache_hook,
    cache_debug_hook,
    ui_hook,
)


# ═══════════════════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════════════════


async def main() -> None:
    runner = HookRunner()
    for hook in HOOKS:
        runner.load(hook)

    args = runner.parse_args()
    ctx = {"args": args, "runner": runner}
    runner.fire("args_parsed", ArgsParsed(args=args), ctx)

    cfg = runner.fire("build_session_config", SessionConfig(args=args), ctx)
    assert isinstance(cfg, SessionConfig)
    sp = runner.fire("before_session_load", SessionPath(args=args), ctx)
    assert isinstance(sp, SessionPath) and sp.path is not None

    prompt = await runner.prompter(args)
    if not prompt:
        return

    session = SessionManager(sp.path)
    agent = AgentSession(
        client=cfg.client,
        model=cfg.model or "claude-sonnet-4-6",
        max_turns=cfg.max_turns,
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
