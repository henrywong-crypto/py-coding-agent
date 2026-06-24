# /// script
# requires-python = ">=3.11"
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

Built-in hooks live in this file's `HOOKS` tuple. Further hooks can be
dropped into sibling `.py` files in this directory: any file exposing a
module-level `HOOKS` tuple is auto-loaded at startup (see `load_extensions`),
importing what it needs from `agent`. `python.py` and `rust.py` are such
extensions — delete one and that language support is simply gone.

When an event fires it carries a typed `Payload` dataclass. Handlers
receive the payload and mutate it in place, in priority order (lower
first, insertion order on ties). Any handler can stop the chain for
that fire by setting `payload.blocked = True`.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

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
    max_turns: int = 50
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

    async def fire(
        self, event: str, payload: Payload | None = None, ctx: dict | None = None
    ) -> Payload | None:
        """Run every handler registered on `event`, in priority order (lower first,
        insertion order on ties). Handlers mutate `payload`. A handler may be a
        plain function or a coroutine function — if it returns an awaitable, it is
        awaited before the next runs. If a handler sets `payload.blocked = True`,
        later handlers are skipped. Returns `payload` so callers can destructure
        directly."""
        if event not in self.events:
            raise ValueError(f"unknown event {event!r}")
        ctx = ctx or {}
        ordered = sorted(self.handlers[event], key=lambda h: h.priority)
        for h in ordered:
            try:
                result = h.fn(payload, ctx)
                if inspect.isawaitable(result):
                    await result
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
    execute: Callable[[dict], tuple[str, bool] | Awaitable[tuple[str, bool]]]
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
    """Wrap a tool body so unexpected exceptions become `(error_text, True)`.

    Handles sync and `async def` bodies alike: an async body is awaited inside
    the same guard, so an exception raised while it runs is caught too — not
    just one raised before it returns the coroutine."""

    if inspect.iscoroutinefunction(fn):

        async def aexecute(args: dict) -> tuple[str, bool]:
            try:
                return await fn(args)
            except Exception as e:
                return f"{type(e).__name__}: {e}", True

        return aexecute

    def execute(args: dict) -> tuple[str, bool]:
        try:
            return fn(args)
        except Exception as e:
            return f"{type(e).__name__}: {e}", True

    return execute


async def _maybe_await(value: Any) -> Any:
    """Await `value` if it's awaitable, else return it as-is.

    Lets the runtime treat sync and async tool/hook bodies uniformly: a tool's
    `execute` may be a plain function or a coroutine function, and callers
    `await _maybe_await(tool.execute(args))` either way.
    """
    return await value if inspect.isawaitable(value) else value


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
        path = Path(args["path"])
        # Create missing parent directories so writing into a new directory
        # tree succeeds instead of raising FileNotFoundError.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"])
        return f"wrote {len(args['content'])} bytes to {args['path']}", False

    return Tool(
        name="write",
        description=(
            "Overwrite a file with the given content. Creates the file and any "
            "missing parent directories."
        ),
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
    async def execute(args: dict) -> tuple[str, bool]:
        # No default timeout: a `grep`/`find` over a large tree should run to
        # completion, not die at an arbitrary 60s cutoff. The model bounds a
        # command it expects to hang or run away by passing `timeout` (seconds).
        # stdin is closed (DEVNULL) so a command that would block on input
        # gets EOF and fails fast instead of hanging the agent forever.
        timeout = args.get("timeout") or None
        proc = await asyncio.create_subprocess_shell(
            args["cmd"],
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"bash timed out after {timeout}s", True
        text = out.decode(errors="replace") + err.decode(errors="replace")
        return text[:20_000] or "(no output)", proc.returncode != 0

    return Tool(
        name="bash",
        description=(
            "Run a shell command. Returns combined stdout+stderr (capped at "
            "20KB). Runs with stdin closed. No timeout by default; pass "
            "`timeout` (seconds) to bound a command that might hang or run away."
        ),
        schema={
            "type": "object",
            "properties": {
                "cmd": {"type": "string"},
                "timeout": {"type": "number", "minimum": 0},
            },
            "required": ["cmd"],
        },
        execute=execute,
    )


@dataclass(frozen=True)
class LspServer:
    """Spec for a language server: how to spawn it and which language it owns."""

    cmd: tuple[str, ...]
    language_id: str  # tool suffix + LSP languageId (e.g. "python", "rust")


class LspClient:
    """Minimal JSON-RPC client speaking LSP over a server's stdio.

    A reader task pumps framed messages off the server's stdout: responses
    resolve the future their request id is waiting on; pushed
    `publishDiagnostics` notifications update `_diagnostics` and wake any
    waiter; server-bound requests are acked with null so the server keeps
    making progress. Because each call awaits its own future, concurrent calls
    on one client are safe with no lock — that's what lets a turn's tool calls
    run in parallel.
    """

    def __init__(self, server: LspServer, root: Path) -> None:
        self.server = server
        self.root = root
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}  # request id -> result future
        self._opened: dict[Path, tuple[int, str]] = {}  # path -> (version, last_text)
        self._diagnostics: dict[str, list[dict]] = {}
        self._diag_event = asyncio.Event()  # set on every publishDiagnostics

    # --- transport ------------------------------------------------------

    async def _start(self) -> None:
        if self._proc is not None:
            return
        self._proc = await asyncio.create_subprocess_exec(
            *self.server.cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=2**21,  # roomy line/frame buffer for big hover/definition bodies
        )
        self._reader_task = asyncio.create_task(self._reader())
        await self._request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": self.root.as_uri(),
                "capabilities": {},
            },
        )
        await self._notify("initialized", {})

    async def _reader(self) -> None:
        out = self._proc.stdout if self._proc else None
        if out is None:
            return
        try:
            while True:
                length = 0
                while True:
                    line = await out.readline()
                    if not line:
                        return
                    if line in (b"\r\n", b"\n"):
                        break
                    if line.lower().startswith(b"content-length:"):
                        length = int(line.split(b":", 1)[1].strip())
                body = await out.readexactly(length)
                await self._dispatch(json.loads(body))
        except (asyncio.IncompleteReadError, asyncio.CancelledError):
            return
        except Exception as e:
            print(
                f"[lsp/{self.server.language_id}] reader exited: {e}",
                file=sys.stderr,
            )

    async def _dispatch(self, msg: dict) -> None:
        if msg.get("method") == "textDocument/publishDiagnostics":
            p = msg.get("params") or {}
            self._diagnostics[p.get("uri", "")] = p.get("diagnostics") or []
            self._diag_event.set()
        elif "id" in msg and "method" not in msg:  # response to one of our requests
            fut = self._pending.pop(msg["id"], None)
            if fut is not None and not fut.done():
                fut.set_result(msg.get("result"))
        elif "id" in msg and "method" in msg:  # server -> client request: ack null
            await self._notify(None, None, _id=msg["id"])

    async def _send(self, msg: dict) -> None:
        body = json.dumps(msg).encode()
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        await self._proc.stdin.drain()

    async def _request(self, method: str, params: dict, timeout: float = 10.0) -> Any:
        self._next_id += 1
        rid = self._next_id
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._send(
            {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
        )
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            return None

    async def _notify(self, method: str | None, params: dict | None, _id=None) -> None:
        msg: dict = {"jsonrpc": "2.0"}
        if _id is not None:  # null ack to a server-bound request
            msg["id"], msg["result"] = _id, None
        else:
            msg["method"], msg["params"] = method, params
        await self._send(msg)

    # --- LSP surface ----------------------------------------------------

    async def open(self, path: Path) -> None:
        """Open `path` on first call; on later calls, send didChange when the
        on-disk content has changed since we last told the server about it.

        Without the change-detection step, edits made between LSP calls are
        invisible to the server — diagnostics, hover, etc. keep reporting on
        the original buffer.
        """
        await self._start()
        text = path.read_text()
        state = self._opened.get(path)
        if state is None:
            self._opened[path] = (1, text)
            await self._notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": path.as_uri(),
                        "languageId": self.server.language_id,
                        "version": 1,
                        "text": text,
                    }
                },
            )
            return
        if state[1] == text:
            return
        version = state[0] + 1
        self._opened[path] = (version, text)
        self._diagnostics.pop(path.as_uri(), None)  # stale; the server will republish
        await self._notify(
            "textDocument/didChange",
            {
                "textDocument": {"uri": path.as_uri(), "version": version},
                "contentChanges": [{"text": text}],
            },
        )

    async def _at(
        self, method: str, path: Path, line: int, col: int, **extra: Any
    ) -> Any:
        await self.open(path)
        return await self._request(
            method,
            {
                "textDocument": {"uri": path.as_uri()},
                "position": {"line": line, "character": col},
                **extra,
            },
        )

    async def hover(self, path: Path, line: int, col: int) -> Any:
        return await self._at("textDocument/hover", path, line, col)

    async def definition(self, path: Path, line: int, col: int) -> Any:
        return await self._at("textDocument/definition", path, line, col)

    async def references(self, path: Path, line: int, col: int) -> Any:
        return await self._at(
            "textDocument/references",
            path,
            line,
            col,
            context={"includeDeclaration": False},
        )

    async def diagnostics(
        self, path: Path, wait: float = 5.0, settle: float = 0.3
    ) -> list[dict]:
        """Open/refresh `path`, then block until the server publishes
        diagnostics for it — returning once they settle, or after `wait`
        seconds.

        The wait is the whole game on a cold server: pyright and especially
        rust-analyzer publish nothing until their first analysis pass
        finishes, so a short fixed sleep returns `[]` — indistinguishable
        from a clean file — and the tool looks broken. Waiting for the first
        publish for this URI fixes that without slowing down a warm server.

        But the first publish isn't the last word: rust-analyzer emits an
        empty set the instant it opens a file, then the real diagnostics
        back-to-back once analysis lands. Returning on the first publish would
        hand back that placeholder. So after the first publish we keep waiting
        for follow-ups until `settle` seconds of quiet — bounded by the same
        `wait` deadline — and return the latest set.
        """
        uri = path.as_uri()
        await self.open(path)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait

        async def wait_event(timeout: float) -> bool:
            """Wait for the next publish; True if one arrived, False on timeout.
            `_diagnostics` is updated before the event fires, so the caller's
            condition check after a True is authoritative."""
            self._diag_event.clear()
            try:
                await asyncio.wait_for(self._diag_event.wait(), timeout)
                return True
            except asyncio.TimeoutError:
                return False

        while uri not in self._diagnostics and (left := deadline - loop.time()) > 0:
            if not await wait_event(left):
                break
        while uri in self._diagnostics and (left := deadline - loop.time()) > 0:
            if not await wait_event(min(settle, left)):
                break  # quiet for `settle` — the server has stopped revising
        return self._diagnostics.get(uri, [])

    async def shutdown(self) -> None:
        if self._proc is None:
            return
        try:
            await self._request("shutdown", {}, timeout=2.0)
            await self._notify("exit", {})
        except Exception:
            pass
        if self._reader_task is not None:
            self._reader_task.cancel()
        try:
            self._proc.terminate()
        except Exception:
            pass


def _fmt_hover(result: Any) -> str:
    if not result:
        return "(no hover info)"
    contents = result.get("contents")
    if isinstance(contents, str):
        return contents
    if isinstance(contents, dict):
        return contents.get("value") or ""
    if isinstance(contents, list):
        parts = [
            c if isinstance(c, str) else (c or {}).get("value", "") for c in contents
        ]
        return "\n\n".join(p for p in parts if p)
    return str(contents)


def _fmt_locations(result: Any) -> str:
    if not result:
        return "(no locations)"
    locs = result if isinstance(result, list) else [result]
    out: list[str] = []
    for loc in locs:
        uri = loc.get("uri") or loc.get("targetUri") or ""
        rng = loc.get("range") or loc.get("targetSelectionRange") or {}
        start = rng.get("start") or {}
        path = uri.removeprefix("file://")
        out.append(f"{path}:{start.get('line', 0) + 1}:{start.get('character', 0) + 1}")
    return "\n".join(out) or "(no locations)"


_LSP_SEVERITY = {1: "error", 2: "warn", 3: "info", 4: "hint"}


def _fmt_diagnostics(diags: list[dict]) -> str:
    if not diags:
        return "(no diagnostics)"
    lines = []
    for d in diags:
        start = (d.get("range") or {}).get("start") or {}
        sev = _LSP_SEVERITY.get(d.get("severity") or 0, "info")
        msg = d.get("message", "").replace("\n", " ")
        lines.append(f"{sev}\tline {start.get('line', 0) + 1}\t{msg}")
    return "\n".join(lines)


def _resolve_symbol(
    text: str, symbol: str, line_hint: int | None
) -> tuple[int, int] | None:
    """0-based (line, col) of the first word-boundary match of `symbol`.

    If `line_hint` (1-based) is given, search that line first, then fall back
    to a top-to-bottom scan. Identifiers match on a word boundary so `add`
    doesn't hit inside `address`; anything else (e.g. `foo.bar`) matches
    literally. Returns None when the symbol isn't found.
    """
    lines = text.splitlines()
    pat = re.compile(rf"\b{re.escape(symbol)}\b") if symbol.isidentifier() else None

    def find(i: int) -> tuple[int, int] | None:
        if not (0 <= i < len(lines)):
            return None
        if pat is not None:
            m = pat.search(lines[i])
            col = m.start() if m else -1
        else:
            col = lines[i].find(symbol)
        return (i, col) if col >= 0 else None

    if line_hint and (hit := find(line_hint - 1)):
        return hit
    for i in range(len(lines)):
        if hit := find(i):
            return hit
    return None


def _lsp_tool(server: LspServer, client: LspClient) -> Tool:
    @_tool_fn
    async def execute(args: dict) -> tuple[str, bool]:
        path = Path(args["path"]).resolve()
        op = args["operation"]
        if op == "diagnostics":
            return (
                _fmt_diagnostics(
                    await client.diagnostics(path, wait=float(args.get("wait", 5.0)))
                ),
                False,
            )
        if op not in ("hover", "definition", "references"):
            return f"unknown operation: {op}", True
        if "line" in args and "character" in args:  # precise position
            line = int(args["line"]) - 1
            col = int(args["character"]) - 1
        elif args.get("symbol"):  # by name — the tool finds the position
            resolved = _resolve_symbol(
                path.read_text(), args["symbol"], args.get("line")
            )
            if resolved is None:
                return f"symbol '{args['symbol']}' not found in {path}", True
            line, col = resolved
        else:
            return f"give `symbol` (or `line`+`character`) for `{op}`", True
        if op == "hover":
            return _fmt_hover(await client.hover(path, line, col)), False
        if op == "definition":
            return _fmt_locations(await client.definition(path, line, col)), False
        return _fmt_locations(await client.references(path, line, col)), False

    return Tool(
        name=f"lsp_{server.language_id}",
        description=(
            f"Query the {server.language_id} language server "
            f"(`{' '.join(server.cmd)}`). "
            "operation ∈ {hover, definition, references, diagnostics}. "
            "To locate a name, pass `symbol` and the tool finds its position — "
            "e.g. `definition path=… symbol=download_file`, or add a `line` to "
            "disambiguate a name used more than once. `line`+`character` "
            "(1-based, matching `read`'s line numbers) is an optional precise "
            "override; both are ignored for `diagnostics`. `wait` "
            "(diagnostics only) is the max seconds to wait for the server to "
            "publish — raise it on a large cold workspace still indexing."
        ),
        schema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["hover", "definition", "references", "diagnostics"],
                },
                "path": {"type": "string"},
                "symbol": {"type": "string"},
                "line": {"type": "integer", "minimum": 1},
                "character": {"type": "integer", "minimum": 1},
                "wait": {"type": "number", "minimum": 0},
            },
            "required": ["operation", "path"],
        },
        execute=execute,
    )


def _install_lsp(api: HookAPI, server: LspServer) -> None:
    """One client, one tool, one teardown — shared body for per-language hooks.

    No-op when `server.cmd[0]` isn't on PATH, so a missing language server
    doesn't pollute the tool surface with something that can only fail.
    """
    if shutil.which(server.cmd[0]) is None:
        return
    client = LspClient(server, Path.cwd())
    api.register_tool(_lsp_tool(server, client))

    @api.on("session_end")
    async def cleanup(_p: Any, _ctx: dict) -> None:
        await client.shutdown()


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
    max_turns: int = 50,
) -> None:
    tool_map = {t.name: t for t in tools}
    schemas = [t.to_anthropic() for t in tools]
    ctx = {"cwd": cwd, "session": session}

    for _ in range(max_turns):
        await runner.fire("turn_start", None, ctx)

        sp = await runner.fire("build_system_prompt", SystemPrompt(cwd=cwd), ctx)
        assert isinstance(sp, SystemPrompt)
        system: Any = sp.system_prompt
        for extra in sp.additional_context:
            system += f"\n\n{extra}"

        mr = ModelRequest(system=system, tools=schemas, messages=session.to_messages())
        await runner.fire("before_model_request", mr, ctx)
        max_tokens = mr.extra.pop("max_tokens", 64000)
        await runner.fire("model_request_prepared", mr, ctx)

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
                await runner.fire("text_delta", TextDelta(text=text), ctx)
                had_text = True
            if had_text:
                await runner.fire("text_end", None, ctx)
            try:
                res = await stream.get_final_message()
            except AssertionError:
                # No events accumulated: the endpoint returned something that
                # wasn't a valid SSE stream (e.g. an error body served with
                # HTTP 200). Surface what we know instead of a bare
                # AssertionError from deep inside the SDK.
                resp = stream.response
                raise RuntimeError(
                    "model stream produced no events "
                    f"(HTTP {resp.status_code}, content-type "
                    f"{resp.headers.get('content-type')!r}); the endpoint did "
                    "not return a valid SSE stream — check the model, params, "
                    "or proxy"
                ) from None

        dumped = res.model_dump(mode="json")
        assistant_content = dumped["content"]
        session.append("assistant", assistant_content)
        await runner.fire(
            "message_end",
            MessageEnd(message=assistant_content, usage=res.usage.model_dump()),
            ctx,
        )

        tool_uses = [b for b in res.content if b.type == "tool_use"]
        if not tool_uses:
            await runner.fire("stop", None, ctx)
            return

        tool_results = []

        # Phase 1 — pre_tool_use, sequential and ordered: these hooks may mutate
        # shared state (reminders, blocking), so they must not run concurrently.
        prepared: list[tuple[Any, PreTool]] = []
        for tu in tool_uses:
            pre = PreTool(id=tu.id, name=tu.name, input=tu.input, state={})
            await runner.fire("pre_tool_use", pre, ctx)
            pending_reminders.extend(pre.additional_context)
            prepared.append((tu, pre))

        # Phase 2 — execute, concurrent: only the I/O-bound `execute` bodies
        # overlap, so a turn's independent tool calls (lsp, bash, cargo) run in
        # parallel. Caveat: concurrent edit/write to the *same* path is unguarded
        # — the model would have to emit conflicting parallel writes (rare);
        # pi-mono's file-mutation-queue is the reference if it ever bites.
        async def run(pre: PreTool) -> tuple[str, bool]:
            if pre.blocked:
                return pre.reason or "blocked by hook", True
            tool = tool_map.get(pre.name)
            if tool is None:
                return f"unknown tool: {pre.name}", True
            return await _maybe_await(tool.execute(pre.input))

        outcomes = await asyncio.gather(*(run(pre) for _tu, pre in prepared))

        # Phase 3 — post_tool_use, sequential and ordered: results are appended in
        # the model's original tool_use order so ids line up turn to turn.
        for (tu, pre), (content, is_error) in zip(prepared, outcomes):
            post = PostTool(
                id=tu.id,
                name=tu.name,
                input=pre.input,
                content=content,
                is_error=is_error,
                state=pre.state,
            )
            await runner.fire("post_tool_use", post, ctx)
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
    max_turns: int = 50

    def _ctx(self) -> dict:
        return {"cwd": os.getcwd(), "session": self.session}

    async def start(self) -> None:
        p = await self.runner.fire(
            "session_start", SessionStart(cwd=os.getcwd()), self._ctx()
        )
        assert isinstance(p, SessionStart)
        self.pending_reminders.extend(p.additional_context)

    async def end(self) -> None:
        await self.runner.fire("session_end", None, self._ctx())

    async def prompt(self, text: str) -> None:
        p = await self.runner.fire(
            "user_prompt_submit", UserPrompt(prompt=text), self._ctx()
        )
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


def _paths_touched(name: str, args: dict, ext: str) -> list[Path]:
    """Return paths ending in ``.{ext}`` that this tool may have written.

    Knows ``edit``/``write`` (path arg) and ``bash`` (``>`` / ``>>`` redirect
    targets, including heredocs). Generic over extension so the language hooks
    in sibling extension files (rust.py, python.py) can reuse it.
    """
    if name in ("edit", "write"):
        path = args.get("path", "")
        if isinstance(path, str) and path.endswith(f".{ext}"):
            return [Path(path)]
        return []
    if name == "bash":
        cmd = args.get("cmd", "")
        if not isinstance(cmd, str):
            return []
        pattern = rf">>?\s*([^\s\'\"`;|&]+\.{re.escape(ext)})\b"
        return [Path(m) for m in re.findall(pattern, cmd)]
    return []


def system_prompt_hook(api: HookAPI) -> None:
    today = date.today().isoformat()

    @api.on("build_system_prompt")
    def build(p: SystemPrompt, _ctx: dict) -> None:
        has_lsp = any(t.name == "lsp_python" for t in api.runner.tools)
        lsp_tool = (
            " and lsp_python (semantic queries by symbol name: hover/definition/references, plus diagnostics)"
            if has_lsp
            else ""
        )
        lsp_rule = (
            "\n- To understand unfamiliar code, use `lsp_python` `definition`/`hover`/`references` by `symbol` (e.g. `references symbol=foo`) rather than guessing from a single `read`."
            "\n- After editing a `.py` file, run `lsp_python` `diagnostics` on it to catch errors before claiming done."
            if has_lsp
            else ""
        )
        p.system_prompt = f"""You are a Python coding assistant. Tools: read, write, edit, bash{lsp_tool}.

Rules:
- Always `read` a file before you `write` or `edit` it.
- Prefer `edit` for small changes. Only `write` for new files or full rewrites.
- If a tool errors, read the error and try again.
- Verify results with tools before claiming done (re-read the file after editing, run the test, check the exit code).{lsp_rule}
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


# The default model: applied by `model_flag_hook` when present, and fallen
# back to in `run()` when that hook has been removed (hooks are optional).
DEFAULT_MODEL = "us.anthropic.claude-opus-4-8"


def model_flag_hook(api: HookAPI) -> None:
    api.register_flag("--model", default=DEFAULT_MODEL)

    @api.on("build_session_config")
    def provide(p: SessionConfig, _ctx: dict) -> None:
        p.model = p.args.model


def max_turns_flag_hook(api: HookAPI) -> None:
    api.register_flag("--max-turns", type=int, default=50)

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
    """--effort LEVEL — sets output_config.effort (default 'xhigh').

    These are the effort levels the default model
    (us.anthropic.claude-opus-4-8) accepts. 'xhigh' is model-dependent:
    claude-sonnet-4-6, for one, only takes low/medium/high/max and 400s on
    'xhigh' — which some endpoints surface as a zero-event stream (a bare
    AssertionError out of the SDK) rather than a clean error.
    """
    _stream_extra_hook(
        api,
        lambda a: {"output_config": {"effort": a.effort}},
        "--effort",
        choices=("low", "medium", "high", "xhigh", "max"),
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


# Built-in hooks, loaded first. Language support lives in sibling extension
# files (rust.py, python.py, …) discovered by load_extensions() — not here.
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
# Extensions + entrypoint
# ═══════════════════════════════════════════════════════════════════════════


def load_extensions(runner: HookRunner, directory: Path | None = None) -> None:
    """Discover and load sibling extension files.

    An *extension* is any `.py` beside this one that exposes a module-level
    `HOOKS` tuple of `(HookAPI) -> None` callables — the same shape agent.py
    uses for its own built-ins. Drop a file in the directory and it loads on
    the next run; there is no registry to edit. Extensions import the toolkit
    they need (`Tool`, `LspServer`, `_install_lsp`, `_paths_touched`, …) from
    `agent`. The agent file itself and `test_*`/`conftest`/`_*` files are
    skipped; a broken extension is logged and skipped, never fatal.
    """
    directory = directory or Path(__file__).resolve().parent
    for path in sorted(directory.glob("*.py")):
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.stem.startswith(("_", "test_")) or path.stem == "conftest":
            continue
        try:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = module
            spec.loader.exec_module(module)
        except Exception:
            print(f"[extension {path.name} failed to import]", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            continue
        for hook in getattr(module, "HOOKS", ()):
            runner.load(hook)


async def main() -> None:
    runner = HookRunner()
    for hook in HOOKS:
        runner.load(hook)
    load_extensions(runner)

    args = runner.parse_args()
    ctx = {"args": args, "runner": runner}
    await runner.fire("args_parsed", ArgsParsed(args=args), ctx)

    cfg = await runner.fire("build_session_config", SessionConfig(args=args), ctx)
    assert isinstance(cfg, SessionConfig)
    sp = await runner.fire("before_session_load", SessionPath(args=args), ctx)
    assert isinstance(sp, SessionPath) and sp.path is not None

    prompt = await runner.prompter(args)
    if not prompt:
        return

    session = SessionManager(sp.path)
    agent = AgentSession(
        client=cfg.client,
        model=cfg.model or DEFAULT_MODEL,
        max_turns=cfg.max_turns,
        session=session,
        runner=runner,
    )

    await agent.start()
    try:
        await agent.prompt(prompt)
    finally:
        await agent.end()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
