"""Rust language extension for agent.py.

Drop-in: exposes a `HOOKS` tuple that agent.py's `load_extensions()`
auto-discovers. Provides a Rust system prompt + plan-then-execute workflow,
workspace context, the `lsp_rust` tool (rust-analyzer), a summarising `cargo`
tool, and an auto-`cargo check` after every `.rs` edit. Delete this file and
the agent loses Rust-specific support — nothing else changes.
"""

from __future__ import annotations

import asyncio
import tomllib
from datetime import date
from pathlib import Path

from agent import (
    HookAPI,
    LspServer,
    PostTool,
    SessionStart,
    SystemPrompt,
    Tool,
    _install_lsp,
    _paths_touched,
    _tool_fn,
)


def _rust_root(start: Path | str) -> tuple[Path, dict] | None:
    """Find the Cargo manifest that owns `start`. Returns `(dir, parsed_toml)`.

    Walks from `start` to filesystem root. Prefers a `[workspace]` ancestor
    over a `[package]` crate, so callers get one place both to issue cargo
    commands and to read workspace metadata.
    """
    cur = Path(start).resolve()
    workspace: tuple[Path, dict] | None = None
    crate: tuple[Path, dict] | None = None
    while True:
        try:
            data = tomllib.loads((cur / "Cargo.toml").read_text())
        except (OSError, ValueError):
            data = None
        if data is not None:
            if "workspace" in data:
                workspace = (cur, data)
            elif crate is None and "package" in data:
                crate = (cur, data)
        if cur == cur.parent:
            return workspace or crate
        cur = cur.parent


def _crate_name(file_path: Path, ws_root: Path) -> str | None:
    """First [package].name walking from `file_path` up to `ws_root`."""
    cur = file_path.resolve().parent
    while cur == ws_root or ws_root in cur.parents:
        try:
            data = tomllib.loads((cur / "Cargo.toml").read_text())
            name = data.get("package", {}).get("name")
            if isinstance(name, str):
                return name
        except (OSError, ValueError):
            pass
        if cur == cur.parent:
            break
        cur = cur.parent
    return None


def _summarise_cargo(stdout: str, stderr: str, code: int) -> str:
    """Boil cargo output to 'OK + warnings + tail' or 'FAILED + first 3 errors'.

    Tolerates both default-format (`error[Exxx]: …\\n  --> path`) and
    `--message-format=short` (`path:L:C: error[Exxx]: …`) diagnostics.
    """
    combined = (stdout + "\n" + stderr).strip()
    lines = combined.splitlines()

    def is_header(line: str, kind: str) -> bool:
        # Default format: "error[E0001]: …" or "warning: …" at start of line.
        # Short format:   "path:L:C: error[E0001]: …" / "… : warning: …".
        return line.startswith(kind) or f": {kind}" in line

    def blocks_of(kind: str) -> list[list[str]]:
        out: list[list[str]] = []
        cur: list[str] = []
        for line in lines:
            if is_header(line, kind):
                if cur:
                    out.append(cur)
                cur = [line]
            elif cur:
                cur.append(line)
        if cur:
            out.append(cur)
        return out

    if code == 0:
        warnings = blocks_of("warning")
        tail = [
            line
            for line in lines
            if line.startswith(("    Finished", "test result:")) or "passed" in line
        ]
        out: list[str] = ["OK"]
        for b in warnings[:3]:
            out.append("")
            out.extend(b[:25])
        if tail:
            out.append("")
            out.extend(tail[-5:])
        return "\n".join(out)

    blocks = blocks_of("error")
    head = f"FAILED — exit {code}, {len(blocks)} error(s)"
    out = [head]
    for b in blocks[:3]:
        out.append("")
        out.extend(b[:25])
    return "\n".join(out)


def rust_lsp_hook(api: HookAPI) -> None:
    """Register the `lsp_rust` tool, served by rust-analyzer."""
    _install_lsp(api, LspServer(cmd=("rust-analyzer",), language_id="rust"))


def rust_system_prompt_hook(api: HookAPI) -> None:
    """Override the default Python prompt when the cwd is a Rust workspace.
    Priority 60 runs after the built-in 50, so this wins when applicable."""

    @api.on("build_system_prompt", priority=60)
    def build(p: SystemPrompt, _ctx: dict) -> None:
        if _rust_root(p.cwd) is None:
            return
        p.system_prompt = f"""You are a Rust coding assistant. Tools: read, write, edit, bash, cargo (and lsp_rust for semantic queries by symbol name: hover/definition/references).

Workflow:
- Always `read` a file before `write`/`edit`. Lines from `read` carry "<n>\\t" prefixes — never include them in `old`/`new`.
- Prefer `edit` for surgical changes. `write` only for new files or full rewrites.
- To understand unfamiliar code, use `lsp_rust` `definition`/`hover`/`references` by `symbol` (e.g. `references symbol=foo`) rather than guessing from a single `read`.
- After every `.rs` `edit`/`write` the agent automatically runs `cargo check` for the affected crate and attaches the result — read it before deciding the next step.
- After meaningful changes run `cargo test -p <crate>`; before claiming done run `cargo clippy --all-targets`.
- Use `anyhow::{{Result, Context, anyhow, bail}}`; no `unwrap`/`expect` outside tests; no empty-value defaults.
- Renames and signature changes: find call sites with `lsp_rust references symbol=<name>` (semantic, survives renames; grep only for non-symbol text), edit each, then `cargo check` to confirm nothing dangles.
- Match patterns from sibling crates of the workspace before inventing new ones — convention reuse > novelty.
- Show, don't tell: prefer code/YAML over prose when explaining.

Current date: {date.today().isoformat()}
Current working directory: {p.cwd}
"""


def rust_workspace_hook(api: HookAPI) -> None:
    """At session start, surface the crate layout from Cargo.toml's
    `[workspace].members` so the model knows the workspace's shape."""

    @api.on("session_start", priority=60)
    def collect(p: SessionStart, _ctx: dict) -> None:
        root = _rust_root(p.cwd)
        if root is None:
            return
        _ws, data = root
        members = data.get("workspace", {}).get("members") or []
        if members:
            p.additional_context.append(
                "<workspace_members>\n  "
                + "\n  ".join(members)
                + "\n</workspace_members>"
            )


def cargo_tool_hook(api: HookAPI) -> None:
    """Register a structured `cargo` tool. Output is summarised to keep
    context cheap — much smaller than `bash cargo …`."""

    schema = {
        "type": "object",
        "properties": {
            "subcommand": {
                "type": "string",
                "enum": ["check", "build", "test", "clippy", "fmt", "run"],
                "description": "cargo subcommand",
            },
            "package": {
                "type": "string",
                "description": "scope to one crate (passed as `-p <package>`)",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "extra args appended after the subcommand",
            },
        },
        "required": ["subcommand"],
    }

    @_tool_fn
    async def execute(args: dict) -> tuple[str, bool]:
        sub = args["subcommand"]
        cmd = ["cargo", sub]
        if args.get("package"):
            cmd += ["-p", args["package"]]
        cmd += list(args.get("args", []))
        root = _rust_root(Path.cwd())
        cwd = str(root[0]) if root else None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return ("cargo not found on PATH", True)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=300)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return (f"`{' '.join(cmd)}` timed out after 300s", True)
        return (
            _summarise_cargo(
                out.decode(errors="replace"),
                err.decode(errors="replace"),
                proc.returncode,
            ),
            proc.returncode != 0,
        )

    api.register_tool(
        Tool(
            name="cargo",
            description=(
                "Run `cargo <subcommand>` (optionally `-p <package>`). Output is "
                "summarised to: 'OK + tail' on success, or 'FAILED, N errors' "
                "plus the first 3 error blocks on failure. Prefer this over "
                "`bash cargo …` — much cheaper on context."
            ),
            schema=schema,
            execute=execute,
        )
    )


def rust_auto_check_hook(api: HookAPI) -> None:
    """After any successful tool call that mutates a `.rs` file, auto-run
    `cargo check -p <crate>` and attach the summary to the next user turn so
    the model fixes errors immediately rather than waiting to be asked.

    Recognises `edit`/`write` (path arg) and `bash` redirects to `.rs` files
    (e.g. heredocs). Extend `_paths_touched` to teach it about new tools."""

    @api.on("post_tool_use", priority=70)
    async def run(p: PostTool, _ctx: dict) -> None:
        if p.is_error:
            return
        touched = _paths_touched(p.name, p.input or {}, "rs")
        if not touched:
            return
        root = _rust_root(Path.cwd())
        if root is None:
            return
        ws, _ = root
        # Dedup affected crates so a multi-file bash heredoc emits one
        # check per crate (None = workspace-wide).
        crates: list[str | None] = []
        for path in touched:
            crate = _crate_name(path, ws)
            if crate not in crates:
                crates.append(crate)
        for crate in crates:
            cmd = ["cargo", "check", "--message-format=short"]
            if crate:
                cmd += ["-p", crate]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(ws),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as e:
                p.additional_context.append(
                    f"<auto_check>{type(e).__name__}: {e}</auto_check>"
                )
                return
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=120)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                p.additional_context.append(
                    "<auto_check>TimeoutError: `cargo check` timed out "
                    "after 120s</auto_check>"
                )
                return
            body = _summarise_cargo(
                out.decode(errors="replace"),
                err.decode(errors="replace"),
                proc.returncode,
            )
            target = f" -p {crate}" if crate else ""
            p.additional_context.append(
                f'<auto_check cmd="cargo check{target}">\n{body}\n</auto_check>'
            )


def rust_plan_hook(api: HookAPI) -> None:
    """Append a plan-then-execute workflow rule to the Rust system prompt.
    The plan is the in-conversation contract — no file is written. For
    substantive work the model sketches design first, waits for approval,
    then implements against the agreed sketch."""

    @api.on("build_system_prompt", priority=65)
    def add(p: SystemPrompt, _ctx: dict) -> None:
        if _rust_root(p.cwd) is None or not p.system_prompt:
            return
        p.system_prompt += """
Plan-then-execute (substantive work only — new crate/module, multi-file refactor):
1. Sketch the design inline first: file tree (one line per file with a one-line comment), key types/signatures, public API surface, conventions worth pinning. Use code/yaml; avoid prose.
2. Pause for user feedback. When pushed back, propose 2-3 named alternatives with one-line tradeoffs rather than guessing.
3. Once stable, execute it — write modules in dependency order; the auto-`cargo check` closes the loop turn by turn.
4. The agreed sketch is the spec for the rest of the session. If the user changes the design, restate the affected slice of the sketch in the next reply before touching code.
"""


HOOKS = (
    rust_system_prompt_hook,
    rust_plan_hook,
    rust_workspace_hook,
    rust_lsp_hook,
    cargo_tool_hook,
    rust_auto_check_hook,
)
