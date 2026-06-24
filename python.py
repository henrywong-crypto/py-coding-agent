"""Python language extension for agent.py.

Drop-in: exposes a `HOOKS` tuple that agent.py's `load_extensions()`
auto-discovers. Provides the `lsp_python` tool (Pyright) and an auto-check
that runs diagnostics after every `.py` edit. Delete this file and the agent
loses Python-specific support — nothing else changes.
"""

from __future__ import annotations

from agent import HookAPI, LspServer, PostTool, _install_lsp, _maybe_await, _paths_touched


def python_lsp_hook(api: HookAPI) -> None:
    """Register the `lsp_python` tool, served by Pyright."""
    _install_lsp(
        api,
        LspServer(cmd=("pyright-langserver", "--stdio"), language_id="python"),
    )


def python_auto_check_hook(api: HookAPI) -> None:
    """After a successful tool call that mutates a `.py` file, run
    `lsp_python diagnostics` on it and attach the result to the next turn so
    the model fixes errors right away.

    No-op when the `lsp_python` tool isn't registered (Pyright not on PATH).
    Reuses agent's `_paths_touched` dispatch."""

    @api.on("post_tool_use", priority=70)
    async def run(p: PostTool, _ctx: dict) -> None:
        if p.is_error:
            return
        tool = next((t for t in api.runner.tools if t.name == "lsp_python"), None)
        if tool is None:
            return
        for path in _paths_touched(p.name, p.input or {}, "py"):
            body, _ = await _maybe_await(
                tool.execute({"operation": "diagnostics", "path": str(path)})
            )
            p.additional_context.append(
                f'<auto_check tool="lsp_python diagnostics" path="{path}">\n'
                f"{body}\n</auto_check>"
            )


HOOKS = (python_lsp_hook, python_auto_check_hook)
