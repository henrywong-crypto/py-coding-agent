# /// script
# requires-python = ">=3.11"
# dependencies = ["anthropic", "rich", "prompt_toolkit", "pytest", "pytest-asyncio", "hypothesis"]
# ///
"""
Tests for agent.py — unit + property-based coverage of the core.

Run:
    uv run test_agent.py

The fuzz suite uses Hypothesis to throw adversarial inputs at the hook
runner, tools, and session manager. It does *not* hit the Anthropic API —
the agent_loop's model interaction is excluded. Everything else is covered.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Make agent.py importable regardless of invocation cwd.
sys.path.insert(0, str(Path(__file__).parent))

from agent import (
    # tools (private factories used in tests)
    _bash_tool,
    _edit_tool,
    _fmt_diagnostics,
    _fmt_hover,
    _fmt_locations,
    _install_lsp,
    _lsp_tool,
    _maybe_await,
    _read_tool,
    _resolve_symbol,
    _session_dir,
    _stream_extra_hook,
    _write_tool,
    # agent session + runtime
    AgentSession,
    DEFAULT_MODEL,
    # payloads
    ArgsParsed,
    Event,
    HookAPI,
    HookRunner,
    LspClient,
    LspServer,
    MessageEnd,
    ModelRequest,
    Payload,
    PostTool,
    PreTool,
    SessionConfig,
    SessionEntry,
    SessionManager,
    SessionPath,
    SessionStart,
    SystemPrompt,
    TextDelta,
    Tool,
    UserPrompt,
    # hook functions
    anthropic_cache_hook,
    anthropic_client_hook,
    bash_tool_hook,
    cache_debug_hook,
    debug_hooks_flag_hook,
    edit_tool_hook,
    lifecycle_hook,
    list_sessions_hook,
    load_extensions,
    max_tokens_flag_hook,
    max_turns_flag_hook,
    model_flag_hook,
    output_effort_hook,
    prompt_arg_hook,
    prompt_toolkit_hook,
    read_tool_hook,
    resume_hook,
    session_history_hook,
    session_path_hook,
    skills_hook,
    strict_hooks_flag_hook,
    system_prompt_hook,
    thinking_hook,
    ui_hook,
    write_tool_hook,
    agent_loop,
)

# Language extensions — auto-discovered at runtime by agent.load_extensions();
# imported explicitly here so their hooks can be unit-tested in isolation.
from python import python_auto_check_hook, python_lsp_hook
from rust import (
    _crate_name,
    _rust_root,
    _summarise_cargo,
    cargo_tool_hook,
    rust_auto_check_hook,
    rust_lsp_hook,
    rust_system_prompt_hook,
)


def _fire(r, event, payload=None, ctx=None):
    """Sync bridge for tests: drive the now-async ``HookRunner.fire`` from a
    synchronous test body. Handlers under unit test are synchronous, so a fresh
    event loop per call is fine; async tests ``await r.fire(...)`` directly.
    """
    return asyncio.run(r.fire(event, payload, ctx))


def _exec(tool, args):
    """Sync bridge for a tool's ``execute``: runs it to completion, awaiting if
    the body is a coroutine (async tools like ``bash``/``lsp_*``/``cargo``). Lets
    a synchronous test call ``_exec(tool, {...})`` regardless of tool flavour;
    async tests ``await _maybe_await(tool.execute(...))`` directly.
    """
    return asyncio.run(_maybe_await(tool.execute(args)))


class _FakeProc:
    """Stand-in for an ``asyncio.subprocess.Process`` — lets cargo/auto-check
    tests stub ``asyncio.create_subprocess_exec`` without spawning anything."""

    def __init__(self, stdout=b"", stderr=b"", returncode=0):
        self._out, self._err, self.returncode = stdout, stderr, returncode

    async def communicate(self):
        return self._out, self._err

    def kill(self):
        pass

    async def wait(self):
        return self.returncode


# ═══════════════════════════════════════════════════════════════════════════
# Payloads
# ═══════════════════════════════════════════════════════════════════════════


class TestPayload:
    def test_base_defaults(self):
        p = Payload()
        assert p.blocked is False
        assert p.reason == ""

    def test_subclass_sets_blocked(self):
        p = UserPrompt(prompt="hi")
        p.blocked = True
        p.reason = "because"
        assert p.blocked and p.reason == "because"

    def test_kw_only_subclass_requires_keyword(self):
        # Positional instantiation must fail — all subclass fields are kw_only.
        with pytest.raises(TypeError):
            UserPrompt("hi")  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# HookRunner construction
# ═══════════════════════════════════════════════════════════════════════════


class TestRunnerInit:
    def test_standard_events_preloaded(self):
        expected = {
            "before_session_load",
            "args_parsed",
            "build_session_config",
            "session_start",
            "user_prompt_submit",
            "turn_start",
            "build_system_prompt",
            "before_model_request",
            "model_request_prepared",
            "text_delta",
            "text_end",
            "message_end",
            "pre_tool_use",
            "post_tool_use",
            "stop",
            "session_end",
        }
        assert set(HookRunner().events) == expected

    def test_events_in_lifecycle_order(self):
        names = list(HookRunner().events.keys())
        assert names[0] == "before_session_load"
        assert names[-1] == "session_end"

    def test_handlers_empty_per_event(self):
        r = HookRunner()
        for name in r.events:
            assert r.handlers[name] == []

    def test_no_tools_initially(self):
        assert HookRunner().tools == []


class TestHookAPI:
    def test_on_registers_handler(self):
        r = HookRunner()
        h = lambda p, c: None
        r.api.on("text_end", h)
        assert r.handlers["text_end"][0].fn is h
        assert r.handlers["text_end"][0].priority == 50

    def test_on_unknown_event_raises(self):
        with pytest.raises(ValueError, match="unknown event"):
            HookRunner().api.on("nope", lambda p, c: None)

    def test_register_event_adds(self):
        r = HookRunner()
        r.api.register_event("my_event", UserPrompt)
        assert r.events["my_event"].payload_cls is UserPrompt

    def test_register_event_without_payload_cls(self):
        r = HookRunner()
        r.api.register_event("signal")
        assert r.events["signal"].payload_cls is None

    def test_register_event_duplicate_raises(self):
        r = HookRunner()
        with pytest.raises(ValueError, match="already registered"):
            r.api.register_event("session_start", SessionStart)

    def test_register_tool_appends(self):
        r = HookRunner()
        t = _read_tool()
        r.api.register_tool(t)
        assert r.tools == [t]

    def test_register_tool_rejects_duplicate_name(self):
        r = HookRunner()
        r.api.register_tool(_read_tool())
        with pytest.raises(ValueError, match="already registered"):
            r.api.register_tool(_read_tool())

    def test_register_tool_distinct_coexist(self):
        r = HookRunner()
        for t in (_read_tool(), _write_tool(), _edit_tool(), _bash_tool()):
            r.api.register_tool(t)
        assert [t.name for t in r.tools] == ["read", "write", "edit", "bash"]

    def test_register_prompter(self):
        r = HookRunner()

        @r.api.prompter
        async def p(args):
            return "x"

        assert r.prompter is p

    def test_register_history_loader(self):
        r = HookRunner()

        @r.api.history_loader
        def loader():
            return ["a"]

        assert r.history_loader is loader


class TestLoad:
    def test_load_invokes_hook(self):
        r = HookRunner()
        calls = []
        r.load(lambda api: calls.append(api))
        assert len(calls) == 1 and isinstance(calls[0], HookAPI)

    def test_load_failure_isolated(self, capsys):
        def broken(_api):
            raise RuntimeError("oops")

        HookRunner().load(broken)
        assert "broken" in capsys.readouterr().err


class TestDescribe:
    def test_lists_events(self):
        out = HookRunner().describe()
        for name in ("session_start", "pre_tool_use", "session_end"):
            assert name in out

    def test_shows_handler_count(self):
        r = HookRunner()
        r.api.on("text_end", lambda p, c: None)
        r.api.on("text_end", lambda p, c: None)
        assert "2 handler(s)" in r.describe()


# ═══════════════════════════════════════════════════════════════════════════
# fire() — payload mutation + priority + block
# ═══════════════════════════════════════════════════════════════════════════


class TestFire:
    def test_no_handlers_returns_payload_unchanged(self):
        p = UserPrompt(prompt="x")
        out = _fire(HookRunner(), "user_prompt_submit", p)
        assert out is p

    def test_none_payload_ok_for_signal_event(self):
        r = HookRunner()
        hits = []
        r.api.on("turn_start", lambda p, c: hits.append(p))
        assert _fire(r, "turn_start") is None
        assert hits == [None]

    def test_handler_mutates_payload(self):
        r = HookRunner()
        r.api.on("build_system_prompt", lambda p, c: setattr(p, "system_prompt", "X"))
        p = SystemPrompt(cwd="/")
        assert _fire(r, "build_system_prompt", p).system_prompt == "X"  # type: ignore[union-attr]

    def test_handlers_run_in_insertion_order_by_default(self):
        r = HookRunner()

        def one(p, c):
            p.additional_context.append("a")

        def two(p, c):
            p.additional_context.append("b")

        def three(p, c):
            p.additional_context.append("c")

        r.api.on("build_system_prompt", one)
        r.api.on("build_system_prompt", two)
        r.api.on("build_system_prompt", three)
        p = SystemPrompt(cwd="/")
        _fire(r, "build_system_prompt", p)
        assert p.additional_context == ["a", "b", "c"]

    def test_priority_orders_handlers(self):
        r = HookRunner()
        r.api.on(
            "build_system_prompt",
            lambda p, c: p.additional_context.append("late"),
            priority=90,
        )
        r.api.on(
            "build_system_prompt",
            lambda p, c: p.additional_context.append("early"),
            priority=10,
        )
        r.api.on(
            "build_system_prompt", lambda p, c: p.additional_context.append("mid")
        )  # 50
        p = SystemPrompt(cwd="/")
        _fire(r, "build_system_prompt", p)
        assert p.additional_context == ["early", "mid", "late"]

    def test_same_priority_preserves_insertion_order(self):
        """sorted() is stable — equal-priority handlers keep registration order."""
        r = HookRunner()
        for letter in "abcde":
            r.api.on(
                "build_system_prompt",
                lambda p, c, v=letter: p.additional_context.append(v),
                priority=50,
            )
        p = SystemPrompt(cwd="/")
        _fire(r, "build_system_prompt", p)
        assert p.additional_context == list("abcde")

    def test_block_short_circuits(self):
        r = HookRunner()
        calls: list[int] = []

        def first(p, c):
            calls.append(1)
            p.blocked = True
            p.reason = "stop"

        def second(p, c):
            calls.append(2)

        r.api.on("user_prompt_submit", first)
        r.api.on("user_prompt_submit", second)
        p = UserPrompt(prompt="hi")
        _fire(r, "user_prompt_submit", p)
        assert p.blocked and p.reason == "stop"
        assert calls == [1]

    def test_block_requires_truthy(self):
        r = HookRunner()
        ran = []
        r.api.on("user_prompt_submit", lambda p, c: None)  # no-op
        r.api.on("user_prompt_submit", lambda p, c: ran.append(1))
        p = UserPrompt(prompt="hi")
        _fire(r, "user_prompt_submit", p)
        assert ran == [1]
        assert p.blocked is False

    def test_handler_exception_isolated_by_default(self, capsys):
        r = HookRunner()
        r.api.on("text_end", lambda p, c: 1 / 0)
        ran = []
        r.api.on("text_end", lambda p, c: ran.append(1))
        _fire(r, "text_end")
        assert ran == [1]
        assert "error" in capsys.readouterr().err.lower()

    def test_fire_unknown_event_raises(self):
        with pytest.raises(ValueError, match="unknown event"):
            _fire(HookRunner(), "nope")

    def test_returns_same_payload_instance(self):
        r = HookRunner()
        r.api.on("user_prompt_submit", lambda p, c: p.additional_context.append("x"))
        p = UserPrompt(prompt="hi")
        out = _fire(r, "user_prompt_submit", p)
        assert out is p


class TestStrictMode:
    def test_default_is_non_strict(self):
        assert HookRunner().strict is False

    def test_non_strict_swallows(self, capsys):
        r = HookRunner()
        ran = []
        r.api.on("text_end", lambda p, c: 1 / 0)
        r.api.on("text_end", lambda p, c: ran.append(1))
        _fire(r, "text_end")
        assert ran == [1]
        assert "error" in capsys.readouterr().err.lower()

    def test_strict_reraises(self):
        r = HookRunner()
        r.strict = True
        r.api.on("text_end", lambda p, c: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            _fire(r, "text_end")

    def test_strict_preserves_exception_type_and_message(self):
        class Boom(Exception):
            pass

        r = HookRunner()
        r.strict = True

        def raiser(p, c):
            raise Boom("specific")

        r.api.on("text_end", raiser)
        with pytest.raises(Boom, match="specific"):
            _fire(r, "text_end")

    def test_strict_stops_at_first_raise(self):
        r = HookRunner()
        r.strict = True
        calls = []
        r.api.on("text_end", lambda p, c: calls.append(1))
        r.api.on("text_end", lambda p, c: 1 / 0)
        r.api.on("text_end", lambda p, c: calls.append(3))
        with pytest.raises(ZeroDivisionError):
            _fire(r, "text_end")
        assert calls == [1]

    def test_strict_toggleable_at_runtime(self):
        r = HookRunner()
        r.api.on("text_end", lambda p, c: 1 / 0)
        _fire(r, "text_end")  # swallowed
        r.strict = True
        with pytest.raises(ZeroDivisionError):
            _fire(r, "text_end")

    def test_strict_does_not_affect_block(self):
        """block short-circuit is a normal control flow, not an exception."""
        r = HookRunner()
        r.strict = True
        r.api.on(
            "user_prompt_submit",
            lambda p, c: (setattr(p, "blocked", True), setattr(p, "reason", "no")),
        )
        p = UserPrompt(prompt="x")
        _fire(r, "user_prompt_submit", p)
        assert p.blocked


# ═══════════════════════════════════════════════════════════════════════════
# Tools (unchanged)
# ═══════════════════════════════════════════════════════════════════════════


class TestReadTool:
    def test_reads_existing(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello")
        content, err = _read_tool().execute({"path": str(f)})
        assert content == "1\thello" and err is False

    def test_multiple_lines(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("one\ntwo\nthree")
        content, err = _read_tool().execute({"path": str(f)})
        assert err is False
        assert content == "1\tone\n2\ttwo\n3\tthree"

    def test_offset_and_limit(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("\n".join(str(i) for i in range(1, 11)))
        content, err = _read_tool().execute({"path": str(f), "offset": 3, "limit": 2})
        assert err is False
        assert content.startswith("4\t4\n5\t5")
        assert "5 more lines" in content
        assert "offset=5" in content

    def test_empty_file(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("")
        content, err = _read_tool().execute({"path": str(f)})
        assert err is False and content == "(empty)"

    def test_missing_file_returns_error(self, tmp_path):
        content, err = _read_tool().execute({"path": str(tmp_path / "nope")})
        assert err is True
        assert "FileNotFoundError" in content

    def test_line_number_padding(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("\n".join("x" for _ in range(12)))
        content, _ = _read_tool().execute({"path": str(f)})
        assert content.splitlines()[0] == " 1\tx"
        assert content.splitlines()[-1] == "12\tx"

    def test_byte_cap_truncates(self, tmp_path):
        f = tmp_path / "big.txt"
        line = "x" * 1000
        f.write_text("\n".join([line] * 100))  # ~100KB
        content, err = _read_tool().execute({"path": str(f), "limit": 100})
        assert err is False
        assert "more lines" in content


class TestWriteTool:
    def test_write_creates_file(self, tmp_path):
        f = tmp_path / "new.txt"
        content, err = _write_tool().execute({"path": str(f), "content": "hi"})
        assert err is False
        assert f.read_text() == "hi"
        assert "wrote 2 bytes" in content

    def test_write_overwrites(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("old")
        _write_tool().execute({"path": str(f), "content": "new"})
        assert f.read_text() == "new"

    def test_write_creates_missing_parent_dirs(self, tmp_path):
        # Regression: writing into a not-yet-existing directory tree used to
        # raise FileNotFoundError. The tool now mkdir -p's the parents.
        f = tmp_path / "a" / "b" / "c" / "new.txt"
        content, err = _write_tool().execute({"path": str(f), "content": "hi"})
        assert err is False
        assert f.read_text() == "hi"


class TestEditTool:
    def test_single_occurrence_replaced(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello world")
        content, err = _edit_tool().execute(
            {"path": str(f), "old": "world", "new": "there"}
        )
        assert err is False
        assert f.read_text() == "hello there"

    def test_multiple_occurrences_fail(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hi hi")
        content, err = _edit_tool().execute({"path": str(f), "old": "hi", "new": "x"})
        assert err is True
        assert "exactly once" in content

    def test_zero_occurrences_fail(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello")
        _, err = _edit_tool().execute({"path": str(f), "old": "zzz", "new": "x"})
        assert err is True


class TestBashTool:
    def test_simple_command(self):
        content, err = _exec(_bash_tool(), {"cmd": "echo hi"})
        assert err is False and "hi" in content

    def test_nonzero_exit_is_error(self):
        _, err = _exec(_bash_tool(), {"cmd": "false"})
        assert err is True

    def test_no_output_placeholder(self):
        content, _ = _exec(_bash_tool(), {"cmd": "true"})
        assert content == "(no output)"

    def test_timeout_param_bounds_runaway_command(self):
        # A slow command dies only when the model passes an explicit timeout.
        content, err = _exec(_bash_tool(), {"cmd": "sleep 5", "timeout": 0.3})
        assert err is True
        assert "timed out after 0.3s" in content

    def test_no_timeout_by_default(self):
        # Default schema/behavior: no timeout key, command runs to completion.
        assert "timeout" not in _bash_tool().schema["required"]
        content, err = _exec(_bash_tool(), {"cmd": "sleep 0.2 && echo done"})
        assert err is False and "done" in content

    def test_stdin_is_closed(self):
        # stdin=DEVNULL: a command that reads stdin gets EOF and exits instead
        # of hanging the agent. With stdin inherited this could block forever.
        content, err = _exec(_bash_tool(), {"cmd": "cat"})
        assert err is False and content == "(no output)"


# ═══════════════════════════════════════════════════════════════════════════
# SessionManager
# ═══════════════════════════════════════════════════════════════════════════


class TestSessionManager:
    def test_creates_fresh_file_with_header(self, tmp_path):
        p = tmp_path / "s.jsonl"
        sm = SessionManager(p)
        assert p.exists()
        header = json.loads(p.read_text().splitlines()[0])
        assert header["type"] == "header"
        assert header["version"] == 1

    def test_append_persists(self, tmp_path):
        p = tmp_path / "s.jsonl"
        sm = SessionManager(p)
        sm.append("user", "hello")
        sm.append("assistant", [{"type": "text", "text": "hi"}])
        lines = p.read_text().splitlines()
        assert len(lines) == 3  # header + 2 entries

    def test_reload_restores_entries(self, tmp_path):
        p = tmp_path / "s.jsonl"
        SessionManager(p).append("user", "one")
        sm2 = SessionManager(p)
        assert len(sm2.entries) == 1
        assert sm2.entries[0].role == "user"
        assert sm2.entries[0].content == "one"

    def test_tool_result_becomes_user_in_to_messages(self, tmp_path):
        sm = SessionManager(tmp_path / "s.jsonl")
        sm.append("user", "q")
        sm.append("tool_result", [{"type": "tool_result", "content": "x"}])
        msgs = sm.to_messages()
        assert [m["role"] for m in msgs] == ["user", "user"]

    def test_parent_id_links(self, tmp_path):
        sm = SessionManager(tmp_path / "s.jsonl")
        e1 = sm.append("user", "a")
        e2 = sm.append("assistant", "b")
        assert e1.parent_id is None
        assert e2.parent_id == e1.id


# ═══════════════════════════════════════════════════════════════════════════
# Built-in hooks
# ═══════════════════════════════════════════════════════════════════════════


class TestSystemPromptHook:
    def test_builds_default_prompt(self):
        r = HookRunner()
        r.load(system_prompt_hook)
        p = SystemPrompt(cwd="/tmp/work")
        _fire(r, "build_system_prompt", p)
        assert "coding assistant" in p.system_prompt
        assert "/tmp/work" in p.system_prompt

    def test_date_captured_at_load(self, monkeypatch):
        import agent

        fake = SimpleNamespace(
            today=lambda: SimpleNamespace(isoformat=lambda: "2026-05-11")
        )
        monkeypatch.setattr(agent, "date", fake)
        r = HookRunner()
        r.load(system_prompt_hook)
        # Change the clock after load — prompt must not change.
        monkeypatch.setattr(
            agent,
            "date",
            SimpleNamespace(
                today=lambda: SimpleNamespace(isoformat=lambda: "2026-05-12")
            ),
        )
        p = SystemPrompt(cwd="/")
        _fire(r, "build_system_prompt", p)
        assert "Current date: 2026-05-11" in p.system_prompt
        assert "2026-05-12" not in p.system_prompt

    def test_stable_across_fires(self):
        r = HookRunner()
        r.load(system_prompt_hook)
        a = SystemPrompt(cwd="/tmp")
        b = SystemPrompt(cwd="/tmp")
        _fire(r, "build_system_prompt", a)
        _fire(r, "build_system_prompt", b)
        assert a.system_prompt == b.system_prompt

    def test_advertises_lsp_python_when_registered(self, monkeypatch):
        """The prompt names lsp_python only when the tool is actually
        registered — it never promises a tool that isn't there."""
        import agent

        monkeypatch.setattr(agent.shutil, "which", lambda cmd: f"/fake/{cmd}")
        r = HookRunner()
        r.load(python_lsp_hook)  # registers lsp_python (binary faked present)
        r.load(system_prompt_hook)
        p = SystemPrompt(cwd="/tmp/work")
        _fire(r, "build_system_prompt", p)
        assert "lsp_python" in p.system_prompt

    def test_omits_lsp_python_when_absent(self):
        r = HookRunner()
        r.load(system_prompt_hook)  # no lsp tool registered
        p = SystemPrompt(cwd="/tmp/work")
        _fire(r, "build_system_prompt", p)
        assert "lsp_python" not in p.system_prompt


class TestSkillsHook:
    def _write_skill(self, path: Path, name: str, description: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nname: {name}\ndescription: {description}\n---\nbody\n")

    def test_no_skills_leaves_context_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        (tmp_path / "home").mkdir()
        (tmp_path / "proj").mkdir()
        r = HookRunner()
        r.load(skills_hook)
        p = SystemPrompt(cwd=str(tmp_path / "proj"))
        _fire(r, "build_system_prompt", p)
        assert p.additional_context == []

    def test_project_skill_surfaced(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"
        proj.mkdir()
        self._write_skill(
            proj / ".agent/skills/hello/SKILL.md", "hello", "Greet someone."
        )
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        (tmp_path / "home").mkdir()
        r = HookRunner()
        r.load(skills_hook)
        p = SystemPrompt(cwd=str(proj))
        _fire(r, "build_system_prompt", p)
        block = "\n".join(p.additional_context)
        assert "<name>hello</name>" in block
        assert "Greet someone." in block

    def test_user_wins_over_project(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        proj = tmp_path / "proj"
        self._write_skill(home / ".py-agent/skills/dup/SKILL.md", "dup", "User.")
        self._write_skill(proj / ".agent/skills/dup/SKILL.md", "dup", "Project.")
        monkeypatch.setenv("HOME", str(home))
        r = HookRunner()
        r.load(skills_hook)
        p = SystemPrompt(cwd=str(proj))
        _fire(r, "build_system_prompt", p)
        block = "\n".join(p.additional_context)
        assert "User." in block
        assert "Project." not in block

    def test_ignores_dot_and_node_modules(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"
        self._write_skill(proj / ".agent/skills/.hidden/SKILL.md", "h", "nope.")
        self._write_skill(proj / ".agent/skills/node_modules/x/SKILL.md", "x", "nope.")
        self._write_skill(proj / ".agent/skills/ok/SKILL.md", "ok", "yep.")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        (tmp_path / "home").mkdir()
        r = HookRunner()
        r.load(skills_hook)
        p = SystemPrompt(cwd=str(proj))
        _fire(r, "build_system_prompt", p)
        block = "\n".join(p.additional_context)
        assert "<name>ok</name>" in block
        assert "hidden" not in block and "node_modules" not in block

    def test_composes_with_system_prompt_hook(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"
        self._write_skill(proj / ".agent/skills/hello/SKILL.md", "hello", "Greet.")
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        (tmp_path / "home").mkdir()
        r = HookRunner()
        r.load(system_prompt_hook)
        r.load(skills_hook)
        r.api.on(
            "build_system_prompt", lambda p, c: p.additional_context.append("extra")
        )
        p = SystemPrompt(cwd=str(proj))
        _fire(r, "build_system_prompt", p)
        assert "coding assistant" in p.system_prompt
        joined = "\n".join(p.additional_context)
        assert "<name>hello</name>" in joined
        assert "extra" in joined


class TestToolHooks:
    @pytest.mark.parametrize(
        "hook,name",
        [
            (read_tool_hook, "read"),
            (write_tool_hook, "write"),
            (edit_tool_hook, "edit"),
            (bash_tool_hook, "bash"),
            (python_lsp_hook, "lsp_python"),
            (rust_lsp_hook, "lsp_rust"),
        ],
    )
    def test_registers_tool(self, hook, name):
        r = HookRunner()
        r.load(hook)
        assert name in [t.name for t in r.tools]


class TestResumeHook:
    def _args(self, *, new=False, session=None):
        return argparse.Namespace(new=new, session=session)

    def _setup(self, tmp_path, monkeypatch, create=True):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        d = _session_dir(str(tmp_path))
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def test_default_picks_most_recent(self, tmp_path, monkeypatch):
        d = self._setup(tmp_path, monkeypatch)
        (d / "s1.jsonl").write_text('{"type":"header"}\n')
        f2 = d / "s2.jsonl"
        f2.write_text('{"type":"header"}\n')
        r = HookRunner()
        r.load(resume_hook)
        p = SessionPath(args=self._args())
        _fire(r, "before_session_load", p)
        assert p.path == f2

    def test_new_flag_skips_resume(self, tmp_path, monkeypatch):
        d = self._setup(tmp_path, monkeypatch)
        (d / "old.jsonl").write_text('{"type":"header"}\n')
        r = HookRunner()
        r.load(resume_hook)
        p = SessionPath(args=self._args(new=True))
        _fire(r, "before_session_load", p)
        assert p.path is None

    def test_session_flag_picks_explicit(self, tmp_path):
        f = tmp_path / "specific.jsonl"
        f.write_text('{"type":"header"}\n')
        r = HookRunner()
        r.load(resume_hook)
        p = SessionPath(args=self._args(session=f))
        _fire(r, "before_session_load", p)
        assert p.path == f


class TestSessionPathHook:
    def test_provides_default_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        r = HookRunner()
        r.load(session_path_hook)
        p = SessionPath(args=argparse.Namespace())
        _fire(r, "before_session_load", p)
        assert p.path is not None
        assert p.path.suffix == ".jsonl"

    def test_default_priority_yields_to_resume(self, tmp_path, monkeypatch):
        """resume_hook (default priority 50) must run BEFORE session_path_hook's
        default (priority 90), so an explicit resume wins."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        r = HookRunner()
        r.load(session_path_hook)
        r.load(resume_hook)
        explicit = tmp_path / "mine.jsonl"
        explicit.write_text('{"type":"header"}\n')
        p = SessionPath(args=argparse.Namespace(new=False, session=explicit))
        _fire(r, "before_session_load", p)
        assert p.path == explicit


class TestFlagHooks:
    def test_model_flag_defaults(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(model_flag_hook)
        args = r.parser.parse_args([])
        p = SessionConfig(args=args)
        _fire(r, "build_session_config", p)
        assert p.model == DEFAULT_MODEL

    def test_model_flag_overridden(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(model_flag_hook)
        args = r.parser.parse_args(["--model", "claude-opus-4"])
        p = SessionConfig(args=args)
        _fire(r, "build_session_config", p)
        assert p.model == "claude-opus-4"

    def test_max_turns_flag(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(max_turns_flag_hook)
        args = r.parser.parse_args(["--max-turns", "7"])
        p = SessionConfig(args=args)
        _fire(r, "build_session_config", p)
        assert p.max_turns == 7


class TestStreamExtraHooks:
    """--max-tokens, --thinking-display, --effort all go through
    _stream_extra_hook which stashes args in args_parsed and injects into
    ModelRequest.extra in before_model_request."""

    def test_max_tokens_injects_into_extra(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(max_tokens_flag_hook)
        args = r.parser.parse_args(["--max-tokens", "8000"])
        _fire(r, "args_parsed", ArgsParsed(args=args))
        p = ModelRequest(system="", tools=[], messages=[])
        _fire(r, "before_model_request", p)
        assert p.extra["max_tokens"] == 8000

    def test_thinking_default_summarized(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(thinking_hook)
        args = r.parser.parse_args([])
        _fire(r, "args_parsed", ArgsParsed(args=args))
        p = ModelRequest(system="", tools=[], messages=[])
        _fire(r, "before_model_request", p)
        assert p.extra["thinking"] == {"type": "adaptive", "display": "summarized"}

    def test_thinking_override(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(thinking_hook)
        args = r.parser.parse_args(["--thinking-display", "hidden"])
        _fire(r, "args_parsed", ArgsParsed(args=args))
        p = ModelRequest(system="", tools=[], messages=[])
        _fire(r, "before_model_request", p)
        assert p.extra["thinking"]["display"] == "hidden"

    def test_effort_default_xhigh(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(output_effort_hook)
        args = r.parser.parse_args([])
        _fire(r, "args_parsed", ArgsParsed(args=args))
        p = ModelRequest(system="", tools=[], messages=[])
        _fire(r, "before_model_request", p)
        assert p.extra["output_config"] == {"effort": "xhigh"}

    def test_effort_invalid_choice_rejected(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(output_effort_hook)
        with pytest.raises(SystemExit):
            r.parser.parse_args(["--effort", "insane"])

    def test_max_tokens_does_nothing_without_args_parsed(self):
        """If args_parsed didn't fire, the hook has no args cached — inject is a no-op."""
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(max_tokens_flag_hook)
        p = ModelRequest(system="", tools=[], messages=[])
        _fire(r, "before_model_request", p)
        assert p.extra == {}


class TestStrictHooksFlag:
    def test_sets_strict_when_flag_given(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(strict_hooks_flag_hook)
        args = r.parser.parse_args(["--strict-hooks"])
        _fire(r, "args_parsed", ArgsParsed(args=args))
        assert r.strict is True

    def test_strict_off_by_default(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(strict_hooks_flag_hook)
        args = r.parser.parse_args([])
        _fire(r, "args_parsed", ArgsParsed(args=args))
        assert r.strict is False


class TestDebugHooksFlag:
    def test_prints_when_enabled(self, capsys):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(debug_hooks_flag_hook)
        args = r.parser.parse_args(["--debug-hooks"])
        _fire(r, "args_parsed", ArgsParsed(args=args), {"runner": r})
        assert "before_session_load" in capsys.readouterr().err

    def test_silent_when_disabled(self, capsys):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(debug_hooks_flag_hook)
        args = r.parser.parse_args([])
        _fire(r, "args_parsed", ArgsParsed(args=args), {"runner": r})
        assert capsys.readouterr().err == ""


class TestAnthropicClientHook:
    def test_provides_client(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        r = HookRunner()
        r.load(anthropic_client_hook)
        p = SessionConfig(args=argparse.Namespace())
        _fire(r, "build_session_config", p)
        assert p.client is not None


class TestAnthropicCacheHook:
    """anthropic_cache_hook tags the final system block and last message with
    ephemeral cache_control so Anthropic caches the prefix."""

    def _run_hook(self, system, messages):
        r = HookRunner()
        r.load(anthropic_cache_hook)
        p = ModelRequest(system=system, tools=[], messages=messages)
        _fire(r, "before_model_request", p)
        return p

    def test_wraps_string_system(self):
        p = self._run_hook("hello", [{"role": "user", "content": "hi"}])
        assert isinstance(p.system, list)
        assert p.system[0]["type"] == "text"
        assert p.system[0]["cache_control"] == {"type": "ephemeral"}

    def test_marks_last_message(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        p = self._run_hook("sys", msgs)
        last = p.messages[-1]
        content = last["content"]
        assert isinstance(content, list)
        assert content[-1]["cache_control"] == {"type": "ephemeral"}

    def test_preserves_earlier_messages_unmodified(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        p = self._run_hook("sys", msgs)
        # First message is untouched
        assert p.messages[0] == {"role": "user", "content": "first"}

    def test_no_messages_no_crash(self):
        p = self._run_hook("sys", [])
        assert p.messages == []

    def test_list_system_marks_last_block(self):
        sys_blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        p = self._run_hook(sys_blocks, [{"role": "user", "content": "x"}])
        assert "cache_control" not in p.system[0]
        assert p.system[1]["cache_control"] == {"type": "ephemeral"}


class TestCacheDebugHook:
    def test_silent_by_default(self, capsys):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(cache_debug_hook)
        args = r.parser.parse_args([])
        _fire(r, "args_parsed", ArgsParsed(args=args))
        p = ModelRequest(
            system="sys", tools=[], messages=[{"role": "user", "content": "hi"}]
        )
        _fire(r, "model_request_prepared", p)
        assert "cache-debug" not in capsys.readouterr().err

    def test_prints_when_enabled(self, capsys):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(cache_debug_hook)
        args = r.parser.parse_args(["--debug-cache"])
        _fire(r, "args_parsed", ArgsParsed(args=args))
        p = ModelRequest(
            system="sys", tools=[], messages=[{"role": "user", "content": "hi"}]
        )
        _fire(r, "model_request_prepared", p)
        out = capsys.readouterr().err
        assert "cache-debug" in out
        assert "bytes: system=" in out


class TestSessionHistoryHook:
    def test_no_dir_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        r = HookRunner()
        r.load(session_history_hook)
        assert r.history_loader() == []  # type: ignore[misc]

    def test_collects_user_prompts_in_order(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        d = _session_dir(str(tmp_path))
        d.mkdir(parents=True)
        for name, prompt in [("s1.jsonl", "one"), ("s2.jsonl", "two")]:
            path = d / name
            path.write_text(
                '{"type":"header"}\n'
                + json.dumps(
                    {
                        "type": "entry",
                        "id": "1",
                        "parentId": None,
                        "role": "user",
                        "content": prompt,
                    }
                )
                + "\n"
            )
        r = HookRunner()
        r.load(session_history_hook)
        out = r.history_loader()  # type: ignore[misc]
        assert "one" in out and "two" in out

    def test_skips_non_user_and_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        d = _session_dir(str(tmp_path))
        d.mkdir(parents=True)
        path = d / "s.jsonl"
        lines = [
            '{"type":"header"}',
            json.dumps(
                {
                    "type": "entry",
                    "id": "1",
                    "parentId": None,
                    "role": "assistant",
                    "content": "x",
                }
            ),
            json.dumps(
                {
                    "type": "entry",
                    "id": "2",
                    "parentId": None,
                    "role": "user",
                    "content": "",
                }
            ),
            json.dumps(
                {
                    "type": "entry",
                    "id": "3",
                    "parentId": None,
                    "role": "user",
                    "content": "keep",
                }
            ),
        ]
        path.write_text("\n".join(lines) + "\n")
        r = HookRunner()
        r.load(session_history_hook)
        out = r.history_loader()  # type: ignore[misc]
        assert out == ["keep"]


class TestListSessionsHook:
    def test_exits_with_no_sessions(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(list_sessions_hook)
        args = r.parser.parse_args(["--list-sessions"])
        with pytest.raises(SystemExit) as exc:
            _fire(r, "args_parsed", ArgsParsed(args=args), {"runner": r})
        assert exc.value.code == 0
        assert "no sessions" in capsys.readouterr().out

    def test_does_nothing_without_flag(self, capsys):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(list_sessions_hook)
        args = r.parser.parse_args([])
        _fire(r, "args_parsed", ArgsParsed(args=args), {"runner": r})
        assert capsys.readouterr().out == ""


class TestPromptToolkitHook:
    def test_registers_prompter(self):
        r = HookRunner()
        r.load(prompt_toolkit_hook)
        assert r.prompter is not None

    @pytest.mark.asyncio
    async def test_prompter_consumes_args(self):
        r = HookRunner()
        r.load(prompt_toolkit_hook)
        result = await r.prompter(argparse.Namespace(prompt=["hello", "world"]))  # type: ignore[misc]
        assert result == "hello world"


# ═══════════════════════════════════════════════════════════════════════════
# UI hook — exercises state threading + rendering without a real console
# ═══════════════════════════════════════════════════════════════════════════


class TestUIHook:
    def test_text_delta_accumulates_then_prints_on_end(self, capsys):
        r = HookRunner()
        r.load(ui_hook)
        _fire(r, "text_delta", TextDelta(text="hello "))
        _fire(r, "text_delta", TextDelta(text="world"))
        assert capsys.readouterr().out == ""  # buffered, no print yet
        _fire(r, "text_end")
        out = capsys.readouterr().out
        assert "hello world" in out

    def test_message_end_prints_cache_stats(self, capsys):
        r = HookRunner()
        r.load(ui_hook)
        usage = {
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 50,
            "input_tokens": 25,
        }
        _fire(r, "message_end", MessageEnd(message=[], usage=usage))
        out = capsys.readouterr().out
        assert "read=100" in out and "write=50" in out and "input=25" in out

    def test_pre_post_tool_use_default_rendering(self, capsys):
        r = HookRunner()
        r.load(read_tool_hook)
        r.load(ui_hook)
        _fire(r, "pre_tool_use", PreTool(id="t1", name="read", input={"path": "/foo"}))
        _fire(r, 
            "post_tool_use",
            PostTool(
                id="t1",
                name="read",
                input={"path": "/foo"},
                content="line1\nline2",
                is_error=False,
            ),
        )
        out = capsys.readouterr().out
        assert "Read" in out
        assert "line1" in out


# ═══════════════════════════════════════════════════════════════════════════
# ToolState — pre_tool_use -> post_tool_use carries per-call state
# ═══════════════════════════════════════════════════════════════════════════


class TestToolState:
    def test_state_threaded_to_post(self):
        """Edit tool's render_call stashes 'pre' into state; render_result reads it."""
        r = HookRunner()
        captured = {}

        def pre_observer(p: PreTool, _c):
            p.state["snapshot"] = p.input.get("path")

        def post_observer(p: PostTool, _c):
            captured["state"] = dict(p.state)

        r.api.on("pre_tool_use", pre_observer)
        r.api.on("post_tool_use", post_observer)

        pre = PreTool(id="x", name="read", input={"path": "/foo"})
        _fire(r, "pre_tool_use", pre)
        post = PostTool(
            id="x",
            name="read",
            input={"path": "/foo"},
            content="ok",
            is_error=False,
            state=pre.state,
        )
        _fire(r, "post_tool_use", post)

        assert captured["state"] == {"snapshot": "/foo"}


# ═══════════════════════════════════════════════════════════════════════════
# AgentSession lifecycle
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentSessionLifecycle:
    def _make(self, tmp_path):
        client = MagicMock()
        sm = SessionManager(tmp_path / "s.jsonl")
        return AgentSession(client=client, model="m", session=sm)

    def test_start_fires_session_start(self, tmp_path):
        a = self._make(tmp_path)
        hits = []
        a.runner.api.on("session_start", lambda p, c: hits.append(p))
        asyncio.run(a.start())
        assert len(hits) == 1 and isinstance(hits[0], SessionStart)

    def test_start_collects_additional_context(self, tmp_path):
        a = self._make(tmp_path)
        a.runner.api.on(
            "session_start", lambda p, c: p.additional_context.append("reminder!")
        )
        asyncio.run(a.start())
        assert "reminder!" in a.pending_reminders

    def test_end_fires_session_end(self, tmp_path):
        a = self._make(tmp_path)
        hits = []
        a.runner.api.on("session_end", lambda p, c: hits.append(True))
        asyncio.run(a.end())
        assert hits == [True]

    @pytest.mark.asyncio
    async def test_prompt_blocked_does_not_call_loop(self, tmp_path):
        a = self._make(tmp_path)
        a.runner.api.on(
            "user_prompt_submit",
            lambda p, c: (setattr(p, "blocked", True), setattr(p, "reason", "no")),
        )
        # If not blocked, this would try to hit the mock client, which we set
        # up to blow up if called.
        a.client.messages.stream = MagicMock(
            side_effect=AssertionError("should not be called")
        )
        await a.prompt("bad prompt")
        # Session has only the header + 0 entries — the user message wasn't
        # appended because the block short-circuited.
        assert a.session.entries == []


# ═══════════════════════════════════════════════════════════════════════════
# agent_loop — stream kwargs propagation (no real API call)
# ═══════════════════════════════════════════════════════════════════════════


class _FakeTextStream:
    def __init__(self, texts: list[str], tool_uses=None, usage=None):
        self._texts = texts
        self._tool_uses = tool_uses or []
        self._usage = usage or {"input_tokens": 1}

    @property
    def text_stream(self):
        async def gen():
            for t in self._texts:
                yield t

        return gen()

    async def get_final_message(self):
        content = [SimpleNamespace(type="text", text="".join(self._texts))]
        content += [
            SimpleNamespace(
                type="tool_use", id=tu["id"], name=tu["name"], input=tu["input"]
            )
            for tu in self._tool_uses
        ]
        return SimpleNamespace(
            content=content,
            model_dump=lambda mode="json": {
                "content": [{"type": "text", "text": "".join(self._texts)}]
            },
            usage=SimpleNamespace(model_dump=lambda: self._usage),
        )


class _FakeStreamCtx:
    def __init__(self, stream, captured_kwargs):
        self._stream = stream
        self._captured = captured_kwargs

    async def __aenter__(self):
        return self._stream

    async def __aexit__(self, *args):
        return False


class TestAgentLoopStreamKwargs:
    @pytest.mark.asyncio
    async def test_stream_receives_extra_kwargs(self, tmp_path):
        """before_model_request handlers' `extra` ends up as **kwargs to stream."""
        captured = {}

        def fake_stream(**kwargs):
            captured.update(kwargs)
            return _FakeStreamCtx(_FakeTextStream(["hi"]), captured)

        client = SimpleNamespace(messages=SimpleNamespace(stream=fake_stream))

        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(max_tokens_flag_hook)
        r.load(thinking_hook)
        args = r.parser.parse_args(["--max-tokens", "500"])
        await r.fire("args_parsed", ArgsParsed(args=args))

        sm = SessionManager(tmp_path / "s.jsonl")
        await agent_loop(client, "m", "/cwd", [], sm, r, [], max_turns=1)  # type: ignore[arg-type]

        assert captured["max_tokens"] == 500
        assert captured["thinking"] == {"type": "adaptive", "display": "summarized"}

    @pytest.mark.asyncio
    async def test_cache_hook_marks_system_before_stream(self, tmp_path):
        captured = {}

        def fake_stream(**kwargs):
            captured.update(kwargs)
            return _FakeStreamCtx(_FakeTextStream(["hi"]), captured)

        client = SimpleNamespace(messages=SimpleNamespace(stream=fake_stream))

        r = HookRunner()
        r.load(system_prompt_hook)
        r.load(anthropic_cache_hook)
        sm = SessionManager(tmp_path / "s.jsonl")
        sm.append("user", "hello")
        await agent_loop(client, "m", "/cwd", [], sm, r, [], max_turns=1)  # type: ignore[arg-type]

        assert isinstance(captured["system"], list)
        assert captured["system"][-1]["cache_control"] == {"type": "ephemeral"}


class TestAgentLoopConcurrency:
    """A turn's independent tool calls run via asyncio.gather, so I/O-bound
    tools overlap instead of running back-to-back."""

    @pytest.mark.asyncio
    async def test_independent_tool_calls_overlap(self, tmp_path):
        calls = 0

        def fake_stream(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:  # first turn: emit two tool_use blocks at once
                tus = [
                    {"id": "a", "name": "slow", "input": {}},
                    {"id": "b", "name": "slow", "input": {}},
                ]
                return _FakeStreamCtx(_FakeTextStream([], tool_uses=tus), {})
            return _FakeStreamCtx(_FakeTextStream(["done"]), {})  # then stop

        client = SimpleNamespace(messages=SimpleNamespace(stream=fake_stream))

        started: list[float] = []

        async def slow(args):
            started.append(asyncio.get_running_loop().time())
            await asyncio.sleep(0.3)
            return ("ok", False)

        r = HookRunner()
        r.api.register_tool(
            Tool(
                name="slow",
                description="",
                schema={"type": "object", "properties": {}},
                execute=slow,
            )
        )
        sm = SessionManager(tmp_path / "s.jsonl")
        sm.append("user", "go")

        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await agent_loop(client, "m", str(tmp_path), r.tools, sm, r, [], max_turns=4)
        elapsed = loop.time() - t0

        # Two 0.3s tools, concurrent → ~0.3s wall-clock, not ~0.6s.
        assert len(started) == 2
        assert elapsed < 0.5, f"tool calls did not overlap (took {elapsed:.2f}s)"


# ═══════════════════════════════════════════════════════════════════════════
# LSP — extension architecture (no real language server spawned)
# ═══════════════════════════════════════════════════════════════════════════


class TestLspServer:
    def test_is_frozen(self):
        s = LspServer(cmd=("x",), language_id="x")
        with pytest.raises(Exception):  # FrozenInstanceError subclass of TypeError
            s.cmd = ("y",)  # type: ignore[misc]

    def test_holds_fields(self):
        s = LspServer(
            cmd=("pyright-langserver", "--stdio"),
            language_id="python",
        )
        assert s.cmd == ("pyright-langserver", "--stdio")
        assert s.language_id == "python"


class TestLspFormatters:
    def test_hover_none(self):
        assert _fmt_hover(None) == "(no hover info)"

    def test_hover_string_contents(self):
        assert _fmt_hover({"contents": "doc"}) == "doc"

    def test_hover_dict_contents(self):
        assert (
            _fmt_hover({"contents": {"kind": "markdown", "value": "**x**"}}) == "**x**"
        )

    def test_hover_list_contents_mixed(self):
        result = {"contents": ["a", {"value": "b"}, ""]}
        assert _fmt_hover(result) == "a\n\nb"

    def test_locations_none(self):
        assert _fmt_locations(None) == "(no locations)"

    def test_locations_single_dict(self):
        loc = {"uri": "file:///foo.py", "range": {"start": {"line": 0, "character": 4}}}
        assert _fmt_locations(loc) == "/foo.py:1:5"

    def test_locations_list_with_target_uri(self):
        # textDocument/definition can return LocationLink with targetUri/targetSelectionRange
        locs = [
            {
                "targetUri": "file:///a.rs",
                "targetSelectionRange": {"start": {"line": 9, "character": 0}},
            },
            {"uri": "file:///b.rs", "range": {"start": {"line": 0, "character": 0}}},
        ]
        out = _fmt_locations(locs).splitlines()
        assert out == ["/a.rs:10:1", "/b.rs:1:1"]

    def test_diagnostics_empty(self):
        assert _fmt_diagnostics([]) == "(no diagnostics)"

    def test_diagnostics_severity_mapping(self):
        diags = [
            {"severity": 1, "range": {"start": {"line": 4}}, "message": "boom"},
            {"severity": 2, "range": {"start": {"line": 9}}, "message": "warn here"},
            {"severity": 4, "range": {"start": {"line": 0}}, "message": "hint"},
        ]
        out = _fmt_diagnostics(diags).splitlines()
        assert out[0] == "error\tline 5\tboom"
        assert out[1] == "warn\tline 10\twarn here"
        assert out[2] == "hint\tline 1\thint"

    def test_diagnostics_collapses_newlines(self):
        diags = [{"severity": 1, "range": {"start": {"line": 0}}, "message": "a\nb"}]
        out = _fmt_diagnostics(diags)
        assert out == "error\tline 1\ta b"


class TestResolveSymbol:
    def test_first_occurrence(self):
        text = "def add(a, b):\n    return a + b\n"
        assert _resolve_symbol(text, "add", None) == (0, 4)

    def test_word_boundary_skips_substring(self):
        # `add` must not match inside `address`
        text = "address = 1\nadd = 2\n"
        assert _resolve_symbol(text, "add", None) == (1, 0)

    def test_line_hint_disambiguates(self):
        text = "x = 1\nx = 2\nx = 3\n"
        assert _resolve_symbol(text, "x", 2) == (1, 0)

    def test_line_hint_miss_falls_back(self):
        text = "a = 1\nb = 2\ntarget = 3\n"
        assert _resolve_symbol(text, "target", 1) == (2, 0)

    def test_non_identifier_matches_literally(self):
        assert _resolve_symbol("foo.bar()\n", "foo.bar", None) == (0, 0)

    def test_not_found_returns_none(self):
        assert _resolve_symbol("nothing here\n", "missing", None) is None


class TestLspTool:
    def _server(self, language_id="python"):
        return LspServer(cmd=("fake",), language_id=language_id)

    def test_name_derived_from_language_id(self):
        assert _lsp_tool(self._server("python"), MagicMock()).name == "lsp_python"
        assert _lsp_tool(self._server("rust"), MagicMock()).name == "lsp_rust"

    def test_schema_required_fields(self):
        t = _lsp_tool(self._server(), MagicMock())
        assert set(t.schema["required"]) == {"operation", "path"}
        op = t.schema["properties"]["operation"]
        assert set(op["enum"]) == {"hover", "definition", "references", "diagnostics"}

    def test_hover_dispatches_zero_based(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("hello")
        client = AsyncMock()
        client.hover.return_value = {"contents": "doc"}
        t = _lsp_tool(self._server(), client)
        out, err = _exec(
            t, {"operation": "hover", "path": str(f), "line": 3, "character": 7}
        )
        assert err is False and out == "doc"
        # 1-based input → 0-based passed downstream
        args, _ = client.hover.call_args
        assert args[1] == 2 and args[2] == 6

    def test_definition_dispatches(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("x")
        client = AsyncMock()
        client.definition.return_value = []
        t = _lsp_tool(self._server(), client)
        out, _ = _exec(
            t, {"operation": "definition", "path": str(f), "line": 1, "character": 1}
        )
        assert out == "(no locations)"
        client.definition.assert_called_once()

    def test_references_dispatches(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("x")
        client = AsyncMock()
        client.references.return_value = None
        t = _lsp_tool(self._server(), client)
        _exec(
            t, {"operation": "references", "path": str(f), "line": 1, "character": 1}
        )
        client.references.assert_called_once()

    def test_symbol_resolves_position(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("def add(a, b):\n    return a + b\n")
        client = AsyncMock()
        client.definition.return_value = []
        t = _lsp_tool(self._server(), client)
        _exec(t, {"operation": "definition", "path": str(f), "symbol": "add"})
        args, _ = client.definition.call_args
        assert args[1:] == (0, 4)  # 0-based (line, col) of `add`

    def test_symbol_with_line_hint(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("x = 1\nx = 2\n")
        client = AsyncMock()
        client.references.return_value = None
        t = _lsp_tool(self._server(), client)
        _exec(
            t, {"operation": "references", "path": str(f), "symbol": "x", "line": 2}
        )
        args, _ = client.references.call_args
        assert args[1:] == (1, 0)  # the hinted line wins over the first match

    def test_symbol_not_found_errors(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("a = 1\n")
        t = _lsp_tool(self._server(), MagicMock())
        out, err = _exec(
            t, {"operation": "hover", "path": str(f), "symbol": "missing"}
        )
        assert err is True and "not found" in out

    def test_requires_symbol_or_position(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("a = 1\n")
        t = _lsp_tool(self._server(), MagicMock())
        out, err = _exec(t, {"operation": "hover", "path": str(f)})
        assert err is True and "symbol" in out

    def test_diagnostics_ignores_position(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("x")
        client = AsyncMock()
        client.diagnostics.return_value = []
        t = _lsp_tool(self._server(), client)
        out, _ = _exec(t, {"operation": "diagnostics", "path": str(f)})
        assert out == "(no diagnostics)"
        client.diagnostics.assert_called_once()
        client.hover.assert_not_called()

    def test_unknown_operation_returns_error(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("x")
        t = _lsp_tool(self._server(), MagicMock())
        out, err = _exec(t, {"operation": "yolo", "path": str(f)})
        assert err is True and "unknown" in out


class TestLspHookInstallation:
    """Per-language hooks register a tool and a session_end teardown.

    These tests never spawn a server: `LspClient.__init__` doesn't fork the
    process (`_start` is lazy), and `shutdown()` is a no-op when `_proc is
    None`, so loading + firing session_end is fully offline. We monkeypatch
    `shutil.which` so registration is deterministic regardless of whether
    pyright/rust-analyzer are installed on the test host.
    """

    @pytest.fixture(autouse=True)
    def _binaries_present(self, monkeypatch):
        import agent

        monkeypatch.setattr(agent.shutil, "which", lambda cmd: f"/fake/{cmd}")

    def test_python_and_rust_coexist(self):
        r = HookRunner()
        r.load(python_lsp_hook)
        r.load(rust_lsp_hook)
        assert sorted(t.name for t in r.tools) == ["lsp_python", "lsp_rust"]

    def test_loading_same_language_twice_collides(self):
        r = HookRunner()
        r.load(python_lsp_hook)
        with pytest.raises(ValueError, match="already registered"):
            python_lsp_hook(r.api)  # bypass r.load()'s exception swallow

    def test_session_end_invokes_shutdown(self, monkeypatch):
        """The session_end handler registered by `_install_lsp` calls client.shutdown()."""
        import agent

        fake = MagicMock()
        fake.shutdown = AsyncMock()  # cleanup handler awaits client.shutdown()
        monkeypatch.setattr(agent, "LspClient", lambda *a, **kw: fake)
        r = HookRunner()
        r.load(python_lsp_hook)
        fake.shutdown.assert_not_called()
        _fire(r, "session_end")
        fake.shutdown.assert_called_once()

    def test_install_lsp_is_extensible(self, monkeypatch):
        """A custom server registered via `_install_lsp` produces `lsp_<language_id>`."""
        import agent

        monkeypatch.setattr(agent, "LspClient", lambda *a, **kw: MagicMock())
        r = HookRunner()
        _install_lsp(r.api, LspServer(cmd=("gopls",), language_id="go"))
        assert "lsp_go" in [t.name for t in r.tools]

    def test_skipped_when_binary_missing(self, monkeypatch):
        """If `shutil.which` returns None, the tool is silently not registered."""
        import agent

        monkeypatch.setattr(agent.shutil, "which", lambda cmd: None)
        r = HookRunner()
        r.load(python_lsp_hook)
        r.load(rust_lsp_hook)
        assert r.tools == []
        # No session_end handler registered either.
        assert r.handlers["session_end"] == []


class TestLspClientDidChange:
    """`open()` must send didChange when the file changes between calls.

    Captures messages by monkeypatching `_send` and `_start` so no subprocess
    is spawned. Without this behavior, edits made between LSP calls are
    invisible to the server (the original demo bug: diagnostics returned
    stale results after introducing a typo).
    """

    def _client(self, tmp_path, monkeypatch):
        c = LspClient(
            LspServer(cmd=("fake",), language_id="python"),
            tmp_path,
        )
        sent: list[dict] = []
        # Async no-op transport: `open` awaits `_start` and `_send`, so the
        # stand-ins must be awaitable. `_send` records each framed message.
        monkeypatch.setattr(c, "_start", AsyncMock())
        monkeypatch.setattr(c, "_send", AsyncMock(side_effect=sent.append))
        return c, sent

    def test_first_open_sends_did_open(self, tmp_path, monkeypatch):
        f = tmp_path / "x.py"
        f.write_text("hello")
        c, sent = self._client(tmp_path, monkeypatch)
        asyncio.run(c.open(f))
        assert len(sent) == 1
        assert sent[0]["method"] == "textDocument/didOpen"
        assert sent[0]["params"]["textDocument"]["text"] == "hello"
        assert sent[0]["params"]["textDocument"]["version"] == 1

    def test_unchanged_content_is_noop(self, tmp_path, monkeypatch):
        f = tmp_path / "x.py"
        f.write_text("hello")
        c, sent = self._client(tmp_path, monkeypatch)
        asyncio.run(c.open(f))
        asyncio.run(c.open(f))
        assert len(sent) == 1  # no second message

    def test_changed_content_sends_did_change(self, tmp_path, monkeypatch):
        f = tmp_path / "x.py"
        f.write_text("v1")
        c, sent = self._client(tmp_path, monkeypatch)
        asyncio.run(c.open(f))
        f.write_text("v2")
        asyncio.run(c.open(f))
        assert [m["method"] for m in sent] == [
            "textDocument/didOpen",
            "textDocument/didChange",
        ]
        change = sent[1]["params"]
        assert change["textDocument"]["version"] == 2
        assert change["contentChanges"] == [{"text": "v2"}]

    def test_change_invalidates_stale_diagnostics(self, tmp_path, monkeypatch):
        f = tmp_path / "x.py"
        f.write_text("v1")
        c, _ = self._client(tmp_path, monkeypatch)
        asyncio.run(c.open(f))
        c._diagnostics[f.as_uri()] = [{"severity": 1, "message": "stale"}]
        f.write_text("v2")
        asyncio.run(c.open(f))
        assert f.as_uri() not in c._diagnostics

    def test_version_increments_on_each_change(self, tmp_path, monkeypatch):
        f = tmp_path / "x.py"
        f.write_text("v1")
        c, sent = self._client(tmp_path, monkeypatch)
        asyncio.run(c.open(f))
        for i in range(2, 5):
            f.write_text(f"v{i}")
            asyncio.run(c.open(f))
        versions = [m["params"]["textDocument"]["version"] for m in sent]
        assert versions == [1, 2, 3, 4]


# ═══════════════════════════════════════════════════════════════════════════
# Rust workflow — pure helpers + hooks, no real cargo invoked
# ═══════════════════════════════════════════════════════════════════════════


class TestRustRoot:
    def test_workspace_takes_priority_over_member_crate(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text('[workspace]\nmembers = ["sub"]\n')
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "Cargo.toml").write_text('[package]\nname = "sub"\nversion = "0.1.0"\n')
        root = _rust_root(sub)
        assert root is not None
        assert root[0] == tmp_path
        assert root[1].get("workspace", {}).get("members") == ["sub"]

    def test_lone_crate_is_its_own_root(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text(
            '[package]\nname = "x"\nversion = "0.1.0"\n'
        )
        root = _rust_root(tmp_path)
        assert root is not None and root[0] == tmp_path

    def test_no_cargo_returns_none(self, tmp_path):
        assert _rust_root(tmp_path) is None

    def test_commented_workspace_does_not_match(self, tmp_path):
        # The old substring-based parser falsely matched on this; tomllib doesn't.
        (tmp_path / "Cargo.toml").write_text(
            "# [workspace] is reserved for later\n"
            '[package]\nname = "x"\nversion = "0.1.0"\n'
        )
        root = _rust_root(tmp_path)
        assert root is not None
        assert "workspace" not in root[1]

    def test_malformed_toml_is_ignored(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("not = valid = toml")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "Cargo.toml").write_text('[package]\nname = "sub"\nversion = "0.1.0"\n')
        root = _rust_root(sub)
        # Parent's broken Cargo.toml is skipped; we fall back to the inner crate.
        assert root is not None and root[0] == sub


class TestCrateName:
    def test_finds_crate_for_nested_file(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text('[workspace]\nmembers = ["a"]\n')
        crate = tmp_path / "a"
        (crate / "src").mkdir(parents=True)
        (crate / "Cargo.toml").write_text(
            '[package]\nname = "alpha"\nversion = "0.1.0"\n'
        )
        f = crate / "src" / "lib.rs"
        f.write_text("")
        assert _crate_name(f, tmp_path) == "alpha"

    def test_returns_none_for_file_outside_any_crate(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[workspace]\nmembers = []\n")
        f = tmp_path / "scratch.rs"
        f.write_text("")
        assert _crate_name(f, tmp_path) is None


class TestSummariseCargo:
    def test_success_keeps_finished_and_test_lines(self):
        out = (
            "    Compiling x v0.1.0\n"
            "    Finished `dev` profile [unoptimized] in 0.5s\n"
            "test result: ok. 5 passed; 0 failed\n"
        )
        s = _summarise_cargo(out, "", 0)
        assert s.startswith("OK\n")
        assert "Finished" in s and "5 passed" in s

    def test_failure_lists_first_three_errors(self):
        err = (
            "error[E0001]: foo\n  --> a.rs:1\n"
            "error[E0002]: bar\n  --> b.rs:2\n"
            "error[E0003]: baz\n  --> c.rs:3\n"
            "error[E0004]: qux\n  --> d.rs:4\n"
        )
        s = _summarise_cargo("", err, 1)
        assert s.startswith("FAILED — exit 1, 4 error(s)")
        assert "E0001" in s and "E0002" in s and "E0003" in s
        assert "E0004" not in s

    def test_success_surfaces_short_format_warning_diagnostics(self):
        # `cargo check --message-format=short` emits per-diagnostic lines as
        # `path:L:C: warning: …`. The previous filter dropped these and only
        # kept the rollup, which hid the actionable content from the model.
        out = (
            "src/lib.rs:7:5: warning: unused import: `anyhow::anyhow`\n"
            "warning: `chat` (lib) generated 1 warning\n"
            "    Finished `dev` profile [unoptimized] in 0.6s\n"
        )
        s = _summarise_cargo(out, "", 0)
        assert s.startswith("OK\n")
        assert "unused import" in s, s
        assert "anyhow::anyhow" in s, s
        assert "Finished" in s

    def test_failure_surfaces_short_format_error_diagnostics(self):
        # Symmetric to the warning case — short-format error headers are
        # `path:L:C: error[Exxx]: …` and must be recognised as block starts.
        err = (
            "src/a.rs:1:1: error[E0001]: undefined symbol\n"
            "src/b.rs:2:2: error[E0002]: type mismatch\n"
            "error: could not compile `chat` due to 2 previous errors\n"
        )
        s = _summarise_cargo("", err, 1)
        assert s.startswith("FAILED")
        assert "E0001" in s and "undefined symbol" in s
        assert "E0002" in s and "type mismatch" in s


class TestCargoToolHook:
    def test_registers_cargo_tool(self):
        r = HookRunner()
        r.load(cargo_tool_hook)
        assert "cargo" in [t.name for t in r.tools]

    def test_runs_at_workspace_root(self, tmp_path, monkeypatch):
        (tmp_path / "Cargo.toml").write_text(
            '[package]\nname = "x"\nversion = "0.1.0"\n'
        )
        captured = {}

        async def fake_exec(*cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["cwd"] = kwargs.get("cwd")
            return _FakeProc(stdout=b"    Finished `dev`")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        r = HookRunner()
        r.load(cargo_tool_hook)
        tool = next(t for t in r.tools if t.name == "cargo")
        out, err = _exec(tool, {"subcommand": "check", "package": "x"})
        assert err is False
        assert captured["cmd"] == ["cargo", "check", "-p", "x"]
        assert captured["cwd"] == str(tmp_path)

    def test_missing_cargo_returns_clean_error(self, monkeypatch):
        async def boom(*a, **kw):
            raise FileNotFoundError()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", boom)
        r = HookRunner()
        r.load(cargo_tool_hook)
        tool = next(t for t in r.tools if t.name == "cargo")
        out, err = _exec(tool, {"subcommand": "check"})
        assert err is True and "cargo not found" in out


class TestRustAutoCheckHook:
    def _post(self, name, path, is_error=False):
        return PostTool(
            id="t",
            name=name,
            input={"path": path},
            content="ok",
            is_error=is_error,
        )

    def test_skips_non_rust_files(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", lambda *a, **kw: called.append(a)
        )
        r = HookRunner()
        r.load(rust_auto_check_hook)
        _fire(r, "post_tool_use", self._post("edit", "/foo.py"))
        assert called == []

    def test_skips_when_outside_workspace(self, tmp_path, monkeypatch):
        called = []
        monkeypatch.chdir(tmp_path)  # no Cargo.toml here
        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", lambda *a, **kw: called.append(a)
        )
        r = HookRunner()
        r.load(rust_auto_check_hook)
        _fire(r, "post_tool_use", self._post("edit", str(tmp_path / "x.rs")))
        assert called == []

    def test_skips_failed_edits(self, tmp_path, monkeypatch):
        (tmp_path / "Cargo.toml").write_text(
            '[package]\nname = "x"\nversion = "0.1.0"\n'
        )
        monkeypatch.chdir(tmp_path)
        called = []
        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", lambda *a, **kw: called.append(a)
        )
        r = HookRunner()
        r.load(rust_auto_check_hook)
        p = self._post("edit", str(tmp_path / "src" / "main.rs"), is_error=True)
        _fire(r, "post_tool_use", p)
        assert called == []

    def test_attaches_summary_for_rust_edit(self, tmp_path, monkeypatch):
        (tmp_path / "Cargo.toml").write_text(
            '[package]\nname = "x"\nversion = "0.1.0"\n'
        )
        monkeypatch.chdir(tmp_path)

        async def fake_exec(*a, **kw):
            return _FakeProc(stdout=b"    Finished `dev`")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        r = HookRunner()
        r.load(rust_auto_check_hook)
        p = self._post("edit", str(tmp_path / "src" / "main.rs"))
        _fire(r, "post_tool_use", p)
        assert any("auto_check" in c for c in p.additional_context)

    def _bash_post(self, cmd):
        return PostTool(
            id="t",
            name="bash",
            input={"cmd": cmd},
            content="ok",
            is_error=False,
        )

    def test_triggers_on_bash_redirect_to_rs(self, tmp_path, monkeypatch):
        # Heredoc append (`cat >> foo.rs << EOF …`) was the workaround used
        # when `edit` rejected a non-unique `old`. Without this branch, no
        # auto_check fired and the model had to ask cargo manually.
        (tmp_path / "Cargo.toml").write_text(
            '[package]\nname = "x"\nversion = "0.1.0"\n'
        )
        monkeypatch.chdir(tmp_path)

        async def fake_exec(*a, **kw):
            return _FakeProc(stdout=b"    Finished `dev`")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        r = HookRunner()
        r.load(rust_auto_check_hook)
        p = self._bash_post("cat >> src/main.rs << 'EOF'\nfoo\nEOF")
        _fire(r, "post_tool_use", p)
        assert any("auto_check" in c for c in p.additional_context)

    def test_skips_bash_without_rs_write(self, tmp_path, monkeypatch):
        (tmp_path / "Cargo.toml").write_text(
            '[package]\nname = "x"\nversion = "0.1.0"\n'
        )
        monkeypatch.chdir(tmp_path)
        called = []
        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", lambda *a, **kw: called.append(a)
        )
        r = HookRunner()
        r.load(rust_auto_check_hook)
        # `grep` reads, doesn't write — must not trigger a check.
        _fire(r, "post_tool_use", self._bash_post("grep foo src/main.rs"))
        assert called == []

    def test_dedups_per_crate_for_multi_file_bash(self, tmp_path, monkeypatch):
        # Two redirects into the same crate must produce exactly one check,
        # not one per file.
        (tmp_path / "Cargo.toml").write_text(
            '[package]\nname = "x"\nversion = "0.1.0"\n'
        )
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)
        runs = []

        async def fake_exec(*cmd, **kw):
            runs.append(list(cmd))
            return _FakeProc(stdout=b"    Finished `dev`")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        r = HookRunner()
        r.load(rust_auto_check_hook)
        cmd = "cat > src/a.rs << 'EOF'\nfoo\nEOF\ncat > src/b.rs << 'EOF'\nbar\nEOF"
        _fire(r, "post_tool_use", self._bash_post(cmd))
        assert len(runs) == 1, runs


class TestPythonAutoCheckHook:
    """Mirror of TestRustAutoCheckHook for the Python side. Uses a stub
    `lsp_python` tool so no pyright subprocess is spawned."""

    def _post(self, name, path, is_error=False):
        return PostTool(
            id="t", name=name, input={"path": path}, content="ok", is_error=is_error
        )

    def _fake_lsp(self, recorder):
        def execute(args):
            recorder.append(args)
            return ("error\tline 1\tboom", False)

        return Tool(
            name="lsp_python",
            description="",
            schema={"type": "object", "properties": {}},
            execute=execute,
        )

    def test_attaches_diagnostics_for_py_edit(self):
        r = HookRunner()
        calls = []
        r.api.register_tool(self._fake_lsp(calls))
        r.load(python_auto_check_hook)
        p = self._post("edit", "/proj/m.py")
        _fire(r, "post_tool_use", p)
        assert calls == [{"operation": "diagnostics", "path": "/proj/m.py"}]
        assert any(
            "lsp_python diagnostics" in c and "boom" in c for c in p.additional_context
        )

    def test_noop_without_lsp_tool(self):
        """If pyright isn't installed, lsp_python is never registered — the
        hook must do nothing rather than error."""
        r = HookRunner()
        r.load(python_auto_check_hook)
        p = self._post("edit", "/proj/m.py")
        _fire(r, "post_tool_use", p)
        assert p.additional_context == []

    def test_skips_non_python_files(self):
        r = HookRunner()
        calls = []
        r.api.register_tool(self._fake_lsp(calls))
        r.load(python_auto_check_hook)
        _fire(r, "post_tool_use", self._post("edit", "/proj/main.rs"))
        assert calls == []

    def test_skips_failed_edits(self):
        r = HookRunner()
        calls = []
        r.api.register_tool(self._fake_lsp(calls))
        r.load(python_auto_check_hook)
        _fire(r, "post_tool_use", self._post("edit", "/proj/m.py", is_error=True))
        assert calls == []


class TestLspToolWaitArg:
    def _tool(self, client):
        return _lsp_tool(LspServer(cmd=("fake",), language_id="python"), client)

    def test_wait_passed_through(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("x")
        client = AsyncMock()
        client.diagnostics.return_value = []
        _exec(self._tool(client), {"operation": "diagnostics", "path": str(f), "wait": 9})
        assert client.diagnostics.call_args.kwargs["wait"] == 9.0

    def test_wait_defaults_to_5(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("x")
        client = AsyncMock()
        client.diagnostics.return_value = []
        _exec(self._tool(client), {"operation": "diagnostics", "path": str(f)})
        assert client.diagnostics.call_args.kwargs["wait"] == 5.0


class TestLspClientDiagnosticsWait:
    """diagnostics() returns the instant the server publishes for the URI,
    instead of always sleeping the full wait. Spawns nothing: _start/_send
    are stubbed and publishes are fed through the dispatch path directly."""

    def _client(self, tmp_path, monkeypatch):
        c = LspClient(LspServer(cmd=("fake",), language_id="python"), tmp_path)
        monkeypatch.setattr(c, "_start", AsyncMock())
        monkeypatch.setattr(c, "_send", AsyncMock())
        return c

    @staticmethod
    def _publish(uri, diags):
        return {
            "method": "textDocument/publishDiagnostics",
            "params": {"uri": uri, "diagnostics": diags},
        }

    def test_returns_cached_immediately(self, tmp_path, monkeypatch):
        f = tmp_path / "x.py"
        f.write_text("hi")
        c = self._client(tmp_path, monkeypatch)
        c._opened[f] = (1, "hi")  # open() is a no-op (content unchanged)
        c._diagnostics[f.as_uri()] = [{"severity": 1, "message": "boom"}]
        assert asyncio.run(c.diagnostics(f, wait=0.0)) == [
            {"severity": 1, "message": "boom"}
        ]

    def test_picks_up_publish(self, tmp_path, monkeypatch):
        f = tmp_path / "x.py"
        f.write_text("hi")
        c = self._client(tmp_path, monkeypatch)
        c._opened[f] = (1, "hi")
        uri = f.as_uri()

        async def go():
            await c._dispatch(self._publish(uri, [{"severity": 2, "message": "warn"}]))
            return await c.diagnostics(f, wait=2.0)

        assert asyncio.run(go()) == [{"severity": 2, "message": "warn"}]

    def test_times_out_to_empty(self, tmp_path, monkeypatch):
        f = tmp_path / "x.py"
        f.write_text("hi")
        c = self._client(tmp_path, monkeypatch)
        c._opened[f] = (1, "hi")  # nothing ever published
        assert asyncio.run(c.diagnostics(f, wait=0.05)) == []

    def test_settles_past_empty_placeholder(self, tmp_path, monkeypatch):
        """rust-analyzer emits an empty set then the real diagnostics
        back-to-back; diagnostics() must settle past the placeholder and
        return the real ones, not the empty first publish."""
        f = tmp_path / "x.py"
        f.write_text("hi")
        c = self._client(tmp_path, monkeypatch)
        c._opened[f] = (1, "hi")
        uri = f.as_uri()

        async def go():
            task = asyncio.create_task(c.diagnostics(f, wait=2.0, settle=0.2))
            await asyncio.sleep(0.02)
            await c._dispatch(self._publish(uri, []))  # empty placeholder first
            await asyncio.sleep(0.02)
            await c._dispatch(self._publish(uri, [{"severity": 1, "message": "boom"}]))
            return await task

        assert asyncio.run(go()) == [{"severity": 1, "message": "boom"}]


class TestRustSystemPromptHook:
    def test_overrides_inside_rust_workspace(self, tmp_path, monkeypatch):
        (tmp_path / "Cargo.toml").write_text("[workspace]\nmembers = []\n")
        monkeypatch.chdir(tmp_path)
        r = HookRunner()
        r.load(system_prompt_hook)  # default Python prompt at priority 50
        r.load(rust_system_prompt_hook)  # priority 60 — wins
        p = SystemPrompt(cwd=str(tmp_path))
        _fire(r, "build_system_prompt", p)
        assert "Rust coding assistant" in p.system_prompt
        assert "Python coding assistant" not in p.system_prompt

    def test_no_op_outside_rust(self, tmp_path):
        r = HookRunner()
        r.load(system_prompt_hook)
        r.load(rust_system_prompt_hook)
        p = SystemPrompt(cwd=str(tmp_path))
        _fire(r, "build_system_prompt", p)
        assert "Python coding assistant" in p.system_prompt


class TestLspToolPositionValidation:
    def test_missing_line_for_hover_returns_friendly_error(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("hello")
        tool = _lsp_tool(LspServer(cmd=("fake",), language_id="python"), MagicMock())
        out, err = _exec(tool, {"operation": "hover", "path": str(f)})
        assert err is True and "line" in out and "hover" in out


# ═══════════════════════════════════════════════════════════════════════════
# LSP integration — spawns the real servers, skips if a server is missing
# ═══════════════════════════════════════════════════════════════════════════


def _which(*candidates: str) -> str | None:
    """Return the first executable on PATH, or None if none are found."""
    import shutil

    return next((c for c in (shutil.which(name) for name in candidates) if c), None)


PYRIGHT = _which("pyright-langserver")
RUST_ANALYZER = _which("rust-analyzer")


@pytest.mark.skipif(PYRIGHT is None, reason="pyright-langserver not on PATH")
class TestPyrightIntegration:
    """End-to-end: real pyright subprocess, real .py file, real LSP roundtrips."""

    @pytest_asyncio.fixture
    async def client(self, tmp_path):
        c = LspClient(
            LspServer(
                cmd=("pyright-langserver", "--stdio"),
                language_id="python",
            ),
            tmp_path,
        )
        yield c
        await c.shutdown()

    @pytest.mark.asyncio
    async def test_hover_on_function_returns_signature(self, client, tmp_path):
        f = tmp_path / "m.py"
        f.write_text(
            "def greet(name: str) -> str:\n    return f'hi {name}'\n\ngreet('world')\n"
        )
        # Hover on `greet` at the call site (line 4, col 1 in 0-based).
        result = await client.hover(f, 3, 0)
        assert result is not None
        text = _fmt_hover(result)
        assert "greet" in text and "str" in text

    @pytest.mark.asyncio
    async def test_definition_jumps_to_function(self, client, tmp_path):
        f = tmp_path / "m.py"
        f.write_text(
            "def greet(name: str) -> str:\n    return name\n\ngreet('world')\n"
        )
        # Definition of `greet` at the call site.
        result = await client.definition(f, 3, 0)
        rendered = _fmt_locations(result)
        assert "m.py:1:" in rendered  # line 1 of the same file

    @pytest.mark.asyncio
    async def test_diagnostics_flag_undefined_name(self, client, tmp_path):
        f = tmp_path / "broken.py"
        f.write_text("print(no_such_name)\n")
        diags = await client.diagnostics(f)
        rendered = _fmt_diagnostics(diags)
        # pyright reports "no_such_name" as undefined
        assert "no_such_name" in rendered or "undefined" in rendered.lower()

    @pytest.mark.asyncio
    async def test_clean_file_has_no_diagnostics(self, client, tmp_path):
        f = tmp_path / "ok.py"
        f.write_text("x: int = 1\nprint(x)\n")
        diags = await client.diagnostics(f)
        # Should be empty or at most whitespace warnings — no errors.
        errors = [d for d in diags if d.get("severity") == 1]
        assert errors == []

    @pytest.mark.asyncio
    async def test_edit_after_open_is_picked_up(self, client, tmp_path):
        """Regression for the didChange bug: a file edited between LSP calls
        was invisible to the server because `open()` early-returned. After
        the fix, the second call sends didChange and pyright re-reports."""
        f = tmp_path / "m.py"
        f.write_text("x: int = 1\nprint(x)\n")
        # Initial open — clean buffer, no errors.
        clean = await client.diagnostics(f)
        assert [d for d in clean if d.get("severity") == 1] == []
        # Edit on disk; second call must send didChange + pick up the error.
        f.write_text("print(no_such_name)\n")
        broken = await client.diagnostics(f, wait=5.0)
        rendered = _fmt_diagnostics(broken)
        assert "no_such_name" in rendered or "undefined" in rendered.lower()

    @pytest.mark.asyncio
    async def test_tool_end_to_end(self, client, tmp_path):
        """The `lsp_python` tool surface: real spawn, real dispatch, real format.

        Exercises both addressing modes against a real server — including the
        symbol path that lets the model query by name without computing a
        column, and a `line` hint that disambiguates which occurrence to use.
        """
        f = tmp_path / "m.py"
        f.write_text(
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
            "\n"
            "result = add(1, 2)\n"
        )
        server = LspServer(cmd=("pyright-langserver", "--stdio"), language_id="python")
        tool = _lsp_tool(server, client)

        # Precise coordinates (back-compat): hover on `add` at its definition.
        precise, err = await _maybe_await(
            tool.execute(
                {"operation": "hover", "path": str(f), "line": 1, "character": 5}
            )
        )
        assert err is False and "add" in precise

        # Same hover addressed by symbol name — the tool finds the column.
        by_symbol, err = await _maybe_await(
            tool.execute({"operation": "hover", "path": str(f), "symbol": "add"})
        )
        assert err is False and "add" in by_symbol

        # Definition from the call site (line 4 hint) jumps back to the def.
        defn, err = await _maybe_await(
            tool.execute(
                {"operation": "definition", "path": str(f), "symbol": "add", "line": 4}
            )
        )
        assert err is False and "m.py:1:" in defn

        # References by symbol surfaces the call site on line 4.
        refs, err = await _maybe_await(
            tool.execute({"operation": "references", "path": str(f), "symbol": "add"})
        )
        assert err is False and "m.py:4:" in refs


@pytest.mark.skipif(RUST_ANALYZER is None, reason="rust-analyzer not on PATH")
class TestRustAnalyzerIntegration:
    """End-to-end: real rust-analyzer subprocess.

    rust-analyzer needs a Cargo.toml to index a crate properly — we set one up
    so the server doesn't bail with `no workspace`.
    """

    @pytest.fixture
    def crate(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text(
            '[package]\nname = "demo"\nversion = "0.1.0"\nedition = "2021"\n'
        )
        src = tmp_path / "src"
        src.mkdir()
        return tmp_path

    @pytest_asyncio.fixture
    async def client(self, crate):
        c = LspClient(
            LspServer(cmd=("rust-analyzer",), language_id="rust"),
            crate,
        )
        yield c
        await c.shutdown()

    @pytest.mark.asyncio
    async def test_hover_on_function_returns_type(self, client, crate):
        f = crate / "src" / "main.rs"
        f.write_text(
            "fn add(a: i32, b: i32) -> i32 { a + b }\nfn main() { add(1, 2); }\n"
        )
        # rust-analyzer indexes the crate before answering hover — waiting for
        # its first diagnostics publish is a proxy for "analysis has landed".
        await client.open(f)
        await client.diagnostics(f, wait=8.0)
        # Hover on `add` at the call site (line 2, ~col 13 zero-based).
        result = await client.hover(f, 1, 12)
        if result is None:
            pytest.skip("rust-analyzer did not return hover (indexing too slow)")
        text = _fmt_hover(result)
        assert "i32" in text or "fn add" in text

    @pytest.mark.asyncio
    async def test_diagnostics_flag_type_error(self, client, crate):
        f = crate / "src" / "main.rs"
        f.write_text('fn main() { let x: i32 = "not an int"; }\n')
        diags = await client.diagnostics(f, wait=10.0)
        rendered = _fmt_diagnostics(diags)
        assert (
            "i32" in rendered
            or "type" in rendered.lower()
            or "mismatch" in rendered.lower()
        )


@pytest.mark.skipif(
    PYRIGHT is None
    or not (
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_BASE_URL")
    ),
    reason="needs pyright on PATH and an Anthropic endpoint (ANTHROPIC_API_KEY or ANTHROPIC_BASE_URL)",
)
class TestLspEndToEnd:
    """Fully live, nothing faked: a real Claude model, asked about a broken
    file, chooses to call the `lsp_python` tool, which `agent_loop` dispatches
    against a real pyright subprocess; the diagnostic flows back to the model.

    This exercises the whole path that was dead before — the model is told the
    tool exists, calls it, and the loop runs it. Skips offline (no Anthropic
    endpoint) or without pyright.
    """

    LIVE_MODEL = DEFAULT_MODEL

    @pytest.mark.asyncio
    async def test_real_model_uses_lsp_diagnostics(self, tmp_path):
        from anthropic import AsyncAnthropic

        f = tmp_path / "broken.py"
        f.write_text("print(no_such_name)\n")

        r = HookRunner()
        for hook in (
            system_prompt_hook,
            read_tool_hook,
            write_tool_hook,
            edit_tool_hook,
            bash_tool_hook,
            python_lsp_hook,
        ):
            r.load(hook)
        assert "lsp_python" in [t.name for t in r.tools]

        sm = SessionManager(tmp_path / "s.jsonl")
        sm.append(
            "user",
            f"Run the lsp_python tool with operation 'diagnostics' on {f}, then "
            "tell me the diagnostic message verbatim. Use only the lsp_python tool.",
        )
        client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY") or "proxy")
        try:
            await agent_loop(
                client, self.LIVE_MODEL, str(tmp_path), r.tools, sm, r, [], max_turns=6
            )
        finally:
            await r.fire("session_end")  # shut the real pyright server down

        dump = json.dumps([e.content for e in sm.entries])
        # The model actually invoked lsp_python (tool_use block by that name) ...
        assert "lsp_python" in dump, dump
        # ... and the real pyright diagnostic flowed back through a tool_result.
        assert "no_such_name" in dump or "undefined" in dump.lower(), dump

    @pytest.mark.asyncio
    async def test_real_model_locates_symbol_by_name(self, tmp_path):
        """The new addressing mode, fully live: the model is asked to locate a
        function and resolves it by `symbol` name — never computing a line or
        column — and the real definition flows back through pyright."""
        from anthropic import AsyncAnthropic

        f = tmp_path / "m.py"
        f.write_text(
            "def add(a: int, b: int) -> int:\n    return a + b\n\nresult = add(1, 2)\n"
        )

        r = HookRunner()
        for hook in (
            system_prompt_hook,
            read_tool_hook,
            write_tool_hook,
            edit_tool_hook,
            bash_tool_hook,
            python_lsp_hook,
        ):
            r.load(hook)
        assert "lsp_python" in [t.name for t in r.tools]

        sm = SessionManager(tmp_path / "s.jsonl")
        sm.append(
            "user",
            f"Using only the lsp_python tool, find where the function `add` is "
            f"defined in {f}. Address it by symbol name — pass symbol='add', do "
            "not compute a line or column. Then tell me which line it is on.",
        )
        client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY") or "proxy")
        try:
            await agent_loop(
                client, self.LIVE_MODEL, str(tmp_path), r.tools, sm, r, [], max_turns=6
            )
        finally:
            await r.fire("session_end")  # shut the real pyright server down

        dump = json.dumps([e.content for e in sm.entries])
        # The model invoked lsp_python addressing the target by symbol name ...
        assert "lsp_python" in dump, dump
        assert "symbol" in dump, dump
        # ... and the real definition (line 1 of m.py) flowed back.
        assert "m.py:1" in dump, dump


class TestLoadExtensions:
    """agent.load_extensions discovers sibling `.py` files exposing a `HOOKS`
    tuple and loads each hook, skipping the agent file and
    `test_*`/`conftest`/`_*` files. A broken extension is isolated, not fatal."""

    def test_loads_hooks_from_sibling(self, tmp_path):
        (tmp_path / "demo.py").write_text(
            "def _h(api):\n    api.register_event('demo_marker')\nHOOKS = (_h,)\n"
        )
        r = HookRunner()
        load_extensions(r, tmp_path)
        assert "demo_marker" in r.events

    def test_skips_test_underscore_and_conftest(self, tmp_path):
        for name in ("test_x.py", "_priv.py", "conftest.py"):
            (tmp_path / name).write_text(
                "def _h(api):\n    api.register_event('nope')\nHOOKS = (_h,)\n"
            )
        r = HookRunner()
        load_extensions(r, tmp_path)
        assert "nope" not in r.events

    def test_module_without_HOOKS_is_ignored(self, tmp_path):
        (tmp_path / "plain.py").write_text("VALUE = 1\n")
        r = HookRunner()
        load_extensions(r, tmp_path)  # imports it, finds no HOOKS, moves on
        assert r.tools == []

    def test_broken_extension_is_isolated(self, tmp_path, capsys):
        (tmp_path / "boom.py").write_text("raise RuntimeError('kaboom')\n")
        (tmp_path / "ok.py").write_text(
            "def _h(api):\n    api.register_event('ok_marker')\nHOOKS = (_h,)\n"
        )
        r = HookRunner()
        load_extensions(r, tmp_path)
        assert "ok_marker" in r.events  # ok.py still loaded despite boom.py
        assert "failed to import" in capsys.readouterr().err


# ═══════════════════════════════════════════════════════════════════════════
# Hypothesis fuzz
# ═══════════════════════════════════════════════════════════════════════════


class TestFuzzFire:
    @given(items=st.lists(st.text(max_size=10), max_size=8))
    def test_accumulate_preserves_order(self, items):
        r = HookRunner()
        for s in items:
            r.api.on(
                "build_system_prompt", lambda p, c, v=s: p.additional_context.append(v)
            )
        p = SystemPrompt(cwd="/")
        _fire(r, "build_system_prompt", p)
        assert p.additional_context == items

    @given(
        priorities=st.lists(
            st.integers(min_value=-100, max_value=100), min_size=1, max_size=8
        )
    )
    def test_priority_respects_order(self, priorities):
        r = HookRunner()
        for i, prio in enumerate(priorities):
            r.api.on(
                "build_system_prompt",
                lambda p, c, idx=i: p.additional_context.append(idx),
                priority=prio,
            )
        p = SystemPrompt(cwd="/")
        _fire(r, "build_system_prompt", p)
        # Expected: indices sorted by priority (stable)
        expected = [i for i, _ in sorted(enumerate(priorities), key=lambda x: x[1])]
        assert p.additional_context == expected

    @given(first_blocks=st.booleans())
    def test_block_short_circuits_iff_true(self, first_blocks):
        r = HookRunner()
        second_ran = []

        def maybe_block(p, c):
            p.blocked = first_blocks

        r.api.on("user_prompt_submit", maybe_block)
        r.api.on("user_prompt_submit", lambda p, c: second_ran.append(True))
        p = UserPrompt(prompt="x")
        _fire(r, "user_prompt_submit", p)
        if first_blocks:
            assert second_ran == []
        else:
            assert second_ran == [True]

    @given(n=st.integers(min_value=0, max_value=20))
    def test_n_handlers_all_run_without_error(self, n):
        r = HookRunner()
        seen = []
        for i in range(n):
            r.api.on("turn_start", lambda p, c, i=i: seen.append(i))
        _fire(r, "turn_start")
        assert seen == list(range(n))


class TestFuzzTools:
    @given(content=st.text(max_size=1000).filter(lambda s: "\r" not in s))
    def test_write_then_read_round_trips(self, content):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "rt.txt"
            _, wr = _write_tool().execute({"path": str(f), "content": content})
            assert wr is False
            out, rd = _read_tool().execute({"path": str(f), "limit": 10000})
            assert rd is False
            if not content:
                assert out == "(empty)"
                return
            recovered = [line.split("\t", 1)[1] for line in out.splitlines()]
            assert recovered == content.splitlines()

    @given(old=st.text(min_size=1, max_size=50), new=st.text(max_size=50))
    def test_edit_tool_never_raises(self, old, new):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "e.txt"
            f.write_text(old)
            content, err = _edit_tool().execute(
                {"path": str(f), "old": old, "new": new}
            )
            assert isinstance(content, str) and isinstance(err, bool)

    @given(cmd=st.text(alphabet="abcdefghijklmnop0123456789 -_", max_size=30))
    @settings(
        max_examples=30,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_bash_tool_never_raises(self, cmd):
        content, err = _exec(_bash_tool(), {"cmd": cmd or "true"})
        assert isinstance(content, str) and isinstance(err, bool)


class TestFuzzSession:
    @given(
        entries=st.lists(
            st.tuples(
                st.sampled_from(["user", "assistant", "tool_result"]),
                st.one_of(
                    st.text(max_size=40),
                    st.lists(
                        st.dictionaries(st.text(max_size=5), st.integers(), max_size=3),
                        max_size=3,
                    ),
                ),
            ),
            max_size=10,
        )
    )
    def test_append_then_reload(self, entries):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.jsonl"
            sm1 = SessionManager(p)
            for role, content in entries:
                sm1.append(role, content)
            sm2 = SessionManager(p)
            assert len(sm2.entries) == len(entries)
            for (role, _), reloaded in zip(entries, sm2.entries):
                assert reloaded.role == role


# ═══════════════════════════════════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
