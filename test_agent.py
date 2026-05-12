# /// script
# requires-python = ">=3.10"
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
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Make agent.py importable regardless of invocation cwd.
sys.path.insert(0, str(Path(__file__).parent))

from agent import (
    # tools (private factories used in tests)
    _bash_tool,
    _edit_tool,
    _read_tool,
    _session_dir,
    _stream_extra_hook,
    _write_tool,
    # agent session + runtime
    AgentSession,
    # payloads
    ArgsParsed,
    Event,
    HookAPI,
    HookRunner,
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
        out = HookRunner().fire("user_prompt_submit", p)
        assert out is p

    def test_none_payload_ok_for_signal_event(self):
        r = HookRunner()
        hits = []
        r.api.on("turn_start", lambda p, c: hits.append(p))
        assert r.fire("turn_start") is None
        assert hits == [None]

    def test_handler_mutates_payload(self):
        r = HookRunner()
        r.api.on("build_system_prompt", lambda p, c: setattr(p, "system_prompt", "X"))
        p = SystemPrompt(cwd="/")
        assert r.fire("build_system_prompt", p).system_prompt == "X"  # type: ignore[union-attr]

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
        r.fire("build_system_prompt", p)
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
        r.fire("build_system_prompt", p)
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
        r.fire("build_system_prompt", p)
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
        r.fire("user_prompt_submit", p)
        assert p.blocked and p.reason == "stop"
        assert calls == [1]

    def test_block_requires_truthy(self):
        r = HookRunner()
        ran = []
        r.api.on("user_prompt_submit", lambda p, c: None)  # no-op
        r.api.on("user_prompt_submit", lambda p, c: ran.append(1))
        p = UserPrompt(prompt="hi")
        r.fire("user_prompt_submit", p)
        assert ran == [1]
        assert p.blocked is False

    def test_handler_exception_isolated_by_default(self, capsys):
        r = HookRunner()
        r.api.on("text_end", lambda p, c: 1 / 0)
        ran = []
        r.api.on("text_end", lambda p, c: ran.append(1))
        r.fire("text_end")
        assert ran == [1]
        assert "error" in capsys.readouterr().err.lower()

    def test_fire_unknown_event_raises(self):
        with pytest.raises(ValueError, match="unknown event"):
            HookRunner().fire("nope")

    def test_returns_same_payload_instance(self):
        r = HookRunner()
        r.api.on("user_prompt_submit", lambda p, c: p.additional_context.append("x"))
        p = UserPrompt(prompt="hi")
        out = r.fire("user_prompt_submit", p)
        assert out is p


class TestStrictMode:
    def test_default_is_non_strict(self):
        assert HookRunner().strict is False

    def test_non_strict_swallows(self, capsys):
        r = HookRunner()
        ran = []
        r.api.on("text_end", lambda p, c: 1 / 0)
        r.api.on("text_end", lambda p, c: ran.append(1))
        r.fire("text_end")
        assert ran == [1]
        assert "error" in capsys.readouterr().err.lower()

    def test_strict_reraises(self):
        r = HookRunner()
        r.strict = True
        r.api.on("text_end", lambda p, c: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            r.fire("text_end")

    def test_strict_preserves_exception_type_and_message(self):
        class Boom(Exception):
            pass

        r = HookRunner()
        r.strict = True

        def raiser(p, c):
            raise Boom("specific")

        r.api.on("text_end", raiser)
        with pytest.raises(Boom, match="specific"):
            r.fire("text_end")

    def test_strict_stops_at_first_raise(self):
        r = HookRunner()
        r.strict = True
        calls = []
        r.api.on("text_end", lambda p, c: calls.append(1))
        r.api.on("text_end", lambda p, c: 1 / 0)
        r.api.on("text_end", lambda p, c: calls.append(3))
        with pytest.raises(ZeroDivisionError):
            r.fire("text_end")
        assert calls == [1]

    def test_strict_toggleable_at_runtime(self):
        r = HookRunner()
        r.api.on("text_end", lambda p, c: 1 / 0)
        r.fire("text_end")  # swallowed
        r.strict = True
        with pytest.raises(ZeroDivisionError):
            r.fire("text_end")

    def test_strict_does_not_affect_block(self):
        """block short-circuit is a normal control flow, not an exception."""
        r = HookRunner()
        r.strict = True
        r.api.on(
            "user_prompt_submit",
            lambda p, c: (setattr(p, "blocked", True), setattr(p, "reason", "no")),
        )
        p = UserPrompt(prompt="x")
        r.fire("user_prompt_submit", p)
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
        content, err = _bash_tool().execute({"cmd": "echo hi"})
        assert err is False and "hi" in content

    def test_nonzero_exit_is_error(self):
        _, err = _bash_tool().execute({"cmd": "false"})
        assert err is True

    def test_no_output_placeholder(self):
        content, _ = _bash_tool().execute({"cmd": "true"})
        assert content == "(no output)"


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
        r.fire("build_system_prompt", p)
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
        r.fire("build_system_prompt", p)
        assert "Current date: 2026-05-11" in p.system_prompt
        assert "2026-05-12" not in p.system_prompt

    def test_stable_across_fires(self):
        r = HookRunner()
        r.load(system_prompt_hook)
        a = SystemPrompt(cwd="/tmp")
        b = SystemPrompt(cwd="/tmp")
        r.fire("build_system_prompt", a)
        r.fire("build_system_prompt", b)
        assert a.system_prompt == b.system_prompt


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
        r.fire("build_system_prompt", p)
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
        r.fire("build_system_prompt", p)
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
        r.fire("build_system_prompt", p)
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
        r.fire("build_system_prompt", p)
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
        r.fire("build_system_prompt", p)
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
        r.fire("before_session_load", p)
        assert p.path == f2

    def test_new_flag_skips_resume(self, tmp_path, monkeypatch):
        d = self._setup(tmp_path, monkeypatch)
        (d / "old.jsonl").write_text('{"type":"header"}\n')
        r = HookRunner()
        r.load(resume_hook)
        p = SessionPath(args=self._args(new=True))
        r.fire("before_session_load", p)
        assert p.path is None

    def test_session_flag_picks_explicit(self, tmp_path):
        f = tmp_path / "specific.jsonl"
        f.write_text('{"type":"header"}\n')
        r = HookRunner()
        r.load(resume_hook)
        p = SessionPath(args=self._args(session=f))
        r.fire("before_session_load", p)
        assert p.path == f


class TestSessionPathHook:
    def test_provides_default_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        r = HookRunner()
        r.load(session_path_hook)
        p = SessionPath(args=argparse.Namespace())
        r.fire("before_session_load", p)
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
        r.fire("before_session_load", p)
        assert p.path == explicit


class TestFlagHooks:
    def test_model_flag_defaults(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(model_flag_hook)
        args = r.parser.parse_args([])
        p = SessionConfig(args=args)
        r.fire("build_session_config", p)
        assert p.model == "claude-sonnet-4-6"

    def test_model_flag_overridden(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(model_flag_hook)
        args = r.parser.parse_args(["--model", "claude-opus-4"])
        p = SessionConfig(args=args)
        r.fire("build_session_config", p)
        assert p.model == "claude-opus-4"

    def test_max_turns_flag(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(max_turns_flag_hook)
        args = r.parser.parse_args(["--max-turns", "7"])
        p = SessionConfig(args=args)
        r.fire("build_session_config", p)
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
        r.fire("args_parsed", ArgsParsed(args=args))
        p = ModelRequest(system="", tools=[], messages=[])
        r.fire("before_model_request", p)
        assert p.extra["max_tokens"] == 8000

    def test_thinking_default_summarized(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(thinking_hook)
        args = r.parser.parse_args([])
        r.fire("args_parsed", ArgsParsed(args=args))
        p = ModelRequest(system="", tools=[], messages=[])
        r.fire("before_model_request", p)
        assert p.extra["thinking"] == {"type": "adaptive", "display": "summarized"}

    def test_thinking_override(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(thinking_hook)
        args = r.parser.parse_args(["--thinking-display", "hidden"])
        r.fire("args_parsed", ArgsParsed(args=args))
        p = ModelRequest(system="", tools=[], messages=[])
        r.fire("before_model_request", p)
        assert p.extra["thinking"]["display"] == "hidden"

    def test_effort_default_xhigh(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(output_effort_hook)
        args = r.parser.parse_args([])
        r.fire("args_parsed", ArgsParsed(args=args))
        p = ModelRequest(system="", tools=[], messages=[])
        r.fire("before_model_request", p)
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
        r.fire("before_model_request", p)
        assert p.extra == {}


class TestStrictHooksFlag:
    def test_sets_strict_when_flag_given(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(strict_hooks_flag_hook)
        args = r.parser.parse_args(["--strict-hooks"])
        r.fire("args_parsed", ArgsParsed(args=args))
        assert r.strict is True

    def test_strict_off_by_default(self):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(strict_hooks_flag_hook)
        args = r.parser.parse_args([])
        r.fire("args_parsed", ArgsParsed(args=args))
        assert r.strict is False


class TestDebugHooksFlag:
    def test_prints_when_enabled(self, capsys):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(debug_hooks_flag_hook)
        args = r.parser.parse_args(["--debug-hooks"])
        r.fire("args_parsed", ArgsParsed(args=args), {"runner": r})
        assert "before_session_load" in capsys.readouterr().err

    def test_silent_when_disabled(self, capsys):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(debug_hooks_flag_hook)
        args = r.parser.parse_args([])
        r.fire("args_parsed", ArgsParsed(args=args), {"runner": r})
        assert capsys.readouterr().err == ""


class TestAnthropicClientHook:
    def test_provides_client(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        r = HookRunner()
        r.load(anthropic_client_hook)
        p = SessionConfig(args=argparse.Namespace())
        r.fire("build_session_config", p)
        assert p.client is not None


class TestAnthropicCacheHook:
    """anthropic_cache_hook tags the final system block and last message with
    ephemeral cache_control so Anthropic caches the prefix."""

    def _run_hook(self, system, messages):
        r = HookRunner()
        r.load(anthropic_cache_hook)
        p = ModelRequest(system=system, tools=[], messages=messages)
        r.fire("before_model_request", p)
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
        r.fire("args_parsed", ArgsParsed(args=args))
        p = ModelRequest(
            system="sys", tools=[], messages=[{"role": "user", "content": "hi"}]
        )
        r.fire("model_request_prepared", p)
        assert "cache-debug" not in capsys.readouterr().err

    def test_prints_when_enabled(self, capsys):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(cache_debug_hook)
        args = r.parser.parse_args(["--debug-cache"])
        r.fire("args_parsed", ArgsParsed(args=args))
        p = ModelRequest(
            system="sys", tools=[], messages=[{"role": "user", "content": "hi"}]
        )
        r.fire("model_request_prepared", p)
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
            r.fire("args_parsed", ArgsParsed(args=args), {"runner": r})
        assert exc.value.code == 0
        assert "no sessions" in capsys.readouterr().out

    def test_does_nothing_without_flag(self, capsys):
        r = HookRunner()
        r.load(prompt_arg_hook)
        r.load(list_sessions_hook)
        args = r.parser.parse_args([])
        r.fire("args_parsed", ArgsParsed(args=args), {"runner": r})
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
        r.fire("text_delta", TextDelta(text="hello "))
        r.fire("text_delta", TextDelta(text="world"))
        assert capsys.readouterr().out == ""  # buffered, no print yet
        r.fire("text_end")
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
        r.fire("message_end", MessageEnd(message=[], usage=usage))
        out = capsys.readouterr().out
        assert "read=100" in out and "write=50" in out and "input=25" in out

    def test_pre_post_tool_use_default_rendering(self, capsys):
        r = HookRunner()
        r.load(read_tool_hook)
        r.load(ui_hook)
        r.fire("pre_tool_use", PreTool(id="t1", name="read", input={"path": "/foo"}))
        r.fire(
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
        r.fire("pre_tool_use", pre)
        post = PostTool(
            id="x",
            name="read",
            input={"path": "/foo"},
            content="ok",
            is_error=False,
            state=pre.state,
        )
        r.fire("post_tool_use", post)

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
        a.start()
        assert len(hits) == 1 and isinstance(hits[0], SessionStart)

    def test_start_collects_additional_context(self, tmp_path):
        a = self._make(tmp_path)
        a.runner.api.on(
            "session_start", lambda p, c: p.additional_context.append("reminder!")
        )
        a.start()
        assert "reminder!" in a.pending_reminders

    def test_end_fires_session_end(self, tmp_path):
        a = self._make(tmp_path)
        hits = []
        a.runner.api.on("session_end", lambda p, c: hits.append(True))
        a.end()
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
        r.fire("args_parsed", ArgsParsed(args=args))

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
        r.fire("build_system_prompt", p)
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
        r.fire("build_system_prompt", p)
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
        r.fire("user_prompt_submit", p)
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
        r.fire("turn_start")
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
        content, err = _bash_tool().execute({"cmd": cmd or "true"})
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
