# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic", "rich", "pytest", "pytest-asyncio", "hypothesis"]
# ///
"""
Tests for agent.py — unit + property-based (fuzz) coverage of the core.

Run:
    uv run test_agent.py

The fuzz suite uses Hypothesis to throw adversarial inputs at the hook
runner, tools, and session manager. It does *not* hit the Anthropic API —
the agent_loop's model interaction is excluded. Everything else is covered.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Make agent.py importable regardless of invocation cwd.
sys.path.insert(0, str(Path(__file__).parent))

from agent import (
    # hook system
    ADDITIONAL_CONTEXT,
    BLOCK,
    CONTENT,
    INPUT,
    IS_ERROR,
    PATH,
    REASON,
    SYSTEM_PROMPT,
    AgentSession,
    Event,
    HookAPI,
    HookRunner,
    Merge,
    Return,
    # tools
    Tool,
    _bash_tool,
    _edit_tool,
    _read_tool,
    _write_tool,
    # hooks
    anthropic_cache_hook,
    bash_tool_hook,
    cache_stats_hook,
    edit_tool_hook,
    lifecycle_hook,
    list_sessions_hook,
    markdown_renderer_hook,
    read_tool_hook,
    resume_hook,
    # session
    SessionEntry,
    SessionManager,
    session_end_printer_hook,
    session_start_printer_hook,
    stdout_renderer_hook,
    system_prompt_hook,
    write_tool_hook,
)

# ═══════════════════════════════════════════════════════════════════════════
# Merge / Return / Event
# ═══════════════════════════════════════════════════════════════════════════


class TestMerge:
    def test_enum_members(self):
        assert {k.name for k in Merge} == {"REPLACE", "ACCUMULATE", "BLOCK"}

    def test_identity_comparison(self):
        assert Merge.BLOCK is Merge.BLOCK
        assert Merge.BLOCK is not Merge.REPLACE


class TestReturn:
    def test_defaults_to_replace(self):
        assert Return("foo").kind is Merge.REPLACE

    def test_custom_kind(self):
        assert Return("foo", kind=Merge.BLOCK).kind is Merge.BLOCK

    def test_frozen(self):
        r = Return("foo")
        with pytest.raises(Exception):
            r.key = "bar"  # type: ignore[misc]

    def test_builtin_constants(self):
        assert BLOCK.key == "block" and BLOCK.kind is Merge.BLOCK
        assert ADDITIONAL_CONTEXT.key == "additional_context"
        assert ADDITIONAL_CONTEXT.kind is Merge.ACCUMULATE
        assert REASON.kind is Merge.REPLACE


class TestEvent:
    def test_default_returns_empty(self):
        assert Event("foo").returns == ()

    def test_stores_returns(self):
        e = Event("x", (BLOCK, REASON))
        assert e.returns == (BLOCK, REASON)


# ═══════════════════════════════════════════════════════════════════════════
# HookRunner construction / HookAPI
# ═══════════════════════════════════════════════════════════════════════════


class TestRunnerInit:
    def test_standard_events_preloaded(self):
        runner = HookRunner()
        expected = {
            "before_session_load",
            "args_parsed",
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
        assert set(runner.events) == expected

    def test_events_in_lifecycle_order(self):
        runner = HookRunner()
        names = list(runner.events.keys())
        assert names[0] == "before_session_load"
        assert names[1] == "args_parsed"
        assert names[-1] == "session_end"

    def test_handlers_empty_per_event(self):
        runner = HookRunner()
        for name in runner.events:
            assert runner.handlers[name] == []

    def test_no_tools_initially(self):
        assert HookRunner().tools == []


class TestHookAPI:
    def test_on_registers_handler(self):
        runner = HookRunner()
        h = lambda e, c: None
        runner.api.on("text_delta", h)
        assert runner.handlers["text_delta"] == [h]

    def test_on_unknown_event_raises(self):
        runner = HookRunner()
        with pytest.raises(ValueError, match="unknown event"):
            runner.api.on("nonexistent", lambda e, c: None)

    def test_register_event_adds_and_orders(self):
        runner = HookRunner()
        runner.api.register_event("my_event", BLOCK, REASON)
        assert "my_event" in runner.events
        assert runner.events["my_event"].returns == (BLOCK, REASON)
        assert list(runner.events.keys())[-1] == "my_event"  # appended

    def test_register_event_duplicate_raises(self):
        runner = HookRunner()
        with pytest.raises(ValueError, match="already registered"):
            runner.api.register_event("session_start", ADDITIONAL_CONTEXT)

    def test_register_tool_appends(self):
        runner = HookRunner()
        t = _read_tool()
        runner.api.register_tool(t)
        assert runner.tools == [t]


# ═══════════════════════════════════════════════════════════════════════════
# HookRunner.fire — the core merge logic
# ═══════════════════════════════════════════════════════════════════════════


class TestFire:
    def test_no_handlers_returns_empty(self):
        assert HookRunner().fire("session_start", {}) == {}

    def test_observe_event_returns_drop(self):
        runner = HookRunner()
        runner.api.on("text_delta", lambda e, c: {"text": "ignored"})
        assert runner.fire("text_delta", {"text": "hi"}) == {}

    def test_replace_last_wins(self):
        runner = HookRunner()
        runner.api.on("build_system_prompt", lambda e, c: {"system_prompt": "A"})
        runner.api.on("build_system_prompt", lambda e, c: {"system_prompt": "B"})
        result = runner.fire("build_system_prompt", {"cwd": "/"})
        assert result["system_prompt"] == "B"

    def test_accumulate_collects_in_order(self):
        runner = HookRunner()
        for letter in "abc":
            runner.api.on(
                "session_start", lambda e, c, x=letter: {"additional_context": x}
            )
        result = runner.fire("session_start", {"cwd": "/"})
        assert result["additional_context"] == ["a", "b", "c"]

    def test_block_short_circuits_and_preserves_reason(self):
        runner = HookRunner()
        calls = []
        runner.api.on(
            "user_prompt_submit",
            lambda e, c: (calls.append(1), {"block": True, "reason": "no way"})[1],
        )
        runner.api.on("user_prompt_submit", lambda e, c: (calls.append(2), None)[1])
        result = runner.fire("user_prompt_submit", {"prompt": "x"})
        assert result == {"block": True, "reason": "no way"}
        assert calls == [1]  # second handler never ran

    def test_block_false_does_not_short_circuit(self):
        runner = HookRunner()
        runner.api.on("user_prompt_submit", lambda e, c: {"block": False})
        runner.api.on("user_prompt_submit", lambda e, c: {"additional_context": "note"})
        result = runner.fire("user_prompt_submit", {"prompt": "x"})
        assert result.get("block") is not True
        assert result["additional_context"] == ["note"]

    def test_handler_exception_isolated(self, capsys):
        runner = HookRunner()
        runner.api.on("text_delta", lambda e, c: 1 / 0)
        runner.api.on("text_delta", lambda e, c: {"text": "ok"})  # still runs
        result = runner.fire("text_delta", {"text": "x"})
        assert result == {}  # text_delta has no declared returns
        assert "error" in capsys.readouterr().err.lower()

    def test_none_return_ignored(self):
        runner = HookRunner()
        runner.api.on("build_system_prompt", lambda e, c: None)
        runner.api.on("build_system_prompt", lambda e, c: {"system_prompt": "x"})
        result = runner.fire("build_system_prompt", {"cwd": "/"})
        assert result["system_prompt"] == "x"

    def test_non_dict_return_ignored(self):
        runner = HookRunner()
        for bad in ["string", 42, [1, 2], True, 3.14]:
            runner.api.on("build_system_prompt", lambda e, c, b=bad: b)
        runner.api.on("build_system_prompt", lambda e, c: {"system_prompt": "ok"})
        result = runner.fire("build_system_prompt", {"cwd": "/"})
        assert result["system_prompt"] == "ok"

    def test_undeclared_keys_dropped(self):
        runner = HookRunner()
        runner.api.on(
            "pre_tool_use",
            lambda e, c: {"block": True, "reason": "x", "magic": "dropped"},
        )
        result = runner.fire("pre_tool_use", {"name": "t", "input": {}})
        assert "magic" not in result
        assert result["block"] is True


class TestLoad:
    def test_load_invokes_hook(self):
        runner = HookRunner()
        called = []
        runner.load(lambda api: called.append(api))
        assert len(called) == 1 and isinstance(called[0], HookAPI)

    def test_load_failure_doesnt_crash(self, capsys):
        runner = HookRunner()

        def broken(api):
            raise RuntimeError("oops")

        runner.load(broken)  # must not raise
        assert "broken" in capsys.readouterr().err


class TestDescribe:
    def test_lists_all_events(self):
        out = HookRunner().describe()
        for name in ("before_session_load", "pre_tool_use", "session_end"):
            assert name in out

    def test_shows_handler_count(self):
        runner = HookRunner()
        runner.api.on("text_delta", lambda e, c: None)
        runner.api.on("text_delta", lambda e, c: None)
        assert "2 handler(s)" in runner.describe()


# ═══════════════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════════════


class TestReadTool:
    def test_reads_existing(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello")
        content, err = _read_tool().execute({"path": str(f)})
        assert content == "hello" and err is False

    def test_missing_file_returns_error(self, tmp_path):
        content, err = _read_tool().execute({"path": str(tmp_path / "nope")})
        assert err is True
        assert "FileNotFoundError" in content


class TestWriteTool:
    def test_creates(self, tmp_path):
        f = tmp_path / "new.txt"
        content, err = _write_tool().execute({"path": str(f), "content": "hi"})
        assert err is False and f.read_text() == "hi"

    def test_overwrites(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("old")
        _write_tool().execute({"path": str(f), "content": "new"})
        assert f.read_text() == "new"

    def test_reports_byte_count(self, tmp_path):
        content, _ = _write_tool().execute(
            {"path": str(tmp_path / "a.txt"), "content": "abc"}
        )
        assert "3 bytes" in content


class TestEditTool:
    def test_unique_replacement(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello world")
        _, err = _edit_tool().execute({"path": str(f), "old": "world", "new": "there"})
        assert err is False and f.read_text() == "hello there"

    def test_multiple_matches_errors(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x x x")
        content, err = _edit_tool().execute({"path": str(f), "old": "x", "new": "y"})
        assert err is True and "exactly once" in content
        assert f.read_text() == "x x x"  # file unchanged

    def test_no_match_errors(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hello")
        _, err = _edit_tool().execute({"path": str(f), "old": "zz", "new": "y"})
        assert err is True

    def test_missing_file_errors(self, tmp_path):
        _, err = _edit_tool().execute(
            {"path": str(tmp_path / "nope"), "old": "a", "new": "b"}
        )
        assert err is True


class TestBashTool:
    def test_success(self):
        content, err = _bash_tool().execute({"cmd": "echo hello"})
        assert "hello" in content and err is False

    def test_failure(self):
        _, err = _bash_tool().execute({"cmd": "exit 1"})
        assert err is True

    def test_empty_output(self):
        content, _ = _bash_tool().execute({"cmd": "true"})
        assert content == "(no output)"

    def test_stderr_captured(self):
        content, _ = _bash_tool().execute({"cmd": "echo oops 1>&2"})
        assert "oops" in content

    def test_truncation_20k(self):
        content, _ = _bash_tool().execute({"cmd": "python3 -c 'print(\"x\" * 25000)'"})
        assert len(content) <= 20_000


# ═══════════════════════════════════════════════════════════════════════════
# SessionManager
# ═══════════════════════════════════════════════════════════════════════════


class TestSessionManager:
    def test_new_session_writes_header(self, tmp_path):
        path = tmp_path / "s.jsonl"
        SessionManager(path)
        first_line = path.read_text().splitlines()[0]
        header = json.loads(first_line)
        assert header["type"] == "header"
        assert header["version"] == 1
        assert "id" in header

    def test_append_creates_entry(self, tmp_path):
        sm = SessionManager(tmp_path / "s.jsonl")
        e = sm.append("user", "hi")
        assert isinstance(e, SessionEntry)
        assert e.role == "user" and e.content == "hi" and e.parent_id is None

    def test_append_chains_parent_ids(self, tmp_path):
        sm = SessionManager(tmp_path / "s.jsonl")
        e1 = sm.append("user", "a")
        e2 = sm.append("assistant", "b")
        e3 = sm.append(
            "tool_result", [{"type": "tool_result", "tool_use_id": "x", "content": "y"}]
        )
        assert e2.parent_id == e1.id
        assert e3.parent_id == e2.id

    def test_load_existing_round_trip(self, tmp_path):
        path = tmp_path / "s.jsonl"
        sm1 = SessionManager(path)
        sm1.append("user", "hi")
        sm1.append("assistant", [{"type": "text", "text": "hello"}])
        sm2 = SessionManager(path)
        assert len(sm2.entries) == 2
        assert sm2.entries[0].role == "user"
        assert sm2.entries[1].role == "assistant"

    def test_to_messages_shape(self, tmp_path):
        sm = SessionManager(tmp_path / "s.jsonl")
        sm.append("user", "hi")
        sm.append("assistant", [{"type": "text", "text": "hello"}])
        sm.append(
            "tool_result",
            [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}],
        )
        msgs = sm.to_messages()
        assert msgs[0] == {"role": "user", "content": "hi"}
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"  # tool_result maps to user

    def test_jsonl_format_is_valid(self, tmp_path):
        path = tmp_path / "s.jsonl"
        sm = SessionManager(path)
        sm.append("user", "hi")
        sm.append("assistant", "hello")
        for line in path.read_text().splitlines():
            json.loads(line)  # must parse


# ═══════════════════════════════════════════════════════════════════════════
# Built-in hooks
# ═══════════════════════════════════════════════════════════════════════════


class TestSystemPromptHook:
    def test_builds_default_prompt(self):
        runner = HookRunner()
        runner.load(system_prompt_hook)
        result = runner.fire("build_system_prompt", {"cwd": "/tmp/work"})
        assert "coding assistant" in result["system_prompt"]
        assert "/tmp/work" in result["system_prompt"]

    def test_system_prompt_excludes_reminders(self):
        """Reminders are no longer baked into the system prompt — they flow into
        the conversation as user messages so the system prompt stays cacheable."""
        runner = HookRunner()
        runner.load(system_prompt_hook)
        result = runner.fire("build_system_prompt", {"cwd": "/"})
        assert "reminder" not in result["system_prompt"].lower()
        assert "<system-reminder>" not in result["system_prompt"]


class TestToolHooks:
    @pytest.mark.parametrize(
        "hook,tool_name",
        [
            (read_tool_hook, "read"),
            (write_tool_hook, "write"),
            (edit_tool_hook, "edit"),
            (bash_tool_hook, "bash"),
        ],
    )
    def test_registers_tool(self, hook, tool_name):
        runner = HookRunner()
        runner.load(hook)
        names = [t.name for t in runner.tools]
        assert tool_name in names


class TestResumeHook:
    def _fake_args(self, *, new=False, session=None):
        from argparse import Namespace

        return Namespace(new=new, session=session)

    def test_default_picks_most_recent(self, tmp_path):
        """Default (no flag) resumes the most recent session."""
        f1 = tmp_path / "s1.jsonl"
        f1.write_text('{"type":"header"}\n')
        f2 = tmp_path / "s2.jsonl"
        f2.write_text('{"type":"header"}\n')
        # f2 is the newer file (written second, newer mtime)

        runner = HookRunner()
        runner.load(resume_hook)
        result = runner.fire(
            "before_session_load",
            {
                "args": self._fake_args(),
                "default_path": tmp_path / "new.jsonl",
            },
        )
        assert result["path"] == f2

    def test_new_flag_forces_fresh_session(self, tmp_path):
        """--new bypasses resume even when prior sessions exist."""
        (tmp_path / "old.jsonl").write_text('{"type":"header"}\n')
        runner = HookRunner()
        runner.load(resume_hook)
        result = runner.fire(
            "before_session_load",
            {
                "args": self._fake_args(new=True),
                "default_path": tmp_path / "new.jsonl",
            },
        )
        assert "path" not in result

    def test_session_flag_picks_explicit(self, tmp_path):
        f = tmp_path / "specific.jsonl"
        f.write_text('{"type":"header"}\n')
        runner = HookRunner()
        runner.load(resume_hook)
        result = runner.fire(
            "before_session_load",
            {
                "args": self._fake_args(session=f),
                "default_path": tmp_path / "new.jsonl",
            },
        )
        assert result["path"] == f

    def test_default_with_no_prior_sessions_starts_fresh(self, tmp_path):
        """Empty session dir → no override, main() creates a new session."""
        runner = HookRunner()
        runner.load(resume_hook)
        result = runner.fire(
            "before_session_load",
            {"args": self._fake_args(), "default_path": tmp_path / "new.jsonl"},
        )
        assert "path" not in result

    def test_default_with_missing_session_dir_starts_fresh(self, tmp_path):
        """First-ever run: session dir doesn't exist yet → start fresh."""
        runner = HookRunner()
        runner.load(resume_hook)
        result = runner.fire(
            "before_session_load",
            {
                "args": self._fake_args(),
                "default_path": tmp_path / "does-not-exist" / "new.jsonl",
            },
        )
        assert "path" not in result


class TestListSessionsHook:
    def _args(self, **kw):
        from argparse import Namespace

        return Namespace(**{"list_sessions": False, **kw})

    def test_disabled_by_default_is_noop(self, capsys):
        runner = HookRunner()
        runner.load(list_sessions_hook)
        runner.fire("args_parsed", {"args": self._args()})
        assert capsys.readouterr().out == ""

    def test_no_sessions_prints_placeholder(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        runner = HookRunner()
        runner.load(list_sessions_hook)
        with pytest.raises(SystemExit) as exc:
            runner.fire("args_parsed", {"args": self._args(list_sessions=True)})
        assert exc.value.code == 0
        assert "no sessions yet" in capsys.readouterr().out

    def test_lists_sessions_newest_first(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        session_dir = tmp_path / ".py-agent" / "sessions"
        # Create two sessions; the second gets a newer mtime via explicit touch.
        p1 = session_dir / "20260505T100000_aaaaaaaa.jsonl"
        sm1 = SessionManager(p1)
        sm1.append("user", "explain rust lifetimes")
        p2 = session_dir / "20260506T120000_bbbbbbbb.jsonl"
        sm2 = SessionManager(p2)
        sm2.append("user", "write a hello world in rust")

        runner = HookRunner()
        runner.load(list_sessions_hook)
        with pytest.raises(SystemExit):
            runner.fire("args_parsed", {"args": self._args(list_sessions=True)})
        out = capsys.readouterr().out
        # Newest (p2) appears first.
        assert out.index("bbbbbbbb") < out.index("aaaaaaaa")
        assert "hello world in rust" in out
        assert "rust lifetimes" in out
        assert "1 entries" in out

    def test_preview_skips_list_content_messages(self, tmp_path, monkeypatch, capsys):
        """First user-role entry with string content is the preview; a prior
        tool_result (list content mapped to user) should be skipped."""
        monkeypatch.setenv("HOME", str(tmp_path))
        path = tmp_path / ".py-agent" / "sessions" / "20260507T130000_ccccccc1.jsonl"
        sm = SessionManager(path)
        sm.append(
            "tool_result",
            [{"type": "tool_result", "tool_use_id": "t1", "content": "ignored"}],
        )
        sm.append("user", "the real first prompt")

        runner = HookRunner()
        runner.load(list_sessions_hook)
        with pytest.raises(SystemExit):
            runner.fire("args_parsed", {"args": self._args(list_sessions=True)})
        out = capsys.readouterr().out
        assert "the real first prompt" in out
        assert "ignored" not in out


class TestPrinterHooks:
    def test_session_start_printer_silent_for_new(self, tmp_path, capsys):
        sm = SessionManager(tmp_path / "s.jsonl")
        runner = HookRunner()
        runner.load(session_start_printer_hook)
        runner.fire("session_start", {"cwd": "/"}, {"session": sm})
        assert "resumed" not in capsys.readouterr().err

    def test_session_start_printer_loud_for_resumed(self, tmp_path, capsys):
        path = tmp_path / "s.jsonl"
        sm1 = SessionManager(path)
        sm1.append("user", "hi")
        sm2 = SessionManager(path)  # reload — entries present
        runner = HookRunner()
        runner.load(session_start_printer_hook)
        runner.fire("session_start", {"cwd": "/"}, {"session": sm2})
        err = capsys.readouterr().err
        assert "resumed" in err and "1 entries" in err

    def test_session_end_printer(self, tmp_path, capsys):
        sm = SessionManager(tmp_path / "s.jsonl")
        runner = HookRunner()
        runner.load(session_end_printer_hook)
        runner.fire("session_end", {}, {"session": sm})
        assert "session saved to" in capsys.readouterr().err


class TestStdoutRendererHook:
    def _args(self, **kw):
        from argparse import Namespace

        return Namespace(**{"markdown": False, **kw})

    def test_prints_text_delta(self, capsys):
        runner = HookRunner()
        runner.load(stdout_renderer_hook)
        runner.fire("args_parsed", {"args": self._args()})
        runner.fire("text_delta", {"text": "hello"})
        assert "hello" in capsys.readouterr().out

    def test_prints_tool_call(self, capsys):
        runner = HookRunner()
        runner.load(stdout_renderer_hook)
        runner.fire("args_parsed", {"args": self._args()})
        runner.fire("pre_tool_use", {"name": "read", "input": {"path": "/x"}})
        out = capsys.readouterr().out
        assert "read" in out and "/x" in out

    def test_suppresses_text_when_markdown_enabled(self, capsys):
        """With --markdown set, stdout_renderer defers text to markdown_renderer."""
        runner = HookRunner()
        runner.load(stdout_renderer_hook)
        runner.fire("args_parsed", {"args": self._args(markdown=True)})
        runner.fire("text_delta", {"text": "hello"})
        runner.fire("text_end", {})
        assert capsys.readouterr().out == ""


class TestMarkdownRendererHook:
    def _args(self, **kw):
        from argparse import Namespace

        return Namespace(**{"markdown": False, **kw})

    def test_disabled_by_default(self, capsys):
        runner = HookRunner()
        runner.load(markdown_renderer_hook)
        runner.fire("args_parsed", {"args": self._args()})
        runner.fire("text_delta", {"text": "# hi\n\n**bold**"})
        runner.fire("text_end", {})
        # No markdown flag → no output from this hook.
        assert capsys.readouterr().out == ""

    def test_renders_on_text_end_when_enabled(self, capsys):
        runner = HookRunner()
        runner.load(markdown_renderer_hook)
        runner.fire("args_parsed", {"args": self._args(markdown=True)})
        runner.fire("text_delta", {"text": "# Heading\n\n"})
        runner.fire("text_delta", {"text": "**bold** text"})
        runner.fire("text_end", {})
        out = capsys.readouterr().out
        # rich strips the raw markdown syntax and prints styled plain text;
        # the visible words survive.
        assert "Heading" in out
        assert "bold" in out
        # the literal "**" markers should be gone after rendering
        assert "**" not in out

    def test_buffer_flushes_between_turns(self, capsys):
        runner = HookRunner()
        runner.load(markdown_renderer_hook)
        runner.fire("args_parsed", {"args": self._args(markdown=True)})

        runner.fire("text_delta", {"text": "first"})
        runner.fire("text_end", {})
        first_out = capsys.readouterr().out

        runner.fire("text_delta", {"text": "second"})
        runner.fire("text_end", {})
        second_out = capsys.readouterr().out

        assert "first" in first_out and "second" not in first_out
        assert "second" in second_out and "first" not in second_out

    def test_registers_markdown_flag(self):
        runner = HookRunner()
        runner.load(markdown_renderer_hook)
        # argparse registered the flag without error
        ns = runner.parser.parse_args(["--markdown"])
        assert ns.markdown is True


# ═══════════════════════════════════════════════════════════════════════════
# AgentSession orchestration (no model calls)
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentSessionLifecycle:
    def test_start_fires_session_start(self, tmp_path):
        sm = SessionManager(tmp_path / "s.jsonl")
        runner = HookRunner()
        fired = []
        runner.api.on("session_start", lambda e, c: fired.append(e))
        agent = AgentSession(client=None, model="m", session=sm, runner=runner)  # type: ignore[arg-type]
        agent.start()
        assert len(fired) == 1

    def test_end_fires_session_end(self, tmp_path):
        sm = SessionManager(tmp_path / "s.jsonl")
        runner = HookRunner()
        fired = []
        runner.api.on("session_end", lambda e, c: fired.append(e))
        agent = AgentSession(client=None, model="m", session=sm, runner=runner)  # type: ignore[arg-type]
        agent.end()
        assert len(fired) == 1

    def test_session_start_reminders_queued(self, tmp_path):
        sm = SessionManager(tmp_path / "s.jsonl")
        runner = HookRunner()
        runner.api.on(
            "session_start", lambda e, c: {"additional_context": "be careful"}
        )
        agent = AgentSession(client=None, model="m", session=sm, runner=runner)  # type: ignore[arg-type]
        agent.start()
        assert agent.pending_reminders == ["be careful"]


# ═══════════════════════════════════════════════════════════════════════════
# Fuzz tests — Hypothesis
# ═══════════════════════════════════════════════════════════════════════════

KNOWN_EVENTS = st.sampled_from(
    [
        "before_session_load",
        "args_parsed",
        "session_start",
        "user_prompt_submit",
        "turn_start",
        "build_system_prompt",
        "before_model_request",
        "text_delta",
        "text_end",
        "message_end",
        "pre_tool_use",
        "post_tool_use",
        "stop",
        "session_end",
    ]
)

random_key = st.text(min_size=1, max_size=10)
random_value = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.text(max_size=20),
    st.lists(st.text(max_size=10), max_size=3),
)
random_handler_return = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.text(max_size=20),
    st.lists(st.integers(), max_size=3),
    st.dictionaries(random_key, random_value, max_size=5),
)


class TestFuzzFire:
    @given(event=KNOWN_EVENTS, returns=st.lists(random_handler_return, max_size=5))
    @settings(
        max_examples=300, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_fire_never_raises(self, event, returns):
        runner = HookRunner()
        for r in returns:
            runner.api.on(event, lambda e, c, _r=r: _r)
        # Must not raise regardless of what handlers return.
        result = runner.fire(event, {"whatever": "payload"}, {})
        assert isinstance(result, dict)

    @given(
        n_handlers=st.integers(min_value=0, max_value=8),
        contexts=st.lists(st.text(min_size=0, max_size=5), max_size=8),
    )
    def test_accumulate_preserves_order(self, n_handlers, contexts):
        runner = HookRunner()
        contexts = contexts[:n_handlers]
        for c in contexts:
            runner.api.on(
                "session_start", lambda e, ctx, v=c: {"additional_context": v}
            )
        result = runner.fire("session_start", {"cwd": "/"})
        assert result.get("additional_context", []) == contexts

    @given(first_blocks=st.booleans(), reason=st.text(max_size=20))
    def test_block_short_circuits_iff_truthy(self, first_blocks, reason):
        runner = HookRunner()
        second_ran = []
        runner.api.on(
            "user_prompt_submit", lambda e, c: {"block": first_blocks, "reason": reason}
        )
        runner.api.on(
            "user_prompt_submit", lambda e, c: (second_ran.append(True), None)[1]
        )
        result = runner.fire("user_prompt_submit", {"prompt": "x"})
        if first_blocks:
            assert result["block"] is True
            assert second_ran == []
        else:
            assert second_ran == [True]


class TestFuzzTools:
    @given(content=st.text(max_size=1000).filter(lambda s: "\r" not in s))
    def test_write_then_read_round_trips(self, content):
        # \r is excluded because Python's text-mode read translates \r and
        # \r\n to \n (universal newlines). That's a read_text quirk, not an
        # agent bug — write_text + read_text both use text mode by default.
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "rt.txt"
            _, wr_err = _write_tool().execute({"path": str(f), "content": content})
            assert wr_err is False
            out, rd_err = _read_tool().execute({"path": str(f)})
            assert rd_err is False
            assert out == content

    @given(
        original=st.text(min_size=1, max_size=100), replacement=st.text(max_size=100)
    )
    def test_edit_tool_never_raises(self, original, replacement):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "e.txt"
            f.write_text(original)
            content, err = _edit_tool().execute(
                {"path": str(f), "old": original, "new": replacement}
            )
            assert isinstance(content, str)
            assert isinstance(err, bool)

    SAFE_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 -_"

    @given(cmd=st.text(alphabet=SAFE_CHARS, max_size=30))
    @settings(deadline=5000, max_examples=50)  # subprocess spawn is slow
    def test_bash_tool_never_raises(self, cmd):
        content, err = _bash_tool().execute({"cmd": cmd or "true"})
        assert isinstance(content, str)
        assert isinstance(err, bool)


class TestFuzzSession:
    @given(
        entries=st.lists(
            st.tuples(
                st.sampled_from(["user", "assistant", "tool_result"]),
                st.one_of(
                    st.text(max_size=50),
                    st.lists(
                        st.dictionaries(st.text(max_size=5), st.integers(), max_size=3),
                        max_size=3,
                    ),
                ),
            ),
            max_size=10,
        ),
    )
    def test_append_then_reload(self, entries):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fuzz.jsonl"
            sm1 = SessionManager(path)
            for role, content in entries:
                sm1.append(role, content)
            sm2 = SessionManager(path)
            assert len(sm2.entries) == len(entries)
            for (role, _), reloaded in zip(entries, sm2.entries):
                assert reloaded.role == role

    @given(
        entries=st.lists(
            st.tuples(st.sampled_from(["user", "assistant"]), st.text(max_size=30)),
            max_size=10,
        ),
    )
    def test_parent_id_chain_is_linear(self, entries):
        with tempfile.TemporaryDirectory() as d:
            sm = SessionManager(Path(d) / "chain.jsonl")
            ids = []
            for role, content in entries:
                ids.append(sm.append(role, content).id)
            for i, entry in enumerate(sm.entries):
                expected_parent = ids[i - 1] if i > 0 else None
                assert entry.parent_id == expected_parent


class TestFuzzHookAPI:
    @given(event=st.text(min_size=1, max_size=30))
    def test_on_unknown_event_always_raises(self, event):
        runner = HookRunner()
        if event in runner.events:
            return  # known event — different test covers this
        with pytest.raises(ValueError):
            runner.api.on(event, lambda e, c: None)

    @given(
        name=st.text(min_size=1, max_size=30),
        n_returns=st.integers(min_value=0, max_value=3),
    )
    def test_register_event_idempotent_via_different_names(self, name, n_returns):
        runner = HookRunner()
        if name in runner.events:
            return  # skip names that collide with the built-in lifecycle
        returns_tuple = (BLOCK, REASON, ADDITIONAL_CONTEXT)[:n_returns]
        runner.api.register_event(name, *returns_tuple)
        assert name in runner.events
        # Second registration must always raise.
        with pytest.raises(ValueError):
            runner.api.register_event(name, *returns_tuple)


class TestFuzzIntegration:
    """Fuzz tests that stress multiple parts of the agent together."""

    @given(
        cwd=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789/_-", min_size=1, max_size=50
        ),
        extra_context=st.lists(st.text(max_size=50), max_size=3),
    )
    def test_system_prompt_hook_robust(self, cwd, extra_context):
        """system_prompt_hook + extra additional_context handlers → always a usable string."""
        runner = HookRunner()
        runner.load(system_prompt_hook)
        for ctx_text in extra_context:
            runner.api.on(
                "build_system_prompt",
                lambda e, c, v=ctx_text: {"additional_context": v},
            )
        result = runner.fire("build_system_prompt", {"cwd": cwd})
        prompt = result["system_prompt"]
        assert isinstance(prompt, str) and prompt  # non-empty
        assert cwd in prompt
        # additional_context from handlers still accumulates in the merged result.
        assert result.get("additional_context", []) == extra_context

    @given(
        event_name=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz_",
            min_size=1,
            max_size=20,
        ),
        handler_returns=st.lists(
            st.dictionaries(
                st.sampled_from(
                    ["block", "reason", "input", "additional_context", "ignored_key"]
                ),
                st.one_of(st.booleans(), st.text(max_size=10), st.integers()),
                max_size=3,
            ),
            max_size=5,
        ),
    )
    def test_custom_event_round_trip(self, event_name, handler_returns):
        """Register a user-defined event, bind handlers that return random dicts, fire it.
        Ensure merged result only contains declared keys and the runner never crashes.
        """
        runner = HookRunner()
        if event_name in runner.events:
            return  # skip collisions with built-in lifecycle names
        runner.api.register_event(event_name, BLOCK, REASON, INPUT, ADDITIONAL_CONTEXT)
        for r in handler_returns:
            runner.api.on(event_name, lambda e, c, v=r: v)
        result = runner.fire(event_name, {})
        # The "ignored_key" should never leak into the result.
        assert "ignored_key" not in result
        # Any key present must be one of the declared return names.
        for key in result:
            assert key in {"block", "reason", "input", "additional_context"}

    @given(
        first_block=st.booleans(),
        extra_keys=st.dictionaries(
            st.sampled_from(["input", "additional_context"]),
            st.text(max_size=10),
            max_size=2,
        ),
        reason=st.text(max_size=30),
    )
    def test_pre_tool_use_block_beats_other_keys(self, first_block, extra_keys, reason):
        """When a pre_tool_use handler blocks, later handlers don't run and the
        blocking handler's other declared keys accompany the block in the result."""
        runner = HookRunner()
        later_ran = []

        def first(event, ctx):
            return {"block": first_block, "reason": reason, **extra_keys}

        def second(event, ctx):
            later_ran.append(True)
            return {"input": {"changed": "by-second"}}

        runner.api.on("pre_tool_use", first)
        runner.api.on("pre_tool_use", second)
        result = runner.fire("pre_tool_use", {"name": "bash", "input": {"cmd": "ls"}})

        if first_block:
            assert result["block"] is True
            assert result["reason"] == reason
            assert later_ran == []
            # second handler's "input" rewrite must not appear in a blocked result
            assert result.get("input") != {"changed": "by-second"}
        else:
            assert later_ran == [True]
            # Without blocking, the later handler's input rewrite wins (last-write).
            assert result.get("input") == {"changed": "by-second"}


# ═══════════════════════════════════════════════════════════════════════════
# anthropic_cache_hook — can resume fully reconstruct the cacheable payload?
# ═══════════════════════════════════════════════════════════════════════════


def _payload_with_cache(
    runner: HookRunner, system: str, tools: list, messages: list
) -> dict:
    """Fire before_model_request through the given runner and return the final payload."""
    override = runner.fire(
        "before_model_request",
        {"system": system, "tools": tools, "messages": messages},
    )
    return {
        "system": override.get("system", system),
        "tools": override.get("tools", tools),
        "messages": override.get("messages", messages),
    }


class TestAnthropicCacheHook:
    def test_marks_system_as_list_with_cache_control(self):
        runner = HookRunner()
        runner.load(anthropic_cache_hook)
        out = _payload_with_cache(
            runner, "you are X", [], [{"role": "user", "content": "hi"}]
        )
        assert out["system"] == [
            {
                "type": "text",
                "text": "you are X",
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def test_tools_pass_through_unchanged(self):
        """Conventional 2-checkpoint pattern: system + last message only.
        Tools render before system; the system breakpoint already caches
        tools + system as a single prefix, so a separate tools marker is
        redundant and is intentionally not applied."""
        runner = HookRunner()
        runner.load(anthropic_cache_hook)
        tools = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        out = _payload_with_cache(
            runner, "s", tools, [{"role": "user", "content": "hi"}]
        )
        assert out["tools"] == tools
        for tool in out["tools"]:
            assert "cache_control" not in tool

    def test_marks_last_message_string_content(self):
        runner = HookRunner()
        runner.load(anthropic_cache_hook)
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        out = _payload_with_cache(runner, "s", [], msgs)
        # all but the last pass through unchanged
        assert out["messages"][0] == {"role": "user", "content": "a"}
        assert out["messages"][1] == {"role": "assistant", "content": "b"}
        # the last is rewritten: string → [ {text, cache_control} ]
        assert out["messages"][2] == {
            "role": "user",
            "content": [
                {"type": "text", "text": "c", "cache_control": {"type": "ephemeral"}}
            ],
        }

    def test_marks_last_message_list_content_last_block_only(self):
        runner = HookRunner()
        runner.load(anthropic_cache_hook)
        # Simulates a tool_result message — content is already a list of blocks.
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok1"},
                    {"type": "tool_result", "tool_use_id": "t2", "content": "ok2"},
                ],
            }
        ]
        out = _payload_with_cache(runner, "s", [], msgs)
        blocks = out["messages"][-1]["content"]
        assert "cache_control" not in blocks[0]
        assert blocks[1]["cache_control"] == {"type": "ephemeral"}

    def test_empty_inputs_noop(self):
        runner = HookRunner()
        runner.load(anthropic_cache_hook)
        out = _payload_with_cache(runner, "", [], [])
        assert out == {"system": "", "tools": [], "messages": []}


class TestResumeCacheParity:
    """
    Does resuming a session reconstruct a byte-identical cacheable prefix?

    We build the `messages` array two ways:
      (a) live — append entries in memory, then to_messages()
      (b) saved — append, throw away the in-memory SessionManager, re-open from disk, to_messages()

    Then we push both through the same runner + anthropic_cache_hook and compare.
    If the prefix (everything before the last message) differs, resume cannot
    hit the same cache. If it matches, any turn-1-on-resume cache miss is NOT
    caused by the resume path.
    """

    def _messages_fresh(self, tmp_path, entries):
        sm = SessionManager(tmp_path / "live.jsonl")
        for role, content in entries:
            sm.append(role, content)
        return sm.to_messages()

    def _messages_resumed(self, tmp_path, entries):
        path = tmp_path / "saved.jsonl"
        sm1 = SessionManager(path)
        for role, content in entries:
            sm1.append(role, content)
        del sm1
        sm2 = SessionManager(path)  # reload from JSONL
        return sm2.to_messages()

    def test_to_messages_identical_after_resume(self, tmp_path):
        entries = [
            ("user", "hello"),
            ("assistant", [{"type": "text", "text": "hi back"}]),
            (
                "tool_result",
                [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
            ),
            ("user", "try again"),
        ]
        live = self._messages_fresh(tmp_path, entries)
        resumed = self._messages_resumed(tmp_path, entries)
        assert live == resumed, "resume altered the message shape before caching"

    def test_cached_payload_identical_after_resume(self, tmp_path):
        entries = [
            ("user", "hello"),
            ("assistant", [{"type": "text", "text": "hi back"}]),
            (
                "tool_result",
                [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
            ),
            ("user", "try again"),
        ]
        live_msgs = self._messages_fresh(tmp_path, entries)
        resumed_msgs = self._messages_resumed(tmp_path, entries)

        system = "you are a coding assistant\ncwd=/tmp"
        tools = [{"name": "read"}, {"name": "write"}]

        r1 = HookRunner()
        r1.load(anthropic_cache_hook)
        r2 = HookRunner()
        r2.load(anthropic_cache_hook)
        live_payload = _payload_with_cache(r1, system, tools, live_msgs)
        resumed_payload = _payload_with_cache(r2, system, tools, resumed_msgs)

        # byte-for-byte JSON equality after serialization — this is what Anthropic
        # hashes for cache lookup, so any drift here = forced cache miss on turn 1.
        assert json.dumps(live_payload, sort_keys=True) == json.dumps(
            resumed_payload, sort_keys=True
        )

    def test_prefix_stable_across_turns(self, tmp_path):
        """
        Turn N and turn N+1 share the same messages[:-1]. The cache_control marker
        on messages[-1] moves each turn, but the cached *prefix* (system + tools +
        all earlier messages) should be byte-identical — otherwise no turn ever
        gets a read hit.
        """
        base = [
            ("user", "hi"),
            ("assistant", [{"type": "text", "text": "hello"}]),
        ]
        turn_n = base + [("user", "first question")]
        turn_n_plus_1 = base + [
            ("user", "first question"),
            (
                "assistant",
                [{"type": "text", "text": "first answer"}],
            ),
            ("user", "second question"),
        ]

        system = "static prompt"
        tools = [{"name": "read"}, {"name": "bash"}]

        def run(entries):
            msgs = self._messages_fresh(tmp_path / f"t_{len(entries)}", entries)
            r = HookRunner()
            r.load(anthropic_cache_hook)
            return _payload_with_cache(r, system, tools, msgs)

        p1 = run(turn_n)
        p2 = run(turn_n_plus_1)

        # system + tools marked identically (the tool cache marker doesn't move)
        assert p1["system"] == p2["system"]
        assert p1["tools"] == p2["tools"]

        # everything before the most-recently-cache-marked message on turn N
        # must survive unchanged into turn N+1 (no cache_control residue because
        # our hook only marks messages[-1], so older messages never got a marker).
        assert p2["messages"][: len(p1["messages"]) - 1] == p1["messages"][:-1]

    def test_last_message_marker_does_not_leak_into_earlier_turns(self, tmp_path):
        """Once messages[-1] on turn N becomes messages[-3] on turn N+1 (after
        assistant reply + tool_result), it must NOT still carry cache_control.
        Otherwise the prefix diverges and reads always miss."""
        entries_t1 = [("user", "q1")]
        entries_t2 = [
            ("user", "q1"),
            ("assistant", [{"type": "text", "text": "a1"}]),
            ("user", "q2"),
        ]
        system = "s"
        tools: list = []

        r1 = HookRunner()
        r1.load(anthropic_cache_hook)
        r2 = HookRunner()
        r2.load(anthropic_cache_hook)

        p1 = _payload_with_cache(
            r1, system, tools, self._messages_fresh(tmp_path / "x1", entries_t1)
        )
        p2 = _payload_with_cache(
            r2, system, tools, self._messages_fresh(tmp_path / "x2", entries_t2)
        )

        # On turn 2, the old user "q1" content should reappear unmodified —
        # stored in the session as a plain string, not as the list-wrapped
        # cache-marked form it took on turn 1.
        assert p2["messages"][0] == {"role": "user", "content": "q1"}
        # And turn 1's exact last-message shape must NOT equal turn 2's
        # first-message shape (the marker was on turn 1's last, not turn 2's first).
        assert p1["messages"][-1] != p2["messages"][0]


class TestCacheBreakpointSizing:
    """
    Anthropic silently drops any cache_control marker whose cumulative prefix
    is below the 1024-token minimum (Sonnet). Use bytes as a rough proxy for
    tokens (~4 chars/token) to catch wasted markers at the system/tools
    positions where the prefix is almost always too small.

    These aren't hard asserts on the hook — they document the observation so
    a future change (e.g. dropping the system/tools markers) has a test that
    fails loudly.
    """

    BYTES_PER_TOKEN = 4  # very rough; 3.5–4.5 is typical for English+code
    MIN_TOKENS = 1024

    def _est_tokens(self, blob: str) -> int:
        return len(blob) // self.BYTES_PER_TOKEN

    def test_system_breakpoint_below_min_for_typical_prompt(self):
        """The default system prompt is far under 1024 tokens — that marker
        gets dropped. Documents why marking system is wasted work."""
        runner = HookRunner()
        runner.load(system_prompt_hook)
        result = runner.fire("build_system_prompt", {"cwd": "/tmp/x"})
        assert self._est_tokens(result["system_prompt"]) < self.MIN_TOKENS

    def test_tools_breakpoint_below_min_with_four_tools(self):
        """system + all four built-in tool schemas is also under 1024 tokens."""
        runner = HookRunner()
        for h in (
            system_prompt_hook,
            read_tool_hook,
            write_tool_hook,
            edit_tool_hook,
            bash_tool_hook,
        ):
            runner.load(h)
        sys_out = runner.fire("build_system_prompt", {"cwd": "/tmp/x"})
        schemas_blob = json.dumps([t.to_anthropic() for t in runner.tools])
        combined = sys_out["system_prompt"] + schemas_blob
        assert self._est_tokens(combined) < self.MIN_TOKENS, (
            "If this starts failing, the system+tools marker may actually "
            "begin caching and this test should be inverted."
        )

    def test_prints_zeros_when_usage_missing(self, capsys):
        runner = HookRunner()
        runner.load(cache_stats_hook)
        runner.fire("message_end", {"message": [], "usage": {}})
        err = capsys.readouterr().err
        assert "[cache] read=0 write=0 input=0" in err

    def test_coerces_none_to_zero(self, capsys):
        """Anthropic returns None (not missing) when there's no cache activity —
        cache_stats must render None as 0, not as the literal string 'None'."""
        runner = HookRunner()
        runner.load(cache_stats_hook)
        runner.fire(
            "message_end",
            {
                "message": [],
                "usage": {
                    "input_tokens": 100,
                    "cache_read_input_tokens": None,
                    "cache_creation_input_tokens": None,
                },
            },
        )
        err = capsys.readouterr().err
        assert "read=0 write=0 input=100" in err
        assert "None" not in err

    def test_reports_real_numbers(self, capsys):
        runner = HookRunner()
        runner.load(cache_stats_hook)
        runner.fire(
            "message_end",
            {
                "message": [],
                "usage": {
                    "input_tokens": 10,
                    "cache_read_input_tokens": 2048,
                    "cache_creation_input_tokens": 512,
                },
            },
        )
        err = capsys.readouterr().err
        assert "read=2048 write=512 input=10" in err


# ═══════════════════════════════════════════════════════════════════════════
# Entry point: `uv run test_agent.py`
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
