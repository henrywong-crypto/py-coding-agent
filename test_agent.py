# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic", "rich", "prompt_toolkit", "pytest", "pytest-asyncio", "hypothesis"]
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
    anthropic_client_hook,
    bash_tool_hook,
    edit_tool_hook,
    lifecycle_hook,
    list_sessions_hook,
    max_turns_flag_hook,
    model_flag_hook,
    read_tool_hook,
    resume_hook,
    session_path_hook,
    strict_hooks_flag_hook,
    # session
    SessionEntry,
    SessionManager,
    system_prompt_hook,
    ui_hook,
    write_tool_hook,
)

# ═══════════════════════════════════════════════════════════════════════════
# Merge / Return / Event
# ═══════════════════════════════════════════════════════════════════════════


class TestMerge:
    def test_enum_members(self):
        assert {k.name for k in Merge} == {"REPLACE", "ACCUMULATE", "BLOCK", "CHAIN"}

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

    def test_register_tool_rejects_duplicate_name(self):
        runner = HookRunner()
        runner.api.register_tool(_read_tool())
        with pytest.raises(ValueError, match="already registered"):
            runner.api.register_tool(_read_tool())

    def test_register_tool_error_message_includes_name(self):
        runner = HookRunner()
        runner.api.register_tool(_bash_tool())
        with pytest.raises(ValueError, match="'bash'"):
            runner.api.register_tool(_bash_tool())

    def test_register_tool_duplicate_does_not_partially_register(self):
        """Collision must raise *before* mutating the list — no half-registered entry."""
        runner = HookRunner()
        runner.api.register_tool(_read_tool())
        with pytest.raises(ValueError):
            runner.api.register_tool(_read_tool())
        assert len(runner.tools) == 1
        assert runner.tools[0].name == "read"

    def test_register_tool_distinct_names_coexist(self):
        runner = HookRunner()
        for t in (_read_tool(), _write_tool(), _edit_tool(), _bash_tool()):
            runner.api.register_tool(t)
        assert [t.name for t in runner.tools] == ["read", "write", "edit", "bash"]

    def test_register_prompter_sets_single(self):
        runner = HookRunner()

        async def p(args):
            return "x"

        runner.api.register_prompter(p)
        assert runner.prompter is p

    def test_register_history_loader_sets_single(self):
        runner = HookRunner()
        loader = lambda: ["a", "b"]
        runner.api.register_history_loader(loader)
        assert runner.history_loader is loader


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


class TestStrictMode:
    """HookRunner.strict — opt-in flag that turns swallowed handler exceptions
    into real errors. Useful for debugging hooks during development."""

    def test_default_is_non_strict(self):
        assert HookRunner().strict is False

    def test_non_strict_swallows_and_continues(self, capsys):
        runner = HookRunner()
        second_ran = []
        runner.api.on("text_delta", lambda e, c: 1 / 0)
        runner.api.on("text_delta", lambda e, c: second_ran.append(True))
        runner.fire("text_delta", {"text": "x"})  # must not raise
        assert second_ran == [True]  # later handler still ran
        assert "error" in capsys.readouterr().err.lower()

    def test_strict_reraises(self):
        runner = HookRunner()
        runner.strict = True
        runner.api.on("text_delta", lambda e, c: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            runner.fire("text_delta", {"text": "x"})

    def test_strict_preserves_exception_type_and_message(self):
        class Boom(Exception):
            pass

        runner = HookRunner()
        runner.strict = True

        def raiser(e, c):
            raise Boom("specific message")

        runner.api.on("text_delta", raiser)
        with pytest.raises(Boom, match="specific message"):
            runner.fire("text_delta", {"text": "x"})

    def test_strict_preserves_traceback(self):
        """`raise` without args re-raises the current exception with its traceback
        intact — so users can see which handler failed and where."""
        import traceback

        runner = HookRunner()
        runner.strict = True

        def raiser(e, c):
            raise RuntimeError("inside-handler")

        runner.api.on("text_delta", raiser)
        try:
            runner.fire("text_delta", {"text": "x"})
        except RuntimeError:
            tb = traceback.format_exc()
            assert "raiser" in tb  # handler frame appears in traceback
            assert "inside-handler" in tb

    def test_strict_stops_at_first_raise(self):
        runner = HookRunner()
        runner.strict = True
        calls = []
        runner.api.on("text_delta", lambda e, c: calls.append(1))
        runner.api.on("text_delta", lambda e, c: 1 / 0)
        runner.api.on("text_delta", lambda e, c: calls.append(3))  # never runs
        with pytest.raises(ZeroDivisionError):
            runner.fire("text_delta", {"text": "x"})
        assert calls == [1]

    def test_strict_toggleable_at_runtime(self):
        runner = HookRunner()
        runner.api.on("text_delta", lambda e, c: 1 / 0)
        runner.fire("text_delta", {"text": "x"})  # off by default → swallowed
        runner.strict = True
        with pytest.raises(ZeroDivisionError):
            runner.fire("text_delta", {"text": "x"})
        runner.strict = False
        runner.fire("text_delta", {"text": "x"})  # flipping back restores swallow

    def test_strict_does_not_affect_successful_handlers(self):
        runner = HookRunner()
        runner.strict = True
        runner.api.on("build_system_prompt", lambda e, c: {"system_prompt": "ok"})
        result = runner.fire("build_system_prompt", {"cwd": "/"})
        assert result == {"system_prompt": "ok"}

    def test_strict_does_not_affect_declared_block_return(self):
        """`block: True` is a valid return, not an exception — strict mode must
        not turn a legitimate BLOCK short-circuit into an error."""
        runner = HookRunner()
        runner.strict = True
        runner.api.on(
            "user_prompt_submit", lambda e, c: {"block": True, "reason": "nope"}
        )
        result = runner.fire("user_prompt_submit", {"prompt": "x"})
        assert result == {"block": True, "reason": "nope"}


class TestChainMerge:
    """CHAIN: each handler sees prior handlers' running result via the event
    payload. Enables composition — two mutators stack instead of stomping."""

    def test_single_handler_behaves_like_replace(self):
        runner = HookRunner()
        runner.api.register_event("e", Return("x", kind=Merge.CHAIN))
        runner.api.on("e", lambda e, c: {"x": 1})
        assert runner.fire("e", {}) == {"x": 1}

    def test_second_handler_reads_first_result_from_payload(self):
        """The defining behavior: the second handler's payload contains what
        the first handler returned, so it can transform instead of overwrite."""
        runner = HookRunner()
        runner.api.register_event("e", Return("x", kind=Merge.CHAIN))
        runner.api.on("e", lambda e, c: {"x": 1})
        runner.api.on("e", lambda e, c: {"x": e["x"] + 10})
        assert runner.fire("e", {})["x"] == 11

    def test_first_handler_sees_initial_caller_payload(self):
        """Before any handler has returned, the chained key reflects the
        initial payload supplied by the caller."""
        runner = HookRunner()
        runner.api.register_event("e", Return("x", kind=Merge.CHAIN))
        runner.api.on("e", lambda e, c: {"x": e.get("x", 0) + 1})
        runner.api.on("e", lambda e, c: {"x": e["x"] * 2})
        assert runner.fire("e", {"x": 5})["x"] == 12  # 5 → 6 → 12

    def test_handler_that_omits_key_leaves_running_value_intact(self):
        runner = HookRunner()
        runner.api.register_event("e", Return("x", kind=Merge.CHAIN))
        runner.api.on("e", lambda e, c: {"x": 7})
        runner.api.on("e", lambda e, c: {"other": "ignored"})
        runner.api.on("e", lambda e, c: {"x": e["x"] * 10})
        assert runner.fire("e", {})["x"] == 70

    def test_caller_payload_is_not_mutated(self):
        """fire() copies the payload before iterating. A CHAIN-propagating
        fire must not mutate the caller's dict — otherwise callers that
        pass the same dict to multiple fires get surprising cross-talk."""
        runner = HookRunner()
        runner.api.register_event("e", Return("x", kind=Merge.CHAIN))
        runner.api.on("e", lambda e, c: {"x": 99})
        runner.api.on("e", lambda e, c: {"x": e["x"] + 1})
        payload = {"x": 1}
        runner.fire("e", payload)
        assert payload == {"x": 1}

    def test_chain_short_circuits_with_block(self):
        """BLOCK wins over CHAIN: later handlers never run, the result
        keeps the last chained value from before the block."""
        runner = HookRunner()
        runner.api.register_event("e", Return("x", kind=Merge.CHAIN), BLOCK, REASON)
        runner.api.on("e", lambda e, c: {"x": 1})
        runner.api.on("e", lambda e, c: {"block": True, "reason": "stop"})
        runner.api.on("e", lambda e, c: {"x": 999})  # unreached
        result = runner.fire("e", {})
        assert result["block"] is True
        assert result["x"] == 1

    def test_builtin_chain_keys_on_registered_events(self):
        """The five built-in CHAIN keys are: input (pre_tool_use), content
        (post_tool_use), system/tools/messages (before_model_request)."""
        runner = HookRunner()

        def chain_keys(event: str) -> set[str]:
            return {
                r.key for r in runner.events[event].returns if r.kind is Merge.CHAIN
            }

        assert chain_keys("pre_tool_use") == {"input"}
        assert chain_keys("post_tool_use") == {"content"}
        assert chain_keys("before_model_request") == {"system", "tools", "messages"}

    def test_two_before_model_request_mutators_compose(self):
        """Real-world payoff: two hooks both transforming `system` no longer
        stomp on each other. The second sees the first's markers intact."""
        runner = HookRunner()

        def add_marker(name):
            return lambda e, c: {"system": e.get("system", "") + f" [{name}]"}

        runner.api.on("before_model_request", add_marker("x"))
        runner.api.on("before_model_request", add_marker("y"))

        result = runner.fire(
            "before_model_request",
            {"system": "base", "tools": [], "messages": []},
        )
        assert result["system"] == "base [x] [y]"


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

    def test_date_captured_at_load_not_per_fire(self, monkeypatch):
        """date.today() is closed over at hook load — firing after a date change
        must not update the prompt, so the cached prefix survives midnight."""
        import agent
        from types import SimpleNamespace

        def fake(iso):
            return SimpleNamespace(today=lambda: SimpleNamespace(isoformat=lambda: iso))

        monkeypatch.setattr(agent, "date", fake("2026-05-11"))
        runner = HookRunner()
        runner.load(system_prompt_hook)

        # Jump the clock forward a day. The already-loaded hook should ignore it.
        monkeypatch.setattr(agent, "date", fake("2026-05-12"))
        prompt = runner.fire("build_system_prompt", {"cwd": "/"})["system_prompt"]

        assert "Current date: 2026-05-11" in prompt
        assert "2026-05-12" not in prompt

    def test_date_stable_across_multiple_fires(self):
        """Same prompt string on every fire — a fresh date.today() per call
        would break prefix cache hits within a long-lived process."""
        runner = HookRunner()
        runner.load(system_prompt_hook)
        a = runner.fire("build_system_prompt", {"cwd": "/tmp"})["system_prompt"]
        b = runner.fire("build_system_prompt", {"cwd": "/tmp"})["system_prompt"]
        c = runner.fire("build_system_prompt", {"cwd": "/tmp"})["system_prompt"]
        assert a == b == c

    def test_two_hook_loads_get_independent_dates(self, monkeypatch):
        """Each call to system_prompt_hook(api) captures its own date —
        loading twice with different clocks produces different prompts."""
        import agent
        from types import SimpleNamespace

        def fake(iso):
            return SimpleNamespace(today=lambda: SimpleNamespace(isoformat=lambda: iso))

        monkeypatch.setattr(agent, "date", fake("2026-01-01"))
        r1 = HookRunner()
        r1.load(system_prompt_hook)

        monkeypatch.setattr(agent, "date", fake("2027-01-01"))
        r2 = HookRunner()
        r2.load(system_prompt_hook)

        p1 = r1.fire("build_system_prompt", {"cwd": "/"})["system_prompt"]
        p2 = r2.fire("build_system_prompt", {"cwd": "/"})["system_prompt"]
        assert "2026-01-01" in p1 and "2027-01-01" not in p1
        assert "2027-01-01" in p2 and "2026-01-01" not in p2


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

    def _session_dir(self, tmp_path, monkeypatch, *, create=True):
        """Arrange HOME + cwd so resume_hook's `_session_dir(os.getcwd())`
        lands in a clean tmp location. Returns the session dir path."""
        from agent import _session_dir

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        d = _session_dir(str(tmp_path))
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def test_default_picks_most_recent(self, tmp_path, monkeypatch):
        """Default (no flag) resumes the most recent session."""
        d = self._session_dir(tmp_path, monkeypatch)
        f1 = d / "s1.jsonl"
        f1.write_text('{"type":"header"}\n')
        f2 = d / "s2.jsonl"
        f2.write_text('{"type":"header"}\n')
        # f2 is the newer file (written second, newer mtime)

        runner = HookRunner()
        runner.load(resume_hook)
        result = runner.fire("before_session_load", {"args": self._fake_args()})
        assert result["path"] == f2

    def test_new_flag_forces_fresh_session(self, tmp_path, monkeypatch):
        """--new bypasses resume even when prior sessions exist."""
        d = self._session_dir(tmp_path, monkeypatch)
        (d / "old.jsonl").write_text('{"type":"header"}\n')
        runner = HookRunner()
        runner.load(resume_hook)
        result = runner.fire("before_session_load", {"args": self._fake_args(new=True)})
        assert "path" not in result

    def test_session_flag_picks_explicit(self, tmp_path):
        f = tmp_path / "specific.jsonl"
        f.write_text('{"type":"header"}\n')
        runner = HookRunner()
        runner.load(resume_hook)
        result = runner.fire(
            "before_session_load", {"args": self._fake_args(session=f)}
        )
        assert result["path"] == f

    def test_default_with_no_prior_sessions_starts_fresh(self, tmp_path, monkeypatch):
        """Empty session dir → no override, main() falls back to session_path_hook."""
        self._session_dir(tmp_path, monkeypatch)  # dir exists but empty
        runner = HookRunner()
        runner.load(resume_hook)
        result = runner.fire("before_session_load", {"args": self._fake_args()})
        assert "path" not in result

    def test_default_with_missing_session_dir_starts_fresh(self, tmp_path, monkeypatch):
        """First-ever run: session dir doesn't exist yet → start fresh."""
        self._session_dir(tmp_path, monkeypatch, create=False)
        runner = HookRunner()
        runner.load(resume_hook)
        result = runner.fire("before_session_load", {"args": self._fake_args()})
        assert "path" not in result


class TestSessionPathHook:
    """Provides the default `{session_dir}/{timestamp}_{uuid}.jsonl` path on
    `before_session_load`. resume_hook registers a later handler that
    overrides this default when a prior session exists."""

    def _arrange(self, tmp_path, monkeypatch):
        from agent import _session_dir

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        return _session_dir(str(tmp_path))

    def test_provides_default_path(self, tmp_path, monkeypatch):
        session_dir = self._arrange(tmp_path, monkeypatch)
        runner = HookRunner()
        runner.load(session_path_hook)
        result = runner.fire("before_session_load", {"args": None})
        assert "path" in result
        assert result["path"].parent == session_dir
        assert result["path"].suffix == ".jsonl"

    def test_path_is_unique_per_fire(self, tmp_path, monkeypatch):
        """Each fire generates a fresh timestamp + uuid tail. Rapid succession
        must still yield distinct paths so concurrent sessions don't collide."""
        self._arrange(tmp_path, monkeypatch)
        runner = HookRunner()
        runner.load(session_path_hook)
        a = runner.fire("before_session_load", {"args": None})["path"]
        b = runner.fire("before_session_load", {"args": None})["path"]
        assert a != b

    def test_default_overridden_by_resume(self, tmp_path, monkeypatch):
        """session_path_hook's default must lose to resume_hook's explicit
        --session path — the REPLACE semantics of PATH + registration order
        let resume win."""
        from argparse import Namespace

        self._arrange(tmp_path, monkeypatch)
        chosen = tmp_path / "chosen.jsonl"
        chosen.write_text('{"type":"header"}\n')

        runner = HookRunner()
        runner.load(session_path_hook)
        runner.load(resume_hook)
        args = Namespace(new=False, session=chosen)
        result = runner.fire("before_session_load", {"args": args})
        assert result["path"] == chosen

    def test_default_survives_when_resume_declines(self, tmp_path, monkeypatch):
        """--new tells resume_hook to return None; the default must survive."""
        from argparse import Namespace

        session_dir = self._arrange(tmp_path, monkeypatch)
        runner = HookRunner()
        runner.load(session_path_hook)
        runner.load(resume_hook)
        args = Namespace(new=True, session=None)
        result = runner.fire("before_session_load", {"args": args})
        assert result["path"].parent == session_dir


class TestAnthropicClientHook:
    """Provides an `AsyncAnthropic` client via `build_session_config`."""

    def test_provides_client(self, monkeypatch):
        from anthropic import AsyncAnthropic

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        runner = HookRunner()
        runner.load(anthropic_client_hook)
        result = runner.fire("build_session_config", {})
        assert isinstance(result["client"], AsyncAnthropic)

    def test_each_fire_produces_fresh_client(self, monkeypatch):
        """Not memoized — each fire returns a new client. Fine in practice
        because main() fires once; the assertion pins the behavior so a
        future memoization is a deliberate choice, not an accident."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        runner = HookRunner()
        runner.load(anthropic_client_hook)
        a = runner.fire("build_session_config", {})["client"]
        b = runner.fire("build_session_config", {})["client"]
        assert a is not b


class TestListSessionsHook:
    def _args(self, **kw):
        from argparse import Namespace

        return Namespace(**{"list_sessions": False, **kw})

    def _session_dir(self, tmp_path):
        from agent import _session_dir

        return _session_dir(str(tmp_path))

    def test_disabled_by_default_is_noop(self, capsys):
        runner = HookRunner()
        runner.load(list_sessions_hook)
        runner.fire("args_parsed", {"args": self._args()})
        assert capsys.readouterr().out == ""

    def test_no_sessions_prints_placeholder(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        runner = HookRunner()
        runner.load(list_sessions_hook)
        with pytest.raises(SystemExit) as exc:
            runner.fire("args_parsed", {"args": self._args(list_sessions=True)})
        assert exc.value.code == 0
        assert "no sessions yet" in capsys.readouterr().out

    def test_lists_sessions_newest_first(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        session_dir = self._session_dir(tmp_path)
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
        monkeypatch.chdir(tmp_path)
        session_dir = self._session_dir(tmp_path)
        path = session_dir / "20260507T130000_ccccccc1.jsonl"
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

    def test_sessions_are_per_cwd_scoped(self, tmp_path, monkeypatch, capsys):
        """A session created in cwd A must not appear when listing from cwd B."""
        monkeypatch.setenv("HOME", str(tmp_path))
        # Session exists under cwd_a
        cwd_a = tmp_path / "project-a"
        cwd_a.mkdir()
        from agent import _session_dir

        sm = SessionManager(_session_dir(str(cwd_a)) / "20260505T100000_aaaaaaaa.jsonl")
        sm.append("user", "hello from project a")

        # List from cwd_b
        cwd_b = tmp_path / "project-b"
        cwd_b.mkdir()
        monkeypatch.chdir(cwd_b)

        runner = HookRunner()
        runner.load(list_sessions_hook)
        with pytest.raises(SystemExit):
            runner.fire("args_parsed", {"args": self._args(list_sessions=True)})
        out = capsys.readouterr().out
        assert "no sessions yet" in out
        assert "aaaaaaaa" not in out

    def test_session_dir_encodes_spaces(self):
        """Spaces in cwd should not appear in the directory name."""
        from agent import _session_dir

        out = _session_dir("/Users/h/My Documents/project").name
        assert " " not in out
        assert out.startswith("--Users-h-My-Documents-project-")

    def test_session_dir_avoids_collisions(self):
        """Paths that differ only in dashes/spaces/slashes must map to distinct dirs."""
        from agent import _session_dir

        paths = [
            "/Users/h/my-project",
            "/Users/h/my project",
            "/Users/h/my/project",
        ]
        dirs = {_session_dir(p).name for p in paths}
        assert len(dirs) == len(paths)


class TestSessionHistoryHook:
    def test_registers_history_loader(self, tmp_path, monkeypatch):
        from agent import session_history_hook

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        runner = HookRunner()
        runner.load(session_history_hook)
        assert runner.history_loader is not None
        assert runner.history_loader() == []  # empty when no sessions

    def test_loads_user_prompts_in_mtime_order(self, tmp_path, monkeypatch):
        from agent import _session_dir, session_history_hook

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        d = _session_dir(str(tmp_path))
        sm1 = SessionManager(d / "s1.jsonl")
        sm1.append("user", "first")
        sm1.append("assistant", [{"type": "text", "text": "reply"}])
        sm2 = SessionManager(d / "s2.jsonl")
        sm2.append("user", "second")

        runner = HookRunner()
        runner.load(session_history_hook)
        assert runner.history_loader() == ["first", "second"]

    def test_skips_non_user_and_non_string_entries(self, tmp_path, monkeypatch):
        from agent import _session_dir, session_history_hook

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        sm = SessionManager(_session_dir(str(tmp_path)) / "s.jsonl")
        sm.append("user", "kept")
        sm.append("assistant", [{"type": "text", "text": "skipped"}])
        sm.append(
            "tool_result",
            [{"type": "tool_result", "tool_use_id": "x", "content": "skipped"}],
        )

        runner = HookRunner()
        runner.load(session_history_hook)
        assert runner.history_loader() == ["kept"]


class TestPromptToolkitHook:
    def _args(self, prompt):
        from argparse import Namespace

        return Namespace(prompt=prompt)

    def test_registers_prompter(self):
        from agent import prompt_toolkit_hook

        runner = HookRunner()
        runner.load(prompt_toolkit_hook)
        assert runner.prompter is not None

    def test_cli_prompt_bypasses_prompt_toolkit(self):
        """When args.prompt is given, the prompter returns joined args without touching stdin."""
        import asyncio
        from agent import prompt_toolkit_hook

        runner = HookRunner()
        runner.load(prompt_toolkit_hook)
        assert (
            asyncio.run(runner.prompter(self._args(["hello", "world"])))
            == "hello world"
        )


class TestUIHook:
    """Single display extension — owns all terminal output."""

    def _register_tools(self, runner):
        from agent import read_tool_hook, edit_tool_hook

        runner.load(read_tool_hook)
        runner.load(edit_tool_hook)

    def test_renders_tool_call_with_default(self, capsys):
        runner = HookRunner()
        self._register_tools(runner)
        runner.load(ui_hook)
        runner.fire("pre_tool_use", {"name": "read", "input": {"path": "/x"}})
        runner.fire(
            "post_tool_use",
            {
                "name": "read",
                "input": {"path": "/x"},
                "content": "hello",
                "is_error": False,
            },
        )
        out = capsys.readouterr().out
        assert "Read" in out and "/x" in out
        assert "⏺" in out and "⎿" in out

    def test_renders_markdown_on_text_end(self, capsys):
        runner = HookRunner()
        runner.load(ui_hook)
        runner.fire("text_delta", {"text": "# H\n\n**bold** text"})
        runner.fire("text_end", {})
        out = capsys.readouterr().out
        assert "H" in out and "bold" in out
        assert "**" not in out  # markdown rendered, not raw

    def test_session_resumed_dim_notice(self, tmp_path, capsys):
        path = tmp_path / "s.jsonl"
        SessionManager(path).append("user", "hi")
        sm = SessionManager(path)  # reload
        runner = HookRunner()
        runner.load(ui_hook)
        runner.fire("session_start", {"cwd": "/"}, {"session": sm})
        assert "resumed" in capsys.readouterr().out

    def test_session_start_silent_for_new(self, tmp_path, capsys):
        sm = SessionManager(tmp_path / "s.jsonl")
        runner = HookRunner()
        runner.load(ui_hook)
        runner.fire("session_start", {"cwd": "/"}, {"session": sm})
        assert "resumed" not in capsys.readouterr().out

    def test_session_end_announces_save(self, tmp_path, capsys):
        sm = SessionManager(tmp_path / "s.jsonl")
        runner = HookRunner()
        runner.load(ui_hook)
        runner.fire("session_end", {}, {"session": sm})
        assert "session saved to" in capsys.readouterr().out

    def test_cache_stats_on_message_end(self, capsys):
        runner = HookRunner()
        runner.load(ui_hook)
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
        out = capsys.readouterr().out
        assert "read=2048" in out and "write=512" in out and "input=10" in out

    def test_edit_render_shows_diff(self, tmp_path, capsys):
        runner = HookRunner()
        self._register_tools(runner)
        runner.load(ui_hook)
        f = tmp_path / "f.txt"
        f.write_text("a\nold\nc\n")
        state: dict = {}
        runner.fire(
            "pre_tool_use",
            {"name": "edit", "input": {"path": str(f)}, "state": state},
        )
        f.write_text("a\nnew\nc\n")
        runner.fire(
            "post_tool_use",
            {
                "name": "edit",
                "input": {"path": str(f)},
                "content": "edited",
                "is_error": False,
                "state": state,
            },
        )
        out = capsys.readouterr().out
        assert "Update" in out
        assert "old" in out and "new" in out
        assert "+1" in out and "-1" in out

    def test_edit_diff_preserves_bracket_content(self, tmp_path, capsys):
        """Rich treats [anything] as markup. Without escaping, a Rust attribute
        like `#[allow(dead_code)]` renders as just `#` because Rich swallows
        the bracket content as an (invalid) style tag. Also bites Python
        type hints (`list[str]`), docstring refs, etc."""
        runner = HookRunner()
        self._register_tools(runner)
        runner.load(ui_hook)
        f = tmp_path / "f.rs"
        f.write_text("pub async fn work() {}\n")
        state: dict = {}
        runner.fire(
            "pre_tool_use",
            {"id": "t1", "name": "edit", "input": {"path": str(f)}, "state": state},
        )
        f.write_text("#[allow(dead_code)]\npub async fn work() {}\n")
        runner.fire(
            "post_tool_use",
            {
                "id": "t1",
                "name": "edit",
                "input": {"path": str(f)},
                "content": "ok",
                "is_error": False,
                "state": state,
            },
        )
        out = capsys.readouterr().out
        assert (
            "#[allow(dead_code)]" in out
        ), f"Rich markup ate the attribute — output was:\n{out}"

    def test_interleaved_pre_calls_attribute_diffs_by_id(self, tmp_path, capsys):
        """Two edits in flight simultaneously: pre(t1), pre(t2), post(t1), post(t2).
        Each call carries its own `state` dict (as agent_loop would create),
        so diffs never cross-attribute — and unlike the old id-keyed stack,
        there's no shared dict in ui_hook that could desync."""
        runner = HookRunner()
        self._register_tools(runner)
        runner.load(ui_hook)

        f_alpha = tmp_path / "alpha.txt"
        f_alpha.write_text("ALPHAOLD\n")
        f_beta = tmp_path / "beta.txt"
        f_beta.write_text("BETAOLD\n")

        state_t1: dict = {}
        state_t2: dict = {}
        runner.fire(
            "pre_tool_use",
            {
                "id": "t1",
                "name": "edit",
                "input": {"path": str(f_alpha)},
                "state": state_t1,
            },
        )
        runner.fire(
            "pre_tool_use",
            {
                "id": "t2",
                "name": "edit",
                "input": {"path": str(f_beta)},
                "state": state_t2,
            },
        )
        f_alpha.write_text("ALPHANEW\n")
        f_beta.write_text("BETANEW\n")
        runner.fire(
            "post_tool_use",
            {
                "id": "t1",
                "name": "edit",
                "input": {"path": str(f_alpha)},
                "content": "ok",
                "is_error": False,
                "state": state_t1,
            },
        )
        runner.fire(
            "post_tool_use",
            {
                "id": "t2",
                "name": "edit",
                "input": {"path": str(f_beta)},
                "content": "ok",
                "is_error": False,
                "state": state_t2,
            },
        )

        out = capsys.readouterr().out
        for token in ("ALPHAOLD", "ALPHANEW", "BETAOLD", "BETANEW"):
            assert out.count(token) == 1, f"{token} not printed exactly once:\n{out}"

        i_ao = out.index("ALPHAOLD")
        i_an = out.index("ALPHANEW")
        i_bo = out.index("BETAOLD")
        i_bn = out.index("BETANEW")
        assert (
            i_ao < i_an < i_bo < i_bn
        ), f"diffs mis-attributed across interleaved calls:\n{out}"

    def test_render_call_exception_does_not_desync_other_tools(self, tmp_path, capsys):
        """A raising render_call must not contaminate another tool's state.
        Each call has its own state dict by construction, so corruption of
        flaky's dict can't reach edit's dict."""

        def render_boom(_tool, _args, _console, _state):
            raise RuntimeError("boom")

        flaky = Tool(
            name="flaky",
            description="",
            schema={"type": "object"},
            execute=lambda a: ("ok", False),
            render_call=render_boom,
        )
        runner = HookRunner()
        runner.api.register_tool(flaky)
        self._register_tools(runner)  # adds real `edit`
        runner.load(ui_hook)

        f = tmp_path / "f.txt"
        f.write_text("before\n")

        flaky_state: dict = {}
        edit_state: dict = {}
        runner.fire(
            "pre_tool_use",
            {"id": "flaky-1", "name": "flaky", "input": {}, "state": flaky_state},
        )
        runner.fire(
            "pre_tool_use",
            {
                "id": "edit-1",
                "name": "edit",
                "input": {"path": str(f)},
                "state": edit_state,
            },
        )
        f.write_text("after\n")
        runner.fire(
            "post_tool_use",
            {
                "id": "flaky-1",
                "name": "flaky",
                "input": {},
                "content": "ok",
                "is_error": False,
                "state": flaky_state,
            },
        )
        runner.fire(
            "post_tool_use",
            {
                "id": "edit-1",
                "name": "edit",
                "input": {"path": str(f)},
                "content": "ok",
                "is_error": False,
                "state": edit_state,
            },
        )

        out = capsys.readouterr().out
        assert "before" in out and "after" in out
        assert "+1" in out and "-1" in out


# ═══════════════════════════════════════════════════════════════════════════
# Per-tool-call state bag — shared dict threaded through pre/post events
# ═══════════════════════════════════════════════════════════════════════════


class TestToolState:
    """`state` is a dict passed through pre_tool_use and post_tool_use events
    for the same tool call. Renderers stash data for themselves; other hooks
    (telemetry, instrumentation) can participate without special plumbing."""

    def _minimal_tool(self, **kwargs) -> Tool:
        return Tool(
            name=kwargs.pop("name", "t"),
            description="",
            schema={},
            execute=lambda a: ("ok", False),
            **kwargs,
        )

    def test_render_call_stashes_value_read_by_render_result(self):
        seen: list = []

        def on_call(_tool, _args, _console, state):
            state["marker"] = "call-was-here"

        def on_result(_tool, _args, _content, _is_error, _console, state):
            seen.append(state.get("marker"))

        runner = HookRunner()
        runner.api.register_tool(
            self._minimal_tool(render_call=on_call, render_result=on_result)
        )
        runner.load(ui_hook)

        state: dict = {}
        runner.fire(
            "pre_tool_use", {"id": "x", "name": "t", "input": {}, "state": state}
        )
        runner.fire(
            "post_tool_use",
            {
                "id": "x",
                "name": "t",
                "input": {},
                "content": "",
                "is_error": False,
                "state": state,
            },
        )
        assert seen == ["call-was-here"]

    def test_non_renderer_hook_can_contribute_to_state(self):
        """A hook with no renderer can stash data into `state` during
        pre_tool_use and have a renderer read it in post_tool_use."""
        durations: list = []

        def on_result(_tool, _args, _content, _is_error, _console, state):
            durations.append(state.get("tag"))

        runner = HookRunner()
        runner.api.register_tool(self._minimal_tool(render_result=on_result))
        runner.load(ui_hook)

        def stamp(event, _ctx):
            event["state"]["tag"] = "stamped-by-non-renderer"

        runner.api.on("pre_tool_use", stamp)

        state: dict = {}
        runner.fire(
            "pre_tool_use", {"id": "x", "name": "t", "input": {}, "state": state}
        )
        runner.fire(
            "post_tool_use",
            {
                "id": "x",
                "name": "t",
                "input": {},
                "content": "",
                "is_error": False,
                "state": state,
            },
        )
        assert durations == ["stamped-by-non-renderer"]

    def test_each_tool_use_has_isolated_state(self):
        """Two concurrent calls must not share state — the caller (agent_loop)
        is responsible for supplying a fresh dict per tool_use."""
        observations: list = []

        def on_call(_tool, args, _console, state):
            state["pre"] = args["tag"]

        def on_result(_tool, args, _content, _is_error, _console, state):
            observations.append((args["tag"], state.get("pre")))

        runner = HookRunner()
        runner.api.register_tool(
            self._minimal_tool(render_call=on_call, render_result=on_result)
        )
        runner.load(ui_hook)

        state_a: dict = {}
        state_b: dict = {}
        for id_, tag, state in [("a", "A", state_a), ("b", "B", state_b)]:
            runner.fire(
                "pre_tool_use",
                {"id": id_, "name": "t", "input": {"tag": tag}, "state": state},
            )
        for id_, tag, state in [("a", "A", state_a), ("b", "B", state_b)]:
            runner.fire(
                "post_tool_use",
                {
                    "id": id_,
                    "name": "t",
                    "input": {"tag": tag},
                    "content": "",
                    "is_error": False,
                    "state": state,
                },
            )
        assert observations == [("A", "A"), ("B", "B")]

    def test_missing_state_falls_back_to_fresh_empty_dict(self):
        """Tests (and any caller not bothering to share state) still work —
        ui_hook defaults to an empty dict per call. Just no cross-phase sharing."""
        runner = HookRunner()
        runner.api.register_tool(self._minimal_tool())
        runner.load(ui_hook)
        # Should not raise.
        runner.fire("pre_tool_use", {"id": "x", "name": "t", "input": {}})
        runner.fire(
            "post_tool_use",
            {
                "id": "x",
                "name": "t",
                "input": {},
                "content": "ok",
                "is_error": False,
            },
        )


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

    def test_max_turns_defaults_to_25(self, tmp_path):
        sm = SessionManager(tmp_path / "s.jsonl")
        agent = AgentSession(client=None, model="m", session=sm)  # type: ignore[arg-type]
        assert agent.max_turns == 25

    def test_max_turns_override(self, tmp_path):
        sm = SessionManager(tmp_path / "s.jsonl")
        agent = AgentSession(client=None, model="m", session=sm, max_turns=100)  # type: ignore[arg-type]
        assert agent.max_turns == 100

    def test_max_turns_threaded_to_agent_loop(self, tmp_path, monkeypatch):
        """AgentSession.prompt must forward max_turns as the 8th positional arg
        to agent_loop — otherwise the flag silently has no effect."""
        import agent as agent_mod
        import asyncio

        captured = {}

        async def fake_agent_loop(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        monkeypatch.setattr(agent_mod, "agent_loop", fake_agent_loop)

        sm = SessionManager(tmp_path / "s.jsonl")
        runner = HookRunner()
        agent = AgentSession(
            client=None, model="m", session=sm, runner=runner, max_turns=7
        )  # type: ignore[arg-type]
        asyncio.run(agent.prompt("hi"))

        # Position 7 (0-indexed) is pending_reminders; position 8 is max_turns.
        # We assert by value rather than by index to stay robust to signature churn.
        all_vals = list(captured["args"]) + list(captured["kwargs"].values())
        assert 7 in all_vals, (
            f"max_turns=7 not forwarded to agent_loop: args={captured['args']} "
            f"kwargs={captured['kwargs']}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Runtime flag hooks — model_flag_hook, max_turns_flag_hook, strict_hooks_flag_hook
# ═══════════════════════════════════════════════════════════════════════════


class TestModelFlag:
    def _loaded(self):
        runner = HookRunner()
        runner.load(model_flag_hook)
        return runner

    def test_default(self):
        args = self._loaded().parser.parse_args([])
        assert args.model == "claude-sonnet-4-6"

    def test_override(self):
        args = self._loaded().parser.parse_args(
            ["--model", "claude-haiku-4-5-20251001"]
        )
        assert args.model == "claude-haiku-4-5-20251001"

    def test_contributes_to_build_session_config(self):
        runner = self._loaded()
        args = runner.parser.parse_args(["--model", "custom-m"])
        result = runner.fire("build_session_config", {"args": args})
        assert result == {"model": "custom-m"}


class TestMaxTurnsFlag:
    def _loaded(self):
        runner = HookRunner()
        runner.load(max_turns_flag_hook)
        return runner

    def test_default(self):
        args = self._loaded().parser.parse_args([])
        assert args.max_turns == 25

    def test_override(self):
        args = self._loaded().parser.parse_args(["--max-turns", "50"])
        assert args.max_turns == 50

    def test_rejects_non_int(self):
        runner = self._loaded()
        with pytest.raises(SystemExit):
            runner.parser.parse_args(["--max-turns", "not-a-number"])

    def test_contributes_to_build_session_config(self):
        runner = self._loaded()
        args = runner.parser.parse_args(["--max-turns", "50"])
        result = runner.fire("build_session_config", {"args": args})
        assert result == {"max_turns": 50}


class TestStrictHooksFlag:
    """--strict-hooks carries behavior in addition to a flag value: an
    args_parsed handler copies it into runner.strict. Each step has its
    own assertion so a regression points at the broken link."""

    def _loaded(self):
        runner = HookRunner()
        runner.load(strict_hooks_flag_hook)
        return runner

    def test_default_false(self):
        args = self._loaded().parser.parse_args([])
        assert args.strict_hooks is False

    def test_flag_sets_true(self):
        args = self._loaded().parser.parse_args(["--strict-hooks"])
        assert args.strict_hooks is True

    def test_args_parsed_applies_strict_to_runner(self):
        runner = self._loaded()
        args = runner.parser.parse_args(["--strict-hooks"])
        assert runner.strict is False  # not yet applied
        runner.fire("args_parsed", {"args": args})
        assert runner.strict is True

    def test_args_parsed_preserves_non_strict_default(self):
        runner = self._loaded()
        args = runner.parser.parse_args([])
        runner.fire("args_parsed", {"args": args})
        assert runner.strict is False

    def test_end_to_end(self):
        """Parse --strict-hooks, fire args_parsed, then a raising handler
        propagates instead of being logged."""
        runner = self._loaded()
        args = runner.parser.parse_args(["--strict-hooks"])
        runner.fire("args_parsed", {"args": args})
        runner.api.on("text_delta", lambda e, c: 1 / 0)
        with pytest.raises(ZeroDivisionError):
            runner.fire("text_delta", {"text": "x"})


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


# ═══════════════════════════════════════════════════════════════════════════
# Entry point: `uv run test_agent.py`
# ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
