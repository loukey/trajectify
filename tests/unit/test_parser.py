"""Tests for the Terminus response parser."""

import pytest

from trajectify.agents.terminus.parser import ParseError, parse_response


def test_parse_direct_json():
    text = '{"analysis": "ok", "plan": "run ls", "commands": [{"keystrokes": "ls\\n", "duration": 0.5}], "is_task_complete": false}'
    result = parse_response(text)
    assert result.analysis == "ok"
    assert result.plan == "run ls"
    assert len(result.commands) == 1
    assert result.commands[0].keystrokes == "ls\n"
    assert result.is_task_complete is False


def test_parse_markdown_fence():
    text = 'Here is my response:\n```json\n{"analysis": "x", "plan": "y", "commands": [], "is_task_complete": true}\n```'
    result = parse_response(text)
    assert result.analysis == "x"
    assert result.is_task_complete is True
    assert result.commands == []


def test_parse_embedded_json():
    text = 'Some text before {"analysis": "a", "plan": "b", "commands": [{"keystrokes": "echo hi\\n", "duration": 1}]} some text after'
    result = parse_response(text)
    assert result.analysis == "a"
    assert len(result.commands) == 1


def test_parse_missing_commands_raises():
    with pytest.raises(ParseError, match="missing 'commands'"):
        parse_response('{"analysis": "a", "plan": "b"}')


def test_parse_invalid_json_raises():
    with pytest.raises(ParseError):
        parse_response("this is not json at all")


def test_parse_default_duration():
    text = '{"analysis": "", "plan": "", "commands": [{"keystrokes": "ls\\n"}]}'
    result = parse_response(text)
    assert result.commands[0].duration == 0.5
