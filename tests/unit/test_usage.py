"""Tests for the unified UsageInfo model."""

from trajectify.models.usage import UsageInfo


def test_usage_defaults():
    u = UsageInfo()
    assert u.prompt_tokens == 0
    assert u.completion_tokens == 0
    assert u.cache_tokens == 0
    assert u.cost_usd == 0.0


def test_usage_with_values():
    u = UsageInfo(prompt_tokens=100, completion_tokens=50, cache_tokens=20, cost_usd=0.5)
    assert u.prompt_tokens == 100
    assert u.completion_tokens == 50
    assert u.cache_tokens == 20
    assert u.cost_usd == 0.5


def test_usage_serialization():
    u = UsageInfo(prompt_tokens=10, completion_tokens=5)
    data = u.model_dump()
    assert data["prompt_tokens"] == 10
    assert data["completion_tokens"] == 5

    u2 = UsageInfo.model_validate(data)
    assert u2 == u
