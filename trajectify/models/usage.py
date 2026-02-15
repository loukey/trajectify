"""Unified UsageInfo model — single source of truth for token usage and cost."""

from __future__ import annotations

from pydantic import BaseModel


class UsageInfo(BaseModel):
    """Token usage and cost for a single LLM call.

    This is the unified model used everywhere in trajectify — both in LLM
    responses and in trajectory step metrics.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_tokens: int = 0
    cost_usd: float = 0.0
