"""Rollout detail model for RL training data collection."""

from typing import TypedDict


class RolloutDetail(TypedDict, total=False):
    """Token-level details of a complete chat history.

    Stores the full conversation as token IDs and log-probabilities so that
    downstream RL training pipelines can reconstruct the trajectory without
    needing raw text.
    """

    prompt_token_ids: list[list[int]]
    """Per-turn full prompt token IDs (including chat history)."""

    completion_token_ids: list[list[int]]
    """Per-turn generated response token IDs."""

    logprobs: list[list[float]]
    """Per-turn log-probabilities aligned with completion_token_ids."""
