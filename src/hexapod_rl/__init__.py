"""Simulator-independent components. Importing configuration does not load torch."""

from .config import ActorCriticConfig, PPOConfig

__all__ = ["ActorCriticConfig", "PPOConfig"]
