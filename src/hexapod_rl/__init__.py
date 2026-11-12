"""Simulator-independent components. Importing configuration does not load torch."""

from .config import ActorCriticConfig

__all__ = ["ActorCriticConfig"]
