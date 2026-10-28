"""Unified multi-provider LLM streaming API (counterpart of pi-ai)."""

from .models import Model, get_model
from .providers.faux import FauxProvider
from .stream import complete, register_provider, stream
from .types import *  # noqa: F403

__all__ = ["FauxProvider", "Model", "complete", "get_model", "register_provider", "stream"]
