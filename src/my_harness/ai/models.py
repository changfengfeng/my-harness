"""Model descriptors and lookup."""

from __future__ import annotations

from dataclasses import dataclass

KNOWN_PROVIDERS = ("anthropic", "openai", "faux")

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-5",
    "faux": "faux",
}


@dataclass(frozen=True)
class Model:
    provider: str
    id: str
    base_url: str | None = None
    context_window: int = 200_000
    max_output_tokens: int = 16_384
    reasoning: bool = True

    def __str__(self) -> str:
        return f"{self.provider}:{self.id}"


def get_model(spec: str, base_url: str | None = None) -> Model:
    """Parse `provider:model_id` (or just `provider`) into a Model."""
    provider, _, model_id = spec.partition(":")
    if provider not in KNOWN_PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}. Known: {', '.join(KNOWN_PROVIDERS)}")
    return Model(provider=provider, id=model_id or DEFAULT_MODELS[provider], base_url=base_url)
