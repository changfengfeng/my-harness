"""my-harness: a minimal coding agent harness.

Layers (each depends only on the layers above it):
    my_harness.ai      unified multi-provider LLM streaming API
    my_harness.agent   agent loop: tool calling, hooks, events, state
    my_harness.coding  coding-agent harness: tools, prompt, sessions, CLI
"""

__version__ = "0.1.0"
