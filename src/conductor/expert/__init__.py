"""Conductor Expert — reusable knowledge base for Conductor-aware agents.

This module provides opt-in access to Conductor's bundled knowledge base,
giving agents deep understanding of the YAML schema, execution model,
authoring patterns, and CLI commands. Agents that need to evaluate, improve,
debug, or generate Conductor workflows can enable it via the
``conductor_expert`` flag at the agent or workflow level.

See :mod:`conductor.expert.loader` for loading and caching details.
"""

from conductor.expert.loader import load_expert_knowledge

__all__ = ["load_expert_knowledge"]
