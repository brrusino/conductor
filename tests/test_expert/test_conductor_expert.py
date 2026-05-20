"""Tests for the Conductor Expert knowledge base feature.

Covers:
- Knowledge loader: loading, caching, wrapper tags
- Schema: field acceptance and type-based validation
- Executor integration: expert knowledge injection via prompt prefix
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from conductor.config.schema import (
    AgentDef,
    GateOption,
    RuntimeConfig,
)
from conductor.executor.agent import AgentExecutor
from conductor.expert.loader import load_expert_knowledge
from conductor.providers.copilot import CopilotProvider

# ---------------------------------------------------------------------------
# Knowledge loader tests
# ---------------------------------------------------------------------------


class TestLoadExpertKnowledge:
    """Tests for the Conductor Expert knowledge loader."""

    def setup_method(self) -> None:
        """Clear the lru_cache before each test."""
        load_expert_knowledge.cache_clear()

    def test_loads_successfully(self) -> None:
        """Knowledge base loads without error."""
        result = load_expert_knowledge()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_wrapped_in_conductor_knowledge_tags(self) -> None:
        """Result is wrapped in <conductor_knowledge> tags."""
        result = load_expert_knowledge()
        assert result.startswith("<conductor_knowledge>\n")
        assert "</conductor_knowledge>" in result

    def test_contains_all_three_documents(self) -> None:
        """Result includes content from yaml-schema, authoring, and execution docs."""
        result = load_expert_knowledge()
        assert "# Knowledge: yaml-schema.md" in result
        assert "# Knowledge: authoring.md" in result
        assert "# Knowledge: execution.md" in result

    def test_contains_header_text(self) -> None:
        """Result includes the descriptive header."""
        result = load_expert_knowledge()
        assert "Conductor knowledge base" in result
        assert "YAML workflow schema" in result

    def test_documents_separated_by_dividers(self) -> None:
        """Documents are separated by --- dividers."""
        result = load_expert_knowledge()
        assert "\n\n---\n\n" in result

    def test_caching_returns_same_object(self) -> None:
        """Subsequent calls return the cached object (same id)."""
        first = load_expert_knowledge()
        second = load_expert_knowledge()
        assert first is second

    def test_substantial_content_size(self) -> None:
        """Knowledge base contains substantial content (>50KB from three docs)."""
        result = load_expert_knowledge()
        size_kb = len(result.encode("utf-8")) / 1024
        assert size_kb > 50, f"Expected >50KB, got {size_kb:.1f}KB"

    def test_raises_on_missing_knowledge_file(self) -> None:
        """Raises RuntimeError with reinstall guidance when a doc is missing."""
        from unittest.mock import patch

        load_expert_knowledge.cache_clear()

        original_fn = load_expert_knowledge.__wrapped__

        with (
            patch("conductor.expert.loader._KNOWLEDGE_DOCS", ["nonexistent-doc.md"]),
            pytest.raises(RuntimeError, match="missing or unreadable"),
        ):
            original_fn()

        load_expert_knowledge.cache_clear()


# ---------------------------------------------------------------------------
# Schema tests — AgentDef.conductor_expert
# ---------------------------------------------------------------------------


class TestAgentDefConductorExpert:
    """Tests for the conductor_expert field on AgentDef."""

    def test_defaults_to_none(self) -> None:
        """conductor_expert defaults to None (inherit from workflow)."""
        agent = AgentDef(name="a", model="gpt-4", prompt="Hello")
        assert agent.conductor_expert is None

    def test_explicit_true(self) -> None:
        """conductor_expert can be set to True."""
        agent = AgentDef(name="a", model="gpt-4", prompt="Hello", conductor_expert=True)
        assert agent.conductor_expert is True

    def test_explicit_false(self) -> None:
        """conductor_expert can be set to False (explicit opt-out)."""
        agent = AgentDef(name="a", model="gpt-4", prompt="Hello", conductor_expert=False)
        assert agent.conductor_expert is False

    def test_forbidden_on_script_agent(self) -> None:
        """script agents cannot have conductor_expert."""
        with pytest.raises(ValidationError, match="script agents cannot have 'conductor_expert'"):
            AgentDef(
                name="s",
                type="script",
                command="echo hi",
                conductor_expert=True,
            )

    def test_forbidden_on_workflow_agent(self) -> None:
        """workflow agents cannot have conductor_expert."""
        with pytest.raises(ValidationError, match="workflow agents cannot have 'conductor_expert'"):
            AgentDef(
                name="w",
                type="workflow",
                workflow="sub.yaml",
                conductor_expert=True,
            )

    def test_forbidden_on_human_gate(self) -> None:
        """human_gate agents cannot have conductor_expert."""
        with pytest.raises(
            ValidationError, match="human_gate agents cannot have 'conductor_expert'"
        ):
            AgentDef(
                name="g",
                type="human_gate",
                prompt="Choose:",
                options=[GateOption(label="Yes", value="y", route="next")],
                conductor_expert=True,
            )

    def test_allowed_on_regular_agent(self) -> None:
        """Regular (provider-backed) agents accept conductor_expert."""
        agent = AgentDef(
            name="reviewer",
            type="agent",
            model="gpt-4",
            prompt="Review this workflow",
            conductor_expert=True,
        )
        assert agent.conductor_expert is True

    def test_allowed_on_default_type_agent(self) -> None:
        """Agents with no explicit type accept conductor_expert."""
        agent = AgentDef(
            name="reviewer",
            model="gpt-4",
            prompt="Review this workflow",
            conductor_expert=True,
        )
        assert agent.conductor_expert is True
        assert agent.type is None


# ---------------------------------------------------------------------------
# Schema tests — RuntimeConfig.conductor_expert
# ---------------------------------------------------------------------------


class TestRuntimeConfigConductorExpert:
    """Tests for the conductor_expert field on RuntimeConfig."""

    def test_defaults_to_false(self) -> None:
        """conductor_expert defaults to False."""
        config = RuntimeConfig()
        assert config.conductor_expert is False

    def test_can_be_enabled(self) -> None:
        """conductor_expert can be set to True."""
        config = RuntimeConfig(conductor_expert=True)
        assert config.conductor_expert is True


# ---------------------------------------------------------------------------
# Executor integration tests
# ---------------------------------------------------------------------------


class TestExecutorConductorExpert:
    """Tests for Conductor Expert injection in AgentExecutor."""

    def setup_method(self) -> None:
        """Clear the knowledge cache before each test."""
        load_expert_knowledge.cache_clear()

    def test_not_injected_by_default(self) -> None:
        """Expert knowledge is NOT injected when neither flag is set."""
        provider = CopilotProvider()
        executor = AgentExecutor(provider)

        agent = AgentDef(name="a", model="gpt-4", prompt="Hello world")
        context: dict = {}
        rendered = executor.render_prompt(agent, context)

        assert "<conductor_knowledge>" not in rendered
        assert "Hello world" in rendered

    def test_injected_when_agent_flag_true(self) -> None:
        """Expert knowledge IS injected when agent.conductor_expert=True."""
        provider = CopilotProvider()
        executor = AgentExecutor(provider)

        agent = AgentDef(name="a", model="gpt-4", prompt="Hello world", conductor_expert=True)
        context: dict = {}
        rendered = executor.render_prompt(agent, context)

        assert "<conductor_knowledge>" in rendered
        assert "</conductor_knowledge>" in rendered
        assert "Hello world" in rendered

    def test_injected_when_workflow_default_true(self) -> None:
        """Expert knowledge IS injected via workflow-level default."""
        provider = CopilotProvider()
        executor = AgentExecutor(provider, conductor_expert_default=True)

        agent = AgentDef(name="a", model="gpt-4", prompt="Hello world")
        context: dict = {}
        rendered = executor.render_prompt(agent, context)

        assert "<conductor_knowledge>" in rendered
        assert "Hello world" in rendered

    def test_agent_false_overrides_workflow_true(self) -> None:
        """Agent conductor_expert=False overrides workflow default=True."""
        provider = CopilotProvider()
        executor = AgentExecutor(provider, conductor_expert_default=True)

        agent = AgentDef(name="a", model="gpt-4", prompt="Hello world", conductor_expert=False)
        context: dict = {}
        rendered = executor.render_prompt(agent, context)

        assert "<conductor_knowledge>" not in rendered
        assert "Hello world" in rendered

    def test_agent_true_overrides_workflow_false(self) -> None:
        """Agent conductor_expert=True works even when workflow default=False."""
        provider = CopilotProvider()
        executor = AgentExecutor(provider, conductor_expert_default=False)

        agent = AgentDef(name="a", model="gpt-4", prompt="Hello world", conductor_expert=True)
        context: dict = {}
        rendered = executor.render_prompt(agent, context)

        assert "<conductor_knowledge>" in rendered
        assert "Hello world" in rendered

    def test_expert_appears_before_prompt(self) -> None:
        """Expert knowledge appears before the agent's rendered prompt."""
        provider = CopilotProvider()
        executor = AgentExecutor(provider, conductor_expert_default=True)

        agent = AgentDef(name="a", model="gpt-4", prompt="MY_PROMPT_HERE")
        context: dict = {}
        rendered = executor.render_prompt(agent, context)

        expert_pos = rendered.index("<conductor_knowledge>")
        prompt_pos = rendered.index("MY_PROMPT_HERE")
        assert expert_pos < prompt_pos

    def test_expert_appears_after_instructions_preamble(self) -> None:
        """Expert knowledge comes after workspace instructions but before prompt."""
        provider = CopilotProvider()
        preamble = "<workspace_instructions>\nFollow conventions.\n</workspace_instructions>\n\n"
        executor = AgentExecutor(
            provider,
            instructions_preamble=preamble,
            conductor_expert_default=True,
        )

        agent = AgentDef(name="a", model="gpt-4", prompt="MY_PROMPT_HERE")
        context: dict = {}
        rendered = executor.render_prompt(agent, context)

        instructions_pos = rendered.index("<workspace_instructions>")
        expert_pos = rendered.index("<conductor_knowledge>")
        prompt_pos = rendered.index("MY_PROMPT_HERE")
        assert instructions_pos < expert_pos < prompt_pos

    def test_should_inject_expert_tristate_logic(self) -> None:
        """Test the _should_inject_expert tri-state resolution."""
        provider = CopilotProvider()

        # Workflow default False, agent None → False
        executor = AgentExecutor(provider, conductor_expert_default=False)
        agent_none = AgentDef(name="a", model="gpt-4", prompt="p")
        assert executor._should_inject_expert(agent_none) is False

        # Workflow default True, agent None → True
        executor = AgentExecutor(provider, conductor_expert_default=True)
        assert executor._should_inject_expert(agent_none) is True

        # Agent explicitly True overrides any default
        agent_true = AgentDef(name="a", model="gpt-4", prompt="p", conductor_expert=True)
        executor = AgentExecutor(provider, conductor_expert_default=False)
        assert executor._should_inject_expert(agent_true) is True

        # Agent explicitly False overrides any default
        agent_false = AgentDef(name="a", model="gpt-4", prompt="p", conductor_expert=False)
        executor = AgentExecutor(provider, conductor_expert_default=True)
        assert executor._should_inject_expert(agent_false) is False
