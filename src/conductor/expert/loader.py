"""Load and cache the bundled Conductor knowledge base.

The knowledge base consists of reference docs originally authored for the
Conductor Claude-Code plugin and bundled as package data under
``conductor/expert/knowledge/``.  Content is loaded once per process via
:func:`functools.lru_cache` and wrapped in ``<conductor_knowledge>`` tags
for clean separation from workspace instructions.

Phase 1 bundles three documents:

* ``yaml-schema.md`` — complete YAML field reference
* ``authoring.md`` — authoring patterns and best practices
* ``execution.md`` — CLI commands, debugging, checkpoint/resume
"""

from __future__ import annotations

import functools
import logging
from importlib import resources

logger = logging.getLogger(__name__)

# Documents to include, in presentation order.
_KNOWLEDGE_DOCS = [
    "yaml-schema.md",
    "authoring.md",
    "execution.md",
]

_HEADER = (
    "The following is the Conductor knowledge base — comprehensive reference "
    "documentation for Conductor's YAML workflow schema, execution model, "
    "authoring patterns, and CLI commands. Use this knowledge when evaluating, "
    "improving, debugging, or generating Conductor workflows."
)


@functools.lru_cache(maxsize=1)
def load_expert_knowledge() -> str:
    """Load the bundled Conductor knowledge base and return it as a tagged string.

    The result is wrapped in ``<conductor_knowledge>`` tags and cached for
    the lifetime of the process (subsequent calls return the same string
    with zero I/O).

    Returns:
        A string containing all knowledge documents separated by ``---``
        dividers and wrapped in XML-style tags.
    """
    knowledge_pkg = resources.files("conductor.expert") / "knowledge"
    sections: list[str] = []

    for name in _KNOWLEDGE_DOCS:
        resource = knowledge_pkg / name
        text = resource.read_text(encoding="utf-8").strip()
        if text:
            sections.append(f"# Knowledge: {name}\n\n{text}")

    combined = "\n\n---\n\n".join(sections)

    total_size_kb = len(combined.encode("utf-8")) / 1024
    logger.info(
        "Loaded Conductor Expert knowledge base (%.1fKB from %d documents)",
        total_size_kb,
        len(sections),
    )

    return f"<conductor_knowledge>\n{_HEADER}\n\n{combined}\n</conductor_knowledge>\n\n"
