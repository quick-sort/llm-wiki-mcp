"""Claude Agent SDK wrapper — delegates wiki operations to llm-wiki-agent skills."""

from __future__ import annotations

import os
from pathlib import Path

from claude_agent_sdk import query
from claude_agent_sdk.types import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
)

# Skill source — llm-wiki-agent cloned into skills/ at deploy time
SKILLS_DIR = Path(__file__).resolve().parents[3] / "skills" / "llm-wiki-agent"


async def run_ingest(wiki_workdir: Path, source: str) -> str:
    """Ingest a source into the wiki via the /wiki-ingest skill."""
    return await _run_agent(f"/wiki-ingest {source}", wiki_workdir)


async def run_query(wiki_workdir: Path, question: str) -> str:
    """Query the wiki via the /wiki-query skill."""
    return await _run_agent(f'/wiki-query "{question}"', wiki_workdir)


async def _run_agent(prompt: str, cwd: Path) -> str:
    """Run a Claude Agent SDK session with llm-wiki-agent skills."""
    cwd.mkdir(parents=True, exist_ok=True)

    env: dict[str, str] = {}
    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        env["ANTHROPIC_API_KEY"] = api_key
    if base_url := os.environ.get("ANTHROPIC_BASE_URL"):
        env["ANTHROPIC_BASE_URL"] = base_url

    options = ClaudeAgentOptions(
        cwd=str(cwd),
        model=os.environ.get("ANTHROPIC_MODEL"),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="acceptEdits",
        skills="all",
        env=env,
        add_dirs=[str(SKILLS_DIR)],
    )

    text_parts: list[str] = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
        elif isinstance(message, ResultMessage):
            if message.result:
                text_parts.append(message.result)

    return "\n".join(text_parts) if text_parts else "No response from agent."
