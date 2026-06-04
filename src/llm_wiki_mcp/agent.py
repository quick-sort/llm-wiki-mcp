"""Claude Agent SDK wrapper — delegates wiki operations to llm-wiki-agent skills."""

from __future__ import annotations

import shutil
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


def _setup_workdir(cwd: Path) -> None:
    """Copy llm-wiki-agent skill files into the wiki working directory."""
    cwd.mkdir(parents=True, exist_ok=True)

    # Copy .claude/commands/ (slash commands: wiki-ingest, wiki-query, etc.)
    src_cmds = SKILLS_DIR / ".claude" / "commands"
    dst_cmds = cwd / ".claude" / "commands"
    if src_cmds.exists() and not dst_cmds.exists():
        shutil.copytree(src_cmds, dst_cmds)

    # Copy CLAUDE.md / AGENTS.md (schema + instructions for the agent)
    for name in ("CLAUDE.md", "AGENTS.md"):
        src = SKILLS_DIR / name
        dst = cwd / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


async def run_ingest(wiki_workdir: Path, source: str) -> str:
    """Ingest a source into the wiki via the /wiki-ingest skill."""
    return await _run_agent(f"/wiki-ingest {source}", wiki_workdir)


async def run_query(wiki_workdir: Path, question: str) -> str:
    """Query the wiki via the /wiki-query skill."""
    return await _run_agent(f'/wiki-query "{question}"', wiki_workdir)


async def _run_agent(prompt: str, cwd: Path) -> str:
    """Run a Claude Agent SDK session with llm-wiki-agent skills."""
    _setup_workdir(cwd)

    options = ClaudeAgentOptions(
        cwd=str(cwd),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="acceptEdits",
        skills="all",
    )

    text_parts: list[str] = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
        elif isinstance(message, ResultMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)

    return "\n".join(text_parts) if text_parts else "No response from agent."
