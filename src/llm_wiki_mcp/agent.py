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
SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills" / "llm-wiki-agent"


async def run_ingest(wiki_workdir: Path, source: str) -> str:
    """Ingest a source into the wiki via the /wiki-ingest skill."""
    return await _run_agent(f"/wiki-ingest {source}", wiki_workdir)


async def run_query(wiki_workdir: Path, question: str) -> str:
    """Query the wiki via the /wiki-query skill."""
    return await _run_agent(f'/wiki-query "{question}"', wiki_workdir)


async def _run_agent(prompt: str, cwd: Path) -> str:
    """Run a Claude Agent SDK session with llm-wiki-agent skills."""
    import shutil
    cwd.mkdir(parents=True, exist_ok=True)

    # 把 llm-wiki-agent 的命令文件复制到工作目录，让 Claude Code 能发现
    commands_src = SKILLS_DIR / ".claude" / "commands"
    commands_dst = cwd / ".claude" / "commands"
    if commands_src.exists():
        commands_dst.mkdir(parents=True, exist_ok=True)
        for f in commands_src.glob("*.md"):
            shutil.copy(f, commands_dst / f.name)

    options = ClaudeAgentOptions(
        cwd=str(cwd),
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="acceptEdits",
        skills="all",
        plugins=[{"type": "local", "path": str(SKILLS_DIR)}],
    )

    text_parts: list[str] = []
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
            elif isinstance(message, ResultMessage):
                if message.result:
                    text_parts.append(message.result)
    except Exception as e:
        if "success" in str(e).lower() and text_parts:
            pass  # 实际执行成功，忽略此错误
        else:
            raise

    return "\n".join(text_parts) if text_parts else "No response from agent."
