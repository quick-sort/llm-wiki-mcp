"""HTTP MCP server for llm-wiki-agent — provides query and ingest tools."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP
from markitdown import MarkItDown

from .agent import run_ingest, run_query

# Mutable config — use _cfg["wikis_root"] everywhere
_cfg: dict = {
    "wikis_root": os.environ.get("LLM_WIKI_ROOT", str(Path.cwd() / "wikis")),
}

_MARKDOWN_EXTENSIONS = {".md", ".markdown", ".txt"}


def _get_markitdown() -> MarkItDown:
    """Create a MarkItDown instance, optionally with LLM client for enhanced conversion."""
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")

    if api_key and base_url:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        return MarkItDown(llm_client=client, llm_model=model)

    return MarkItDown()


def _save_raw(wiki_dir: Path, name: str, content: str) -> Path:
    """Save raw content to wiki's raw/ directory and return the file path."""
    raw_dir = wiki_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace("/", "_").replace(" ", "_")
    path = raw_dir / f"{safe_name}.md"
    path.write_text(content)
    return path


def _convert_file(file_path: str) -> tuple[str, str]:
    """Convert a non-markdown file to markdown using markitdown.

    Returns (name, markdown_content).
    """
    p = Path(file_path)
    name = p.stem
    md = _get_markitdown()
    result = md.convert(str(p))
    return name, result.text_content


async def _fetch_url(url: str) -> tuple[str, str]:
    """Fetch content from a URL, return (name, content)."""
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    name = url.rstrip("/").split("/")[-1] or "webpage"
    return name, resp.text


mcp = FastMCP(
    "llm-wiki-mcp",
    instructions=(
        "A wiki knowledge base MCP server. Use 'ingest' to add documents (files, URLs, or raw text) "
        "to a named wiki, and 'query' to ask questions about the wiki's knowledge. "
        "Powered by llm-wiki-agent skills via Claude Agent SDK."
    ),
)


@mcp.tool()
async def list_wikis() -> list[str]:
    """List all wiki names.

    Returns a list of wiki names found in the wikis root directory.
    """
    wikis_root = Path(_cfg["wikis_root"])
    if not wikis_root.exists():
        return []
    return sorted(
        d.name for d in wikis_root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )


@mcp.tool()
async def ingest(wiki_name: str, source: str = "", content: str = "") -> str:
    """Ingest content into a wiki knowledge base.

    Supports three input modes:
    - Local file path via `source` (PDF, DOCX, PPTX, etc. auto-converted via markitdown)
    - URL via `source` (starts with http:// or https://)
    - Raw text via `content`

    After saving, the llm-wiki-agent /wiki-ingest skill processes it — extracting
    entities, concepts, and cross-references into structured wiki pages.

    Args:
        wiki_name: Name of the wiki to ingest into (created if it doesn't exist).
        source: A local file path or URL to ingest.
        content: Raw text content to ingest. Used when source is empty.
    """
    if not source and not content:
        return "Error: provide either 'source' (file path or URL) or 'content' (raw text)."

    wiki_dir = Path(_cfg["wikis_root"]) / wiki_name

    if source.startswith(("http://", "https://")):
        name, text = await _fetch_url(source)
        local_path = _save_raw(wiki_dir, name, text)
        return await run_ingest(wiki_dir, str(local_path))

    if content:
        local_path = _save_raw(wiki_dir, f"{wiki_name}_raw", content)
        return await run_ingest(wiki_dir, str(local_path))

    # Local file — convert non-markdown via markitdown
    p = Path(source)
    if not p.exists():
        return f"Error: file not found: {source}"

    if p.suffix.lower() in _MARKDOWN_EXTENSIONS:
        return await run_ingest(wiki_dir, source)

    name, md_content = _convert_file(source)
    local_path = _save_raw(wiki_dir, name, md_content)
    return await run_ingest(wiki_dir, str(local_path))


@mcp.tool()
async def query(wiki_name: str, question: str) -> str:
    """Query a wiki knowledge base and get an AI-synthesized answer.

    Uses the llm-wiki-agent /wiki-query skill to search wiki pages and
    synthesize a comprehensive answer with [[wikilinks]] citations.

    Args:
        wiki_name: Name of the wiki to query.
        question: The question to ask about the wiki's knowledge.
    """
    wiki_dir = Path(_cfg["wikis_root"]) / wiki_name
    if not wiki_dir.exists():
        return f"Wiki '{wiki_name}' does not exist. Create it by ingesting a document first."
    return await run_query(wiki_dir, question)


def main():
    parser = argparse.ArgumentParser(description="llm-wiki-mcp server")
    parser.add_argument(
        "--wikis-root",
        default=_cfg["wikis_root"],
        help="Root directory for wiki storage (default: ./wikis)",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="streamable-http",
        help="Transport mode (default: streamable-http)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for HTTP transport (default: 8080)",
    )
    args = parser.parse_args()

    _cfg["wikis_root"] = args.wikis_root
    Path(args.wikis_root).mkdir(parents=True, exist_ok=True)

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
