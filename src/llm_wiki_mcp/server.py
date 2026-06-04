"""HTTP MCP server for llm-wiki-agent — provides query and ingest tools."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .agent import run_ingest, run_query

# Mutable config — use _cfg["wikis_root"] everywhere
_cfg: dict = {
    "wikis_root": os.environ.get("LLM_WIKI_ROOT", str(Path.cwd() / "wikis")),
}

mcp = FastMCP(
    "llm-wiki-mcp",
    instructions=(
        "A wiki knowledge base MCP server. Use 'ingest' to add documents (files or URLs) "
        "to a named wiki, and 'query' to ask questions about the wiki's knowledge. "
        "Powered by llm-wiki-agent skills via Claude Agent SDK."
    ),
)


@mcp.tool()
async def ingest(wiki_name: str, source: str) -> str:
    """Ingest a document into a wiki knowledge base.

    The source is fetched (URL or local path) and saved to the wiki's raw/ directory,
    then the llm-wiki-agent /wiki-ingest skill processes it — extracting entities,
    concepts, and cross-references into structured wiki pages.

    Args:
        wiki_name: Name of the wiki to ingest into (created if it doesn't exist).
        source: A local file path or URL to ingest.
    """
    wiki_dir = Path(_cfg["wikis_root"]) / wiki_name
    return await run_ingest(wiki_dir, source)


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

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
