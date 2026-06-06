# llm-wiki-mcp

HTTP MCP server providing wiki knowledge base tools (`query` and `ingest`), powered by [llm-wiki-agent](https://github.com/samuraigpt/llm-wiki-agent) skills via the **Claude Agent SDK**.

## Architecture

```
MCP Client (Claude Code, etc.)
    ↓ HTTP / stdio
llm-wiki-mcp (FastMCP server)
    ↓ Claude Agent SDK (query())
Claude Code agent session
    ↓ auto-loaded skills
llm-wiki-agent (.claude/commands/)
    ↓ /wiki-ingest, /wiki-query
Wiki markdown files (entities, concepts, sources, syntheses)
```

The MCP server is a thin layer. The actual knowledge extraction, entity/concept page creation, cross-referencing, and contradiction detection are all handled by the `llm-wiki-agent` skill running inside a Claude Agent SDK session.

## Quick Start

```bash
# Install dependencies
uv sync

# Clone llm-wiki-agent skills
git clone --depth 1 https://github.com/SamurAIGPT/llm-wiki-agent.git skills/llm-wiki-agent

# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# Start the server (HTTP mode, default port 8080)
uv run llm-wiki-mcp

# Or use stdio mode
uv run llm-wiki-mcp --transport stdio
```

## Tools

### `ingest`

Ingest a document (file path or URL) into a named wiki. The `llm-wiki-agent` skill handles reading, converting, extracting knowledge, and building cross-referenced pages.

```
ingest(wiki_name="my-research", source="raw/papers/attention-is-all-you-need.md")
ingest(wiki_name="my-research", source="report.pdf")
```

### `query`

Query a wiki with a natural language question. The skill searches all wiki pages and synthesizes an answer with `[[wikilinks]]`.

```
query(wiki_name="my-research", question="What are the main approaches to attention?")
```

## Docker

```bash
docker build -t llm-wiki-mcp .
docker run -e ANTHROPIC_API_KEY=sk-ant-... -p 8080:8080 -v ./wikis:/home/agent/wikis llm-wiki-mcp
```

Optional: configure custom base URL or models:

```bash
docker run \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e ANTHROPIC_BASE_URL=https://custom.anthropic.com \
  -e ANTHROPIC_SMALL_FAST_MODEL=claude-haiku-4-20250514 \
  -e ANTHROPIC_DEFAULT_SONNET_MODEL=claude-sonnet-4-20250514 \
  -e ANTHROPIC_DEFAULT_OPUS_MODEL=claude-opus-4-20250514 \
  -e ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-haiku-4-20250514 \
  -p 8080:8080 \
  -v ./wikis:/home/agent/wikis \
  llm-wiki-mcp
```
