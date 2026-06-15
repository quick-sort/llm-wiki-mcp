"""HTTP MCP server for llm-wiki-agent — provides query and ingest tools."""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import date
from pathlib import Path

import httpx
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import TransportSecuritySettings
from markitdown import MarkItDown

from .agent import run_ingest, run_query

# Mutable config — use _cfg["wikis_root"] everywhere
_cfg: dict = {
    "wikis_root": os.environ.get("LLM_WIKI_ROOT", str(Path.cwd() / "wikis")),
}

_MARKDOWN_EXTENSIONS = {".md", ".markdown", ".txt"}
_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".epub", ".odt", ".rtf", ".ipynb"}

# 权限分组
_QUERY_TOOLS = {"list_wikis", "query"}
_ADMIN_TOOLS = {"list_wikis", "query", "ingest", "list_sources", "delete_source", "update_source"}


def _get_markitdown() -> MarkItDown:
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if api_key and base_url:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        return MarkItDown(llm_client=client, llm_model=model)
    return MarkItDown()


def _save_raw(wiki_dir: Path, name: str, content: str) -> Path:
    raw_dir = wiki_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace("/", "_").replace(" ", "_")
    path = raw_dir / f"{safe_name}.md"
    path.write_text(content)
    return path


def _convert_file(file_path: str) -> tuple[str, str]:
    p = Path(file_path)
    name = p.stem
    md = _get_markitdown()
    result = md.convert(str(p))
    return name, result.text_content


async def _fetch_url(url: str) -> tuple[str, str]:
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    filename = url.rstrip("/").split("/")[-1] or "webpage"
    name = Path(filename).stem or "webpage"
    suffix = Path(filename).suffix.lower()
    content_type = resp.headers.get("content-type", "")

    if suffix in _DOCUMENT_EXTENSIONS or "application/pdf" in content_type:
        md = _get_markitdown()
        result = md.convert(resp.content)
        return name, result.text_content

    if "text/html" in content_type or suffix in ("", ".html", ".htm"):
        import trafilatura
        extracted = trafilatura.extract(resp.text, include_links=True, include_tables=True)
        if extracted:
            return name, extracted
        import re
        text = re.sub(r"<[^>]+>", "", resp.text)
        return name, text[:50000]

    return name, resp.text


async def _snapshot_wiki(wiki_dir: Path) -> dict:
    """记录当前 wiki 目录下所有文件的路径和修改时间"""
    snapshot = {}
    wiki_path = wiki_dir / "wiki"
    if wiki_path.exists():
        for f in wiki_path.rglob("*.md"):
            snapshot[str(f.relative_to(wiki_dir))] = f.stat().st_mtime
    return snapshot


def _diff_snapshot(before: dict, after: dict) -> dict:
    """对比快照，返回新增和修改的文件"""
    created = []
    updated = []
    for path, mtime in after.items():
        if path not in before:
            created.append(path)
        elif mtime != before[path]:
            updated.append(path)
    return {"created": created, "updated": updated}


def _update_pages_meta(wiki_dir: Path, source_name: str, affected: dict):
    """更新 .meta/pages.json，记录原始文件与 wiki 页面的关系"""
    meta_dir = wiki_dir / ".meta"
    meta_dir.mkdir(exist_ok=True)
    meta_file = meta_dir / "pages.json"

    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    all_pages = affected["created"] + affected["updated"]
    for page in all_pages:
        if page not in data:
            data[page] = {"sources": [], "last_updated": ""}
        if source_name not in data[page]["sources"]:
            data[page]["sources"].append(source_name)
        data[page]["last_updated"] = str(date.today())

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


mcp = FastMCP(
    "llm-wiki-mcp",
    instructions=(
        "A wiki knowledge base MCP server. Use 'ingest' to add documents (files, URLs, or raw text) "
        "to a named wiki, and 'query' to ask questions about the wiki's knowledge. "
        "Powered by llm-wiki-agent skills via Claude Agent SDK."
    ),
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
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


def _has_glob_pattern(path: str) -> bool:
    return any(c in path for c in ("*", "?", "["))


async def _ingest_file(wiki_dir: Path, file_path: str) -> str:
    """Ingest a single local file into the wiki."""
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"

    if p.suffix.lower() in _MARKDOWN_EXTENSIONS:
        return await run_ingest(wiki_dir, file_path)

    if p.suffix.lower() in _DOCUMENT_EXTENSIONS:
        name, md_content = _convert_file(file_path)
        local_path = _save_raw(wiki_dir, name, md_content)
        return await run_ingest(wiki_dir, str(local_path))

    return await run_ingest(wiki_dir, file_path)


@mcp.tool()
async def ingest(wiki_name: str, source: str = "", content: str = "") -> str:
    """Ingest content into a wiki knowledge base.

    Supports three input modes:
    - Local file path via `source` (glob patterns like * are expanded to match multiple files)
    - URL via `source` (starts with http:// or https://)
    - Raw text via `content`

    After saving, the llm-wiki-agent /wiki-ingest skill processes it — extracting
    entities, concepts, and cross-references into structured wiki pages.

    Args:
        wiki_name: Name of the wiki to ingest into (created if it doesn't exist).
        source: A local file path (supports glob) or URL to ingest.
        content: Raw text content to ingest. Used when source is empty.
    """
    if not source and not content:
        return "Error: provide either 'source' (file path or URL) or 'content' (raw text)."

    wiki_dir = Path(_cfg["wikis_root"]) / wiki_name
    before = await _snapshot_wiki(wiki_dir)
    result = ""
    source_name = ""

    if source.startswith(("http://", "https://")):
        name, text = await _fetch_url(source)
        local_path = _save_raw(wiki_dir, name, text)
        source_name = f"raw/{local_path.name}"
        result = await run_ingest(wiki_dir, str(local_path))

    elif content:
        local_path = _save_raw(wiki_dir, f"{wiki_name}_raw", content)
        source_name = f"raw/{local_path.name}"
        result = await run_ingest(wiki_dir, str(local_path))

    elif _has_glob_pattern(source):
        files = sorted(glob.glob(source))
        if not files:
            return f"Error: no files matched pattern: {source}"
        results = []
        for f in files:
            if Path(f).is_dir():
                continue
            file_before = await _snapshot_wiki(wiki_dir)
            file_result = await _ingest_file(wiki_dir, f)
            file_after = await _snapshot_wiki(wiki_dir)
            affected = _diff_snapshot(file_before, file_after)
            _update_pages_meta(wiki_dir, f"raw/{Path(f).name}", affected)
            results.append(f"[{f}] {file_result}")
        return f"Ingested {len(results)} file(s):\n" + "\n".join(results)

    else:
        source_name = f"raw/{Path(source).name}"
        result = await _ingest_file(wiki_dir, source)

    after = await _snapshot_wiki(wiki_dir)
    affected = _diff_snapshot(before, after)
    if affected["created"] or affected["updated"]:
        _update_pages_meta(wiki_dir, source_name, affected)

    return result


@mcp.tool()
async def list_sources(wiki_name: str) -> dict:
    """列出指定 wiki 中已经 ingest 过的所有原始文件。

    Args:
        wiki_name: wiki 名称。
    """
    wiki_dir = Path(_cfg["wikis_root"]) / wiki_name
    meta_file = wiki_dir / ".meta" / "pages.json"

    if not wiki_dir.exists():
        return {"error": f"Wiki '{wiki_name}' 不存在"}

    if not meta_file.exists():
        return {"sources": [], "count": 0}

    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_sources = set()
    for page_info in data.values():
        for source in page_info.get("sources", []):
            all_sources.add(source)

    sources = sorted(all_sources)
    return {"sources": sources, "count": len(sources)}


@mcp.tool()
async def delete_source(wiki_name: str, source_name: str) -> dict:
    """删除一个已 ingest 的原始文件，并自动处理相关 wiki 页面。

    - 如果某个 wiki 页面只有这一个来源 → 直接删除该页面
    - 如果某个 wiki 页面有多个来源 → 保留页面，从 sources 里移除该文件

    Args:
        wiki_name: wiki 名称。
        source_name: 原始文件名，格式如 "raw/锐鹰传感产品总览.md"
    """
    wiki_dir = Path(_cfg["wikis_root"]) / wiki_name
    meta_file = wiki_dir / ".meta" / "pages.json"

    if not wiki_dir.exists():
        return {"error": f"Wiki '{wiki_name}' 不存在"}

    if not meta_file.exists():
        return {"error": "没有找到 pages.json，请先 ingest 文件"}

    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    deleted_pages = []
    updated_pages = []

    for page, info in list(data.items()):
        if source_name not in info.get("sources", []):
            continue
        if len(info["sources"]) == 1:
            page_path = wiki_dir / page
            if page_path.exists():
                page_path.unlink()
                deleted_pages.append(page)
            del data[page]
        else:
            info["sources"].remove(source_name)
            info["last_updated"] = str(date.today())
            updated_pages.append(page)

    raw_path = wiki_dir / source_name
    if raw_path.exists():
        raw_path.unlink()

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "deleted_pages": deleted_pages,
        "updated_pages": updated_pages,
        "deleted_source": source_name,
        "message": f"已删除来源 {source_name}，删除了 {len(deleted_pages)} 个页面，更新了 {len(updated_pages)} 个页面"
    }


@mcp.tool()
async def update_source(wiki_name: str, source_name: str, new_source: str = "", new_content: str = "") -> dict:
    """更新一个已 ingest 的原始文件，并局部重建相关 wiki 页面。

    Args:
        wiki_name: wiki 名称。
        source_name: 原始文件名，格式如 "raw/锐鹰传感产品总览.md"
        new_source: 新的文件路径（可选，与 new_content 二选一）
        new_content: 新的文件内容（可选，与 new_source 二选一）
    """
    wiki_dir = Path(_cfg["wikis_root"]) / wiki_name
    meta_file = wiki_dir / ".meta" / "pages.json"

    if not wiki_dir.exists():
        return {"error": f"Wiki '{wiki_name}' 不存在"}

    if not meta_file.exists():
        return {"error": "没有找到 pages.json，请先 ingest 文件"}

    if not new_source and not new_content:
        return {"error": "请提供 new_source 或 new_content"}

    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages_to_delete = []
    for page, info in list(data.items()):
        if source_name in info.get("sources", []) and len(info["sources"]) == 1:
            page_path = wiki_dir / page
            if page_path.exists():
                page_path.unlink()
            pages_to_delete.append(page)
            del data[page]

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    raw_path = wiki_dir / source_name
    if new_content:
        raw_path.write_text(new_content, encoding="utf-8")
    elif new_source:
        import shutil
        shutil.copy(new_source, raw_path)

    before = await _snapshot_wiki(wiki_dir)
    result = await run_ingest(wiki_dir, str(raw_path))
    after = await _snapshot_wiki(wiki_dir)
    affected = _diff_snapshot(before, after)
    if affected["created"] or affected["updated"]:
        _update_pages_meta(wiki_dir, source_name, affected)

    return {
        "deleted_pages": pages_to_delete,
        "rebuilt_pages": affected["created"] + affected["updated"],
        "source": source_name,
        "message": f"已更新 {source_name}，删除了 {len(pages_to_delete)} 个旧页面，重建了 {len(affected['created']) + len(affected['updated'])} 个页面"
    }


# ══════════════════════════════════════════════════════════════════════════════
# 权限中间件（纯 ASGI 实现）— 仅拦截 tools/call，不缓冲/修改响应体
# ══════════════════════════════════════════════════════════════════════════════

class MCPAuthMiddleware:
    """根据 Authorization header 中的 key，限制可调用的工具。

    - MCP_ADMIN_KEY 命中 → 可调用全部工具
    - MCP_QUERY_KEY 命中 → 仅可调用 query / list_wikis
    - 未配置任何 key → 不限制（本地开发）

    仅拦截 tools/call 请求；tools/list 不过滤（query key 仍能看到全部
    工具名称，但调用管理员工具会被拒绝）。纯 ASGI 实现，避免
    BaseHTTPMiddleware 与 streamable-http 的 SSE 响应不兼容的问题。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/mcp" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return

        admin_key = os.environ.get("MCP_ADMIN_KEY", "")
        query_key = os.environ.get("MCP_QUERY_KEY", "")

        if not admin_key and not query_key:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode("latin-1")
        token = auth.removeprefix("Bearer ").strip()

        if admin_key and token == admin_key:
            allowed = _ADMIN_TOOLS
        elif query_key and token == query_key:
            allowed = _QUERY_TOOLS
        else:
            allowed = set()

        # 缓冲请求体（一次性读完，再重放给下游）
        body = b""
        more_body = True
        while more_body:
            message = await receive()
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        payload = None
        if body:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = None

        if payload and payload.get("method") == "tools/call":
            tool_name = payload.get("params", {}).get("name")
            if tool_name not in allowed:
                error_body = json.dumps({
                    "jsonrpc": "2.0",
                    "id": payload.get("id"),
                    "error": {
                        "code": -32601,
                        "message": f"Tool '{tool_name}' is not available for this credential",
                    },
                }).encode("utf-8")

                await send({
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [(b"content-type", b"application/json")],
                })
                await send({
                    "type": "http.response.body",
                    "body": error_body,
                })
                return

        # 重放请求体给下游（首次返回缓冲的 body，之后转发原始 receive）
        sent = False

        async def receive_replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, receive_replay, send)


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

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return

    # streamable-http：手动构建 app 以挂载权限中间件
    app = mcp.streamable_http_app()
    app = MCPAuthMiddleware(app)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
