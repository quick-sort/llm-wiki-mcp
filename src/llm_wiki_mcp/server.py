"""llm-wiki: OpenAI 兼容的 wiki Agent 服务 + 知识库管理 API。

接口：
- GET  /v1/models                     返回可用模型列表
- POST /v1/chat/completions           wiki 问答（流式/非流式）
- POST /api/ingest                    注入文件到 wiki
- GET  /api/wikis                     列出所有 wiki
- GET  /api/wikis/{name}/sources      列出已 ingest 的文件
- GET  /api/wikis/{name}/raw_files    列出 raw 文件及 ingest 状态
- DELETE /api/wikis/{name}/sources/{source_name}  删除来源
- PUT /api/wikis/{name}/sources/{source_name}     更新来源
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import time
import uuid
from datetime import date
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from markitdown import MarkItDown
from pydantic import BaseModel

from .agent import run_ingest, run_query

app = FastAPI(title="llm-wiki-agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 配置 ──────────────────────────────────────────────────────────────────────

def get_cfg() -> dict:
    return {
        "wikis_root": os.environ.get("LLM_WIKI_ROOT", str(Path.cwd() / "wikis")),
        "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "base_url": os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        "wiki_name": os.environ.get("LLM_WIKI_NAME", "reagle"),
    }

_MARKDOWN_EXTENSIONS = {".md", ".markdown", ".txt"}
_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".epub", ".odt", ".rtf", ".ipynb"}


# ── 工具函数（复用自 llm-wiki-mcp）──────────────────────────────────────────

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
    snapshot = {}
    wiki_path = wiki_dir / "wiki"
    if wiki_path.exists():
        for f in wiki_path.rglob("*.md"):
            snapshot[str(f.relative_to(wiki_dir))] = f.stat().st_mtime
    return snapshot


def _diff_snapshot(before: dict, after: dict) -> dict:
    created, updated = [], []
    for path, mtime in after.items():
        if path not in before:
            created.append(path)
        elif mtime != before[path]:
            updated.append(path)
    return {"created": created, "updated": updated}


def _update_pages_meta(wiki_dir: Path, source_name: str, affected: dict):
    meta_dir = wiki_dir / ".meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_file = meta_dir / "pages.json"

    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    for page in affected["created"] + affected["updated"]:
        if page not in data:
            data[page] = {"sources": [], "last_updated": ""}
        if source_name not in data[page]["sources"]:
            data[page]["sources"].append(source_name)
        data[page]["last_updated"] = str(date.today())

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _has_glob_pattern(path: str) -> bool:
    return any(c in path for c in ("*", "?", "["))


async def _ingest_file(wiki_dir: Path, file_path: str) -> str:
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"

    if p.suffix.lower() in _MARKDOWN_EXTENSIONS:
        raw_dir = wiki_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        local_path = raw_dir / p.name
        if local_path.resolve() != p.resolve():
            shutil.copy(file_path, local_path)
        return await run_ingest(wiki_dir, str(local_path))

    if p.suffix.lower() in _DOCUMENT_EXTENSIONS:
        name, md_content = _convert_file(file_path)
        local_path = _save_raw(wiki_dir, name, md_content)
        return await run_ingest(wiki_dir, str(local_path))

    return await run_ingest(wiki_dir, file_path)


# ── Pydantic 模型 ─────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "llm-wiki-agent"
    messages: list[Message]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 4096


class IngestRequest(BaseModel):
    wiki_name: str
    source: str = ""
    content: str = ""


class UpdateSourceRequest(BaseModel):
    new_source: str = ""
    new_content: str = ""


# ── OpenAI 兼容接口 ───────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    cfg = get_cfg()
    return {
        "object": "list",
        "data": [
            {
                "id": "llm-wiki-agent",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llm-wiki-agent",
                "description": f"Wiki 知识库问答助手（基于 {cfg['wiki_name']} wiki）",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    cfg = get_cfg()

    # 取最后一条 user 消息作为问题
    question = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            question = msg.content
            break

    if not question:
        raise HTTPException(status_code=400, detail="No user message found")

    wiki_dir = Path(cfg["wikis_root"]) / cfg["wiki_name"]
    if not wiki_dir.exists():
        raise HTTPException(status_code=500, detail=f"Wiki '{cfg['wiki_name']}' 不存在")

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            _stream_query(wiki_dir, question, completion_id, created),
            media_type="text/event-stream",
        )
    else:
        result = await run_query(wiki_dir, question)
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": "llm-wiki-agent",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })


async def _stream_query(
    wiki_dir: Path, question: str, completion_id: str, created: int
) -> AsyncIterator[str]:
    """流式返回 run_query 结果（run_query 本身不流式，这里分块发送）"""
    result = await run_query(wiki_dir, question)

    # 按字符分块流式发送
    chunk_size = 10
    for i in range(0, len(result), chunk_size):
        chunk_text = result[i:i + chunk_size]
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "llm-wiki-agent",
            "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # 结束
    end_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "llm-wiki-agent",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(end_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ── 知识库管理 API ────────────────────────────────────────────────────────────

@app.get("/api/wikis")
async def list_wikis_api():
    cfg = get_cfg()
    wikis_root = Path(cfg["wikis_root"])
    if not wikis_root.exists():
        return {"wikis": []}
    return {"wikis": sorted(d.name for d in wikis_root.iterdir() if d.is_dir() and not d.name.startswith("."))}


@app.post("/api/ingest")
async def ingest_api(req: IngestRequest):
    cfg = get_cfg()
    wiki_dir = Path(cfg["wikis_root"]) / req.wiki_name
    before = await _snapshot_wiki(wiki_dir)
    result = ""
    source_name = ""

    if not req.source and not req.content:
        raise HTTPException(status_code=400, detail="请提供 source 或 content")

    if req.source.startswith(("http://", "https://")):
        name, text = await _fetch_url(req.source)
        local_path = _save_raw(wiki_dir, name, text)
        source_name = f"raw/{local_path.name}"
        result = await run_ingest(wiki_dir, str(local_path))
    elif req.content:
        local_path = _save_raw(wiki_dir, f"{req.wiki_name}_raw", req.content)
        source_name = f"raw/{local_path.name}"
        result = await run_ingest(wiki_dir, str(local_path))
    elif _has_glob_pattern(req.source):
        files = sorted(glob.glob(req.source))
        if not files:
            raise HTTPException(status_code=400, detail=f"No files matched: {req.source}")
        results = []
        for f in files:
            if Path(f).is_dir():
                continue
            fb = await _snapshot_wiki(wiki_dir)
            fr = await _ingest_file(wiki_dir, f)
            fa = await _snapshot_wiki(wiki_dir)
            _update_pages_meta(wiki_dir, f"raw/{Path(f).name}", _diff_snapshot(fb, fa))
            results.append(f)
        return {"ingested": results, "count": len(results)}
    else:
        source_name = f"raw/{Path(req.source).name}"
        result = await _ingest_file(wiki_dir, req.source)

    after = await _snapshot_wiki(wiki_dir)
    affected = _diff_snapshot(before, after)
    if affected["created"] or affected["updated"]:
        _update_pages_meta(wiki_dir, source_name, affected)

    return {"result": result, "source": source_name}


@app.get("/api/wikis/{wiki_name}/sources")
async def list_sources_api(wiki_name: str):
    cfg = get_cfg()
    wiki_dir = Path(cfg["wikis_root"]) / wiki_name
    meta_file = wiki_dir / ".meta" / "pages.json"

    if not wiki_dir.exists():
        raise HTTPException(status_code=404, detail=f"Wiki '{wiki_name}' 不存在")
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


@app.get("/api/wikis/{wiki_name}/raw_files")
async def list_raw_files_api(wiki_name: str):
    cfg = get_cfg()
    wiki_dir = Path(cfg["wikis_root"]) / wiki_name
    raw_dir = wiki_dir / "raw"

    if not wiki_dir.exists():
        raise HTTPException(status_code=404, detail=f"Wiki '{wiki_name}' 不存在")
    if not raw_dir.exists():
        return {"all": [], "ingested": [], "not_ingested": [], "count": {"all": 0, "ingested": 0, "not_ingested": 0}}

    all_files = sorted(f"raw/{f.relative_to(raw_dir)}" for f in raw_dir.rglob("*") if f.is_file())

    ingested_set = set()
    meta_file = wiki_dir / ".meta" / "pages.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for page_info in data.values():
            for source in page_info.get("sources", []):
                ingested_set.add(source)

    ingested = [f for f in all_files if f in ingested_set]
    not_ingested = [f for f in all_files if f not in ingested_set]

    return {
        "all": all_files,
        "ingested": ingested,
        "not_ingested": not_ingested,
        "count": {"all": len(all_files), "ingested": len(ingested), "not_ingested": len(not_ingested)},
    }


@app.delete("/api/wikis/{wiki_name}/sources/{source_name:path}")
async def delete_source_api(wiki_name: str, source_name: str):
    cfg = get_cfg()
    wiki_dir = Path(cfg["wikis_root"]) / wiki_name
    meta_file = wiki_dir / ".meta" / "pages.json"

    if not wiki_dir.exists():
        raise HTTPException(status_code=404, detail=f"Wiki '{wiki_name}' 不存在")
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="没有找到 pages.json")

    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    deleted_pages, updated_pages = [], []
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

    return {"deleted_pages": deleted_pages, "updated_pages": updated_pages, "deleted_source": source_name}


@app.put("/api/wikis/{wiki_name}/sources/{source_name:path}")
async def update_source_api(wiki_name: str, source_name: str, req: UpdateSourceRequest):
    cfg = get_cfg()
    wiki_dir = Path(cfg["wikis_root"]) / wiki_name
    meta_file = wiki_dir / ".meta" / "pages.json"

    if not wiki_dir.exists():
        raise HTTPException(status_code=404, detail=f"Wiki '{wiki_name}' 不存在")
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="没有找到 pages.json")
    if not req.new_source and not req.new_content:
        raise HTTPException(status_code=400, detail="请提供 new_source 或 new_content")

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
    if req.new_content:
        raw_path.write_text(req.new_content, encoding="utf-8")
    elif req.new_source:
        shutil.copy(req.new_source, raw_path)

    before = await _snapshot_wiki(wiki_dir)
    result = await run_ingest(wiki_dir, str(raw_path))
    after = await _snapshot_wiki(wiki_dir)
    affected = _diff_snapshot(before, after)
    if affected["created"] or affected["updated"]:
        _update_pages_meta(wiki_dir, source_name, affected)

    return {"deleted_pages": pages_to_delete, "rebuilt_pages": affected["created"] + affected["updated"], "source": source_name}


# ── 启动 ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="llm-wiki-agent server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--wikis-root", default=None)
    parser.add_argument("--wiki-name", default=None)
    args = parser.parse_args()

    if args.wikis_root:
        os.environ["LLM_WIKI_ROOT"] = args.wikis_root
    if args.wiki_name:
        os.environ["LLM_WIKI_NAME"] = args.wiki_name

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
