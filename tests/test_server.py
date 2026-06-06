"""Tests for llm_wiki_mcp.server — content fetching, conversion, and saving."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from markitdown import DocumentConverterResult


# ---------------------------------------------------------------------------
# _save_raw
# ---------------------------------------------------------------------------
class TestSaveRaw:
    def test_creates_raw_dir_and_saves_content(self, tmp_path: Path):
        from llm_wiki_mcp.server import _save_raw

        result = _save_raw(tmp_path, "doc-name", "# Hello\n\nSome content.")

        assert result == tmp_path / "raw" / "doc-name.md"
        assert result.read_text() == "# Hello\n\nSome content."

    def test_sanitizes_name_with_slashes_and_spaces(self, tmp_path: Path):
        from llm_wiki_mcp.server import _save_raw

        result = _save_raw(tmp_path, "some/evil name", "text")

        assert result == tmp_path / "raw" / "some_evil_name.md"
        assert result.read_text() == "text"

    def test_existing_raw_dir_is_reused(self, tmp_path: Path):
        from llm_wiki_mcp.server import _save_raw

        raw = tmp_path / "raw"
        raw.mkdir()

        _save_raw(tmp_path, "doc", "content")
        assert raw.is_dir()


# ---------------------------------------------------------------------------
# _convert_file
# ---------------------------------------------------------------------------
class TestConvertFile:
    def test_converts_docx_via_markitdown(self):
        from llm_wiki_mcp.server import _convert_file

        with patch(
            "llm_wiki_mcp.server._get_markitdown"
        ) as mock_get_md:
            md_instance = MagicMock()
            md_instance.convert.return_value = DocumentConverterResult(
                markdown="# Converted document content",
            )
            mock_get_md.return_value = md_instance

            name, content = _convert_file("/fake/path/document.docx")

            assert name == "document"
            assert content == "# Converted document content"
            md_instance.convert.assert_called_once_with("/fake/path/document.docx")

    def test_converts_pdf_via_markitdown(self):
        from llm_wiki_mcp.server import _convert_file

        with patch(
            "llm_wiki_mcp.server._get_markitdown"
        ) as mock_get_md:
            md_instance = MagicMock()
            md_instance.convert.return_value = DocumentConverterResult(
                markdown="# PDF content as markdown",
            )
            mock_get_md.return_value = md_instance

            name, content = _convert_file("/fake/path/report.pdf")

            assert name == "report"
            assert content == "# PDF content as markdown"

    def test_converts_xlsx_via_markitdown(self):
        from llm_wiki_mcp.server import _convert_file

        with patch(
            "llm_wiki_mcp.server._get_markitdown"
        ) as mock_get_md:
            md_instance = MagicMock()
            md_instance.convert.return_value = DocumentConverterResult(
                markdown="| A | B |\n|---|---|\n| 1 | 2 |",
            )
            mock_get_md.return_value = md_instance

            name, content = _convert_file("/fake/path/data.xlsx")

            assert name == "data"
            assert content == "| A | B |\n|---|---|\n| 1 | 2 |"

    def test_handles_file_in_subdirectory(self):
        from llm_wiki_mcp.server import _convert_file

        with patch(
            "llm_wiki_mcp.server._get_markitdown"
        ) as mock_get_md:
            md_instance = MagicMock()
            md_instance.convert.return_value = DocumentConverterResult(
                markdown="ok"
            )
            mock_get_md.return_value = md_instance

            name, _ = _convert_file("/a/b/c/file.docx")

            assert name == "file"


# ---------------------------------------------------------------------------
# _fetch_url — HTML (trafilatura)
# ---------------------------------------------------------------------------
class TestFetchUrlHtml:
    @pytest.mark.anyio
    async def test_html_page_extracted_by_trafilatura(self):
        from llm_wiki_mcp.server import _fetch_url

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.text = "<html><body><article>Clean text</article></body></html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("llm_wiki_mcp.server.httpx.AsyncClient", return_value=mock_client),
            patch(
                "trafilatura.extract", return_value="Clean text as markdown"
            ) as mock_extract,
        ):
            name, content = await _fetch_url("https://example.com/article")

            assert name == "article"
            assert content == "Clean text as markdown"
            mock_extract.assert_called_once_with(
                mock_response.text, include_links=True, include_tables=True
            )

    @pytest.mark.anyio
    async def test_html_fallback_when_trafilatura_returns_none(self):
        from llm_wiki_mcp.server import _fetch_url

        html = "<html><body><p>Some raw paragraph.</p></body></html>"
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("llm_wiki_mcp.server.httpx.AsyncClient", return_value=mock_client),
            patch("trafilatura.extract", return_value=None),
        ):
            name, content = await _fetch_url("https://example.com/broken")

            # Fallback strips HTML tags
            assert "<p>" not in content
            assert "Some raw paragraph." in content

    @pytest.mark.anyio
    async def test_html_url_with_no_suffix(self):
        from llm_wiki_mcp.server import _fetch_url

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "<html>content</html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("llm_wiki_mcp.server.httpx.AsyncClient", return_value=mock_client),
            patch("trafilatura.extract", return_value="extracted") as mock_extract,
        ):
            name, content = await _fetch_url("https://example.com/foo")

            mock_extract.assert_called_once()
            assert name == "foo"
            assert content == "extracted"

    @pytest.mark.anyio
    async def test_html_url_suffix_only_taken_from_last_segment(self):
        from llm_wiki_mcp.server import _fetch_url

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "<html>c</html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("llm_wiki_mcp.server.httpx.AsyncClient", return_value=mock_client),
            patch("trafilatura.extract", return_value="ok"),
        ):
            name, _ = await _fetch_url("https://example.com/path/to/page")
            assert name == "page"

    @pytest.mark.anyio
    async def test_url_with_trailing_slash_handles_name_correctly(self):
        from llm_wiki_mcp.server import _fetch_url

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "<html>c</html>"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("llm_wiki_mcp.server.httpx.AsyncClient", return_value=mock_client),
            patch("trafilatura.extract", return_value="ok"),
        ):
            name, _ = await _fetch_url("https://example.com/")
            assert name == "example"

    @pytest.mark.anyio
    async def test_truncates_fallback_text_to_50k_chars(self):
        from llm_wiki_mcp.server import _fetch_url

        big = "x" * 60000
        html = f"<html><body>{big}</body></html>"
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = html
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("llm_wiki_mcp.server.httpx.AsyncClient", return_value=mock_client),
            patch("trafilatura.extract", return_value=None),
        ):
            _, content = await _fetch_url("https://example.com/big")
            assert len(content) <= 50000


# ---------------------------------------------------------------------------
# _fetch_url — Documents (markitdown)
# ---------------------------------------------------------------------------
class TestFetchUrlDocuments:
    @pytest.mark.anyio
    async def test_pdf_suffix_uses_markitdown(self):
        from llm_wiki_mcp.server import _fetch_url

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/octet-stream"}
        mock_response.content = b"%PDF-1.4 fake pdf bytes"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("llm_wiki_mcp.server.httpx.AsyncClient", return_value=mock_client),
            patch(
                "llm_wiki_mcp.server._get_markitdown"
            ) as mock_get_md,
        ):
            md_instance = MagicMock()
            md_instance.convert.return_value = DocumentConverterResult(
                markdown="# PDF via markitdown"
            )
            mock_get_md.return_value = md_instance

            name, content = await _fetch_url("https://example.com/report.pdf")

            assert name == "report"
            assert content == "# PDF via markitdown"
            md_instance.convert.assert_called_once_with(b"%PDF-1.4 fake pdf bytes")

    @pytest.mark.anyio
    async def test_docx_suffix_uses_markitdown(self):
        from llm_wiki_mcp.server import _fetch_url

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/octet-stream"}
        mock_response.content = b"PK fake docx bytes"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("llm_wiki_mcp.server.httpx.AsyncClient", return_value=mock_client),
            patch(
                "llm_wiki_mcp.server._get_markitdown"
            ) as mock_get_md,
        ):
            md_instance = MagicMock()
            md_instance.convert.return_value = DocumentConverterResult(
                markdown="# DOCX converted"
            )
            mock_get_md.return_value = md_instance

            name, content = await _fetch_url("https://example.com/doc.docx")

            assert name == "doc"
            assert content == "# DOCX converted"

    @pytest.mark.anyio
    async def test_xlsx_suffix_uses_markitdown(self):
        from llm_wiki_mcp.server import _fetch_url

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/octet-stream"}
        mock_response.content = b"PK fake xlsx bytes"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("llm_wiki_mcp.server.httpx.AsyncClient", return_value=mock_client),
            patch("llm_wiki_mcp.server._get_markitdown") as mock_get_md,
        ):
            md_instance = MagicMock()
            md_instance.convert.return_value = DocumentConverterResult(
                markdown="| Col1 | Col2 |"
            )
            mock_get_md.return_value = md_instance

            name, content = await _fetch_url("https://example.com/data.xlsx")

            assert name == "data"
            assert content == "| Col1 | Col2 |"

    @pytest.mark.anyio
    async def test_pptx_suffix_uses_markitdown(self):
        from llm_wiki_mcp.server import _fetch_url

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/octet-stream"}
        mock_response.content = b"PK fake pptx bytes"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("llm_wiki_mcp.server.httpx.AsyncClient", return_value=mock_client),
            patch("llm_wiki_mcp.server._get_markitdown") as mock_get_md,
        ):
            md_instance = MagicMock()
            md_instance.convert.return_value = DocumentConverterResult(
                markdown="# Slide deck"
            )
            mock_get_md.return_value = md_instance

            name, content = await _fetch_url("https://example.com/slides.pptx")

            assert name == "slides"

    @pytest.mark.anyio
    async def test_pdf_content_type_without_suffix_uses_markitdown(self):
        from llm_wiki_mcp.server import _fetch_url

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.content = b"%PDF-1.4"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("llm_wiki_mcp.server.httpx.AsyncClient", return_value=mock_client),
            patch("llm_wiki_mcp.server._get_markitdown") as mock_get_md,
        ):
            md_instance = MagicMock()
            md_instance.convert.return_value = DocumentConverterResult(
                markdown="# PDF"
            )
            mock_get_md.return_value = md_instance

            name, content = await _fetch_url(
                "https://example.com/download?file=123"
            )

            # Path stem keeps query string, so name reflects the full last segment
            assert name == "download?file=123"
            assert content == "# PDF"
            md_instance.convert.assert_called_once_with(b"%PDF-1.4")

    @pytest.mark.anyio
    async def test_all_document_extensions_routed_to_markitdown(self):
        from llm_wiki_mcp.server import _fetch_url, _DOCUMENT_EXTENSIONS

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/octet-stream"}
        mock_response.content = b"binary"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("llm_wiki_mcp.server.httpx.AsyncClient", return_value=mock_client),
            patch("llm_wiki_mcp.server._get_markitdown") as mock_get_md,
        ):
            md_instance = MagicMock()
            md_instance.convert.return_value = DocumentConverterResult(
                markdown="ok"
            )
            mock_get_md.return_value = md_instance

            for ext in sorted(_DOCUMENT_EXTENSIONS):
                name, _ = await _fetch_url(f"https://example.com/file{ext}")
                assert name == "file", f"Failed for extension {ext}"


# ---------------------------------------------------------------------------
# _fetch_url — Plain text / fallback
# ---------------------------------------------------------------------------
class TestFetchUrlPlainText:
    @pytest.mark.anyio
    async def test_plain_text_returned_as_is(self):
        from llm_wiki_mcp.server import _fetch_url

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/plain; charset=utf-8"}
        mock_response.text = "Just some plain text."
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("llm_wiki_mcp.server.httpx.AsyncClient", return_value=mock_client):
            name, content = await _fetch_url("https://example.com/data.txt")

            assert name == "data"
            assert content == "Just some plain text."

    @pytest.mark.anyio
    async def test_unknown_binary_type_returned_as_text(self):
        from llm_wiki_mcp.server import _fetch_url

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.text = '{"key": "value"}'
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("llm_wiki_mcp.server.httpx.AsyncClient", return_value=mock_client):
            name, content = await _fetch_url("https://example.com/api.json")

            assert name == "api"
            assert content == '{"key": "value"}'


# ---------------------------------------------------------------------------
# _get_markitdown
# ---------------------------------------------------------------------------
class TestGetMarkItDown:
    def test_plain_markitdown_when_no_env_vars(self):
        from llm_wiki_mcp.server import _get_markitdown

        with patch.dict(os.environ, {}, clear=True):
            md = _get_markitdown()
            assert md._llm_client is None

    def test_llm_markitdown_when_env_vars_set(self):
        from llm_wiki_mcp.server import _get_markitdown

        env = {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_BASE_URL": "https://api.example.com",
            "OPENAI_MODEL": "gpt-4o-mini",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch("openai.OpenAI"),
        ):
            md = _get_markitdown()
            assert md._llm_client is not None
            assert md._llm_model == "gpt-4o-mini"

    def test_default_model_when_not_specified(self):
        from llm_wiki_mcp.server import _get_markitdown

        env = {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_BASE_URL": "https://api.example.com",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch("openai.OpenAI"),
        ):
            md = _get_markitdown()
            assert md._llm_model == "gpt-4o"

    def test_no_llm_when_only_api_key_set(self):
        from llm_wiki_mcp.server import _get_markitdown

        env = {"OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            md = _get_markitdown()
            assert md._llm_client is None

    def test_no_llm_when_only_base_url_set(self):
        from llm_wiki_mcp.server import _get_markitdown

        env = {"OPENAI_BASE_URL": "https://api.example.com"}
        with patch.dict(os.environ, env, clear=True):
            md = _get_markitdown()
            assert md._llm_client is None


# ---------------------------------------------------------------------------
# _has_glob_pattern
# ---------------------------------------------------------------------------
class TestHasGlobPattern:
    def test_star_is_glob(self):
        from llm_wiki_mcp.server import _has_glob_pattern

        assert _has_glob_pattern("*.pdf") is True
        assert _has_glob_pattern("/path/to/*.md") is True

    def test_question_mark_is_glob(self):
        from llm_wiki_mcp.server import _has_glob_pattern

        assert _has_glob_pattern("file?.txt") is True

    def test_bracket_is_glob(self):
        from llm_wiki_mcp.server import _has_glob_pattern

        assert _has_glob_pattern("file[0-9].txt") is True

    def test_plain_path_is_not_glob(self):
        from llm_wiki_mcp.server import _has_glob_pattern

        assert _has_glob_pattern("/path/to/file.pdf") is False
        assert _has_glob_pattern("file.txt") is False


# ---------------------------------------------------------------------------
# ingest — glob expansion
# ---------------------------------------------------------------------------
class TestIngestGlob:
    @pytest.mark.anyio
    async def test_glob_matches_multiple_files(self, tmp_path: Path):
        from llm_wiki_mcp.server import ingest, _cfg

        _cfg["wikis_root"] = str(tmp_path / "wikis")

        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "b.txt").write_text("bbb")
        (tmp_path / "c.pdf").write_text("not a real pdf")

        with patch("llm_wiki_mcp.server.run_ingest", new_callable=AsyncMock, return_value="ok"):
            result = await ingest(
                wiki_name="test",
                source=str(tmp_path / "*.txt"),
            )

        assert "Ingested 2 file(s)" in result
        assert "a.txt" in result
        assert "b.txt" in result
        assert "c.pdf" not in result

    @pytest.mark.anyio
    async def test_glob_no_match_returns_error(self, tmp_path: Path):
        from llm_wiki_mcp.server import ingest, _cfg

        _cfg["wikis_root"] = str(tmp_path / "wikis")

        result = await ingest(
            wiki_name="test",
            source=str(tmp_path / "*.nonexistent"),
        )

        assert "Error" in result
        assert "no files matched" in result

    @pytest.mark.anyio
    async def test_glob_skips_directories(self, tmp_path: Path):
        from llm_wiki_mcp.server import ingest, _cfg

        _cfg["wikis_root"] = str(tmp_path / "wikis")

        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "subdir").mkdir()

        with patch("llm_wiki_mcp.server.run_ingest", new_callable=AsyncMock, return_value="ok"):
            result = await ingest(
                wiki_name="test",
                source=str(tmp_path / "*"),
            )

        assert "Ingested 1 file(s)" in result

    @pytest.mark.anyio
    async def test_plain_path_still_works(self, tmp_path: Path):
        from llm_wiki_mcp.server import ingest, _cfg

        _cfg["wikis_root"] = str(tmp_path / "wikis")

        f = tmp_path / "single.txt"
        f.write_text("hello")

        with patch("llm_wiki_mcp.server.run_ingest", new_callable=AsyncMock, return_value="ok"):
            result = await ingest(
                wiki_name="test",
                source=str(f),
            )

        assert "ok" == result
