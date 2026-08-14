"""Tests for the Tavily search tool, using a fake async client (zero network)."""

from __future__ import annotations

import asyncio

from leximeter.client.search_tavily import TavilySearchTool


class FakeTavilyClient:
    """Return a canned Tavily payload; records the request for assertions."""

    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    async def search(self, **kwargs):
        self.calls.append(kwargs)
        return self._payload


def run(coro):
    return asyncio.run(coro)


def make_tool(payload, **tool_kwargs):
    tool = TavilySearchTool(api_key="tvly-test", **tool_kwargs)
    tool._client = FakeTavilyClient(payload)  # inject fake, skip import/network
    return tool


def test_search_formats_answer_and_results():
    tool = make_tool(
        {
            "query": "quantum computing",
            "answer": "Quantum computing is advancing rapidly.",
            "results": [
                {"title": "Tavily", "content": "A search API for AI agents.", "url": "https://tavily.com"},
                {"title": "Example", "content": "An example page.", "url": "https://example.com"},
            ],
        }
    )
    text = run(tool.search("quantum computing"))
    lines = text.splitlines()
    # the Tavily `answer` summary leads the prompt-ready context
    assert lines[0] == "Quantum computing is advancing rapidly."
    assert "- Tavily — A search API for AI agents. (https://tavily.com)" in lines
    assert lines[-1].startswith("- Example")


def test_search_skips_empty_fields():
    tool = make_tool(
        {
            "results": [
                {"title": "Tavily", "content": "", "url": "https://tavily.com"},
            ]
        }
    )
    text = run(tool.search("query"))
    assert text == "- Tavily (https://tavily.com)"


def test_search_empty_results_returns_empty_string():
    tool = make_tool({"query": "nothing", "results": []})
    assert run(tool.search("nothing")) == ""


def test_search_failure_returns_empty_string():
    class Boom:
        async def search(self, **kwargs):
            raise RuntimeError("quota exceeded")

    tool = TavilySearchTool(api_key="tvly-test")
    tool._client = Boom()
    assert run(tool.search("x")) == ""


def test_search_no_api_key_returns_empty_string():
    tool = TavilySearchTool(api_key=None)  # no key -> no client -> empty context
    assert run(tool.search("x")) == ""


def test_search_context_size_maps_to_search_depth():
    # GABRIEL's search_context_size vocabulary maps onto Tavily search_depth.
    tool = make_tool({"results": []}, search_context_size="high")
    run(tool.search("q"))
    assert tool._client.calls[0]["search_depth"] == "advanced"


def test_search_context_size_accepts_legacy_aliases():
    # small -> low -> fast; large -> high -> advanced (GABRIEL compat)
    tool = make_tool({"results": []}, search_context_size="large")
    run(tool.search("q"))
    assert tool._client.calls[0]["search_depth"] == "advanced"


def test_search_context_size_rejects_unknown():
    import pytest

    with pytest.raises(ValueError):
        make_tool({"results": []}, search_context_size="ultra")


def test_web_search_filters_map_to_tavily_params():
    # allowed_domains -> include_domains; location hint -> country
    tool = make_tool(
        {"results": []},
        web_search_filters={
            "allowed_domains": ["arxiv.org", "nature.com"],
            "country": "US",
        },
    )
    run(tool.search("q"))
    call = tool._client.calls[0]
    assert call["include_domains"] == ["arxiv.org", "nature.com"]
    assert call["country"] == "US"


def test_web_search_filters_empty_domains_skipped():
    tool = make_tool({"results": []}, web_search_filters={"allowed_domains": []})
    run(tool.search("q"))
    call = tool._client.calls[0]
    assert call["include_domains"] is None
