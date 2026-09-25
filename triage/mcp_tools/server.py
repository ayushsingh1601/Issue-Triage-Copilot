"""MCP server exposing the five read-only issue-triage tools."""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from triage.mcp_tools.tools import TriageTools


def create_server(tools: TriageTools) -> FastMCP:
    mcp = FastMCP("issue-triage-copilot")

    @mcp.tool()
    def classify_issue(issue: str) -> dict[str, Any]:
        return tools.classify_issue(issue)

    @mcp.tool()
    def search_past_issues(
        query: str,
        repo: str | None = None,
        issue_type: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        return tools.search_past_issues(query, repo=repo, issue_type=issue_type, limit=limit)

    @mcp.tool()
    def get_issue_details(issue_id: str) -> dict[str, Any] | None:
        return tools.get_issue_details(issue_id)

    @mcp.tool()
    def get_runbook_steps(issue_type: str, query: str = "") -> list[dict[str, Any]]:
        return tools.get_runbook_steps(issue_type, query)

    @mcp.tool()
    def get_resolution_patterns(issue_type: str, repo: str | None = None) -> dict[str, Any]:
        return tools.get_resolution_patterns(issue_type, repo)

    return mcp
