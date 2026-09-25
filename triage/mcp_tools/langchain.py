"""LangChain wiring for the MCP tools with strict per-agent access groups."""
from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.client.session import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session
from triage.mcp_tools.server import create_server
from triage.mcp_tools.tools import TriageTools

TOOL_GROUPS = {
    "orchestrator": ["classify_issue", "get_resolution_patterns"],
    "historical": ["search_past_issues", "get_issue_details"],
    "process": ["get_runbook_steps"],
}


class AgentToolbox:
    def __init__(self, tools: TriageTools) -> None:
        self._server = create_server(tools)
        self._session_cm = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> AgentToolbox:
        self._session_cm = create_connected_server_and_client_session(self._server)
        self._session = await self._session_cm.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._session_cm.__aexit__(*exc)
        self._session = None

    async def group(self, name: str) -> list[BaseTool]:
        assert self._session is not None, "toolbox not entered"
        all_tools = await load_mcp_tools(self._session)
        by_name = {tool.name: tool for tool in all_tools}
        return [by_name[tool_name] for tool_name in TOOL_GROUPS[name]]
