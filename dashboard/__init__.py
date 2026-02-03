"""
Dashboard Module
================

Real-time monitoring and visualization components for the AI Trading Firm.

Components:
- DashboardServer: FastAPI server with WebSocket support
- DashboardState: State management for trading system data
- ConnectionManager: WebSocket connection management
"""

from __future__ import annotations

from dashboard.server import (
    DashboardServer,
    DashboardState,
    ConnectionManager,
    create_dashboard_server,
    AgentInfo,
    AgentStatus,
    PositionInfo,
    SignalInfo,
    DecisionInfo,
    RiskLimit,
    Metrics,
)

__version__ = "1.0.0"

__all__ = [
    "DashboardServer",
    "DashboardState",
    "ConnectionManager",
    "create_dashboard_server",
    "AgentInfo",
    "AgentStatus",
    "PositionInfo",
    "SignalInfo",
    "DecisionInfo",
    "RiskLimit",
    "Metrics",
]
