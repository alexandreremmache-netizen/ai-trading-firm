"""
AI Trading Firm - Agents Module
================================

Multi-agent architecture with strict separation of responsibilities.
Per CLAUDE.md: Each agent has a single, well-defined responsibility.
"""

from agents.macro_agent import MacroAgent
from agents.stat_arb_agent import StatArbAgent
from agents.momentum_agent import MomentumAgent
from agents.market_making_agent import MarketMakingAgent
from agents.macdv_agent import MACDvAgent
# Phase 6 agents
from agents.session_agent import SessionAgent
from agents.index_spread_agent import IndexSpreadAgent
from agents.ttm_squeeze_agent import TTMSqueezeAgent
from agents.event_driven_agent import EventDrivenAgent
from agents.mean_reversion_agent import MeanReversionAgent
# Core agents
from agents.cio_agent import CIOAgent
from agents.risk_agent import RiskAgent
from agents.compliance_agent import ComplianceAgent
from agents.execution_agent import ExecutionAgentImpl

# Legacy import for backwards compatibility
from agents.risk_compliance_agent import RiskComplianceAgent

__all__ = [
    # Signal Agents (parallel execution - fan-out)
    "MacroAgent",
    "StatArbAgent",
    "MomentumAgent",
    "MarketMakingAgent",
    "MACDvAgent",
    # Phase 6 Signal Agents
    "SessionAgent",
    "IndexSpreadAgent",
    "TTMSqueezeAgent",
    "EventDrivenAgent",
    "MeanReversionAgent",
    # Decision Agent (single authority per CLAUDE.md)
    "CIOAgent",
    # Validation Agents (sequential after CIO)
    "RiskAgent",
    "ComplianceAgent",
    # Execution Agent (only agent that sends orders)
    "ExecutionAgentImpl",
    # Legacy
    "RiskComplianceAgent",
]
