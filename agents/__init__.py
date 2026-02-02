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
from agents.options_vol_agent import OptionsVolAgent
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
    "OptionsVolAgent",
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
