"""
Agent Factory
=============

Factory for creating and configuring trading agents.

This module extracts agent creation logic from TradingFirmOrchestrator
following the Single Responsibility Principle (SRP).

The factory:
- Creates signal agents (parallel execution)
- Creates decision agent (CIO - single authority)
- Creates validation agents (Risk, Compliance)
- Creates execution agent
- Creates compliance/surveillance agents (EU/AMF)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.agent_base import AgentConfig

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.logger import AuditLogger
    from core.broker import IBBroker


logger = logging.getLogger(__name__)


@dataclass
class AgentFactoryConfig:
    """Configuration for agent factory."""
    agents_config: dict[str, Any] = field(default_factory=dict)
    risk_config: dict[str, Any] = field(default_factory=dict)
    compliance_config: dict[str, Any] = field(default_factory=dict)
    surveillance_config: dict[str, Any] = field(default_factory=dict)
    transaction_reporting_config: dict[str, Any] = field(default_factory=dict)
    sector_map: dict[str, str] = field(default_factory=dict)


@dataclass
class CreatedAgents:
    """Container for all created agents."""
    signal_agents: list[Any] = field(default_factory=list)
    cio_agent: Any = None
    risk_agent: Any = None
    compliance_agent: Any = None
    execution_agent: Any = None
    surveillance_agent: Any = None
    transaction_reporting_agent: Any = None

    def get_all_agents(self) -> list[Any]:
        """Get all non-None agents as a list."""
        agents = list(self.signal_agents)
        if self.cio_agent:
            agents.append(self.cio_agent)
        if self.risk_agent:
            agents.append(self.risk_agent)
        if self.compliance_agent:
            agents.append(self.compliance_agent)
        if self.execution_agent:
            agents.append(self.execution_agent)
        if self.surveillance_agent:
            agents.append(self.surveillance_agent)
        if self.transaction_reporting_agent:
            agents.append(self.transaction_reporting_agent)
        return agents


class AgentFactory:
    """
    Factory for creating trading system agents.

    Centralizes agent creation logic, improving testability
    and following Single Responsibility Principle.

    Usage:
        factory = AgentFactory(event_bus, audit_logger, broker, config)
        agents = factory.create_all_agents()
    """

    def __init__(
        self,
        event_bus: "EventBus",
        audit_logger: "AuditLogger",
        broker: "IBBroker | None" = None,
        config: AgentFactoryConfig | None = None,
    ):
        self._event_bus = event_bus
        self._audit_logger = audit_logger
        self._broker = broker
        self._config = config or AgentFactoryConfig()

    def create_all_agents(self) -> CreatedAgents:
        """
        Create all trading agents.

        Returns:
            CreatedAgents container with all initialized agents
        """
        result = CreatedAgents()

        # Create signal agents (parallel execution)
        result.signal_agents = self._create_signal_agents()

        # Create decision agent (CIO)
        result.cio_agent = self._create_cio_agent()

        # Create validation agents
        result.risk_agent = self._create_risk_agent()
        result.compliance_agent = self._create_compliance_agent()

        # Create execution agent
        result.execution_agent = self._create_execution_agent()

        # Create compliance/surveillance agents
        result.surveillance_agent = self._create_surveillance_agent()
        result.transaction_reporting_agent = self._create_transaction_reporting_agent()

        self._log_agent_summary(result)

        return result

    def _create_signal_agents(self) -> list[Any]:
        """Create all signal agents (parallel fan-out)."""
        from agents.macro_agent import MacroAgent
        from agents.stat_arb_agent import StatArbAgent
        from agents.momentum_agent import MomentumAgent
        from agents.market_making_agent import MarketMakingAgent

        agents_config = self._config.agents_config
        signal_agents = []

        logger.info("Creating signal agents...")

        # Macro Agent
        if agents_config.get("macro", {}).get("enabled", True):
            macro_agent = MacroAgent(
                config=AgentConfig(
                    name="MacroAgent",
                    enabled=True,
                    parameters=agents_config.get("macro", {}),
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            signal_agents.append(macro_agent)
            self._event_bus.register_signal_agent("MacroAgent")

        # StatArb Agent
        if agents_config.get("stat_arb", {}).get("enabled", True):
            stat_arb_agent = StatArbAgent(
                config=AgentConfig(
                    name="StatArbAgent",
                    enabled=True,
                    parameters=agents_config.get("stat_arb", {}),
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            signal_agents.append(stat_arb_agent)
            self._event_bus.register_signal_agent("StatArbAgent")

        # Momentum Agent
        if agents_config.get("momentum", {}).get("enabled", True):
            momentum_agent = MomentumAgent(
                config=AgentConfig(
                    name="MomentumAgent",
                    enabled=True,
                    parameters=agents_config.get("momentum", {}),
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            signal_agents.append(momentum_agent)
            self._event_bus.register_signal_agent("MomentumAgent")

        # Market Making Agent
        if agents_config.get("market_making", {}).get("enabled", True):
            mm_agent = MarketMakingAgent(
                config=AgentConfig(
                    name="MarketMakingAgent",
                    enabled=True,
                    parameters=agents_config.get("market_making", {}),
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            signal_agents.append(mm_agent)
            self._event_bus.register_signal_agent("MarketMakingAgent")

        # Sentiment Agent (LLM-powered news analysis)
        if agents_config.get("sentiment", {}).get("enabled", True):
            from agents.sentiment_agent import SentimentAgent
            from core.llm_client import LLMClient

            sentiment_config = agents_config.get("sentiment", {})
            llm_client = LLMClient(config=sentiment_config)

            sentiment_agent = SentimentAgent(
                config=AgentConfig(
                    name="SentimentAgent",
                    enabled=True,
                    parameters=sentiment_config,
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
                llm_client=llm_client,
            )
            signal_agents.append(sentiment_agent)
            self._event_bus.register_signal_agent("SentimentAgent")

        # Chart Analysis Agent (Claude Vision pattern recognition)
        if agents_config.get("chart_analysis", {}).get("enabled", True):
            from agents.chart_analysis_agent import ChartAnalysisAgent

            chart_config = agents_config.get("chart_analysis", {})
            chart_agent = ChartAnalysisAgent(
                config=AgentConfig(
                    name="ChartAnalysisAgent",
                    enabled=True,
                    parameters=chart_config,
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            signal_agents.append(chart_agent)
            self._event_bus.register_signal_agent("ChartAnalysisAgent")

        # Forecasting Agent (LLM-powered price prediction)
        if agents_config.get("forecasting", {}).get("enabled", True):
            from agents.forecasting_agent import ForecastingAgent
            from core.llm_client import LLMClient

            forecasting_config = agents_config.get("forecasting", {})
            llm_client = LLMClient(config=forecasting_config.get("llm", {}))

            forecasting_agent = ForecastingAgent(
                config=AgentConfig(
                    name="ForecastingAgent",
                    enabled=True,
                    parameters=forecasting_config,
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
                llm_client=llm_client,
            )
            signal_agents.append(forecasting_agent)
            self._event_bus.register_signal_agent("ForecastingAgent")

        # ============ Phase 6 New Strategy Agents ============

        # Session Agent (Opening Range Breakout, Session Momentum)
        if agents_config.get("session", {}).get("enabled", True):
            from agents.session_agent import SessionAgent

            session_config = agents_config.get("session", {})
            session_agent = SessionAgent(
                config=AgentConfig(
                    name="SessionAgent",
                    enabled=True,
                    parameters=session_config,
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            signal_agents.append(session_agent)
            self._event_bus.register_signal_agent("SessionAgent")

        # Index Spread Agent (MES/MNQ pairs trading)
        if agents_config.get("index_spread", {}).get("enabled", True):
            from agents.index_spread_agent import IndexSpreadAgent

            spread_config = agents_config.get("index_spread", {})
            spread_agent = IndexSpreadAgent(
                config=AgentConfig(
                    name="IndexSpreadAgent",
                    enabled=True,
                    parameters=spread_config,
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            signal_agents.append(spread_agent)
            self._event_bus.register_signal_agent("IndexSpreadAgent")

        # TTM Squeeze Agent (Volatility Breakout)
        if agents_config.get("ttm_squeeze", {}).get("enabled", True):
            from agents.ttm_squeeze_agent import TTMSqueezeAgent

            squeeze_config = agents_config.get("ttm_squeeze", {})
            squeeze_agent = TTMSqueezeAgent(
                config=AgentConfig(
                    name="TTMSqueezeAgent",
                    enabled=True,
                    parameters=squeeze_config,
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            signal_agents.append(squeeze_agent)
            self._event_bus.register_signal_agent("TTMSqueezeAgent")

        # Event-Driven Agent (FOMC, NFP, CPI)
        if agents_config.get("event_driven", {}).get("enabled", True):
            from agents.event_driven_agent import EventDrivenAgent

            event_config = agents_config.get("event_driven", {})
            event_agent = EventDrivenAgent(
                config=AgentConfig(
                    name="EventDrivenAgent",
                    enabled=True,
                    parameters=event_config,
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            signal_agents.append(event_agent)
            self._event_bus.register_signal_agent("EventDrivenAgent")

        # Mean Reversion Agent (RSI, Bollinger Bands, Z-score)
        if agents_config.get("mean_reversion", {}).get("enabled", True):
            from agents.mean_reversion_agent import MeanReversionAgent

            reversion_config = agents_config.get("mean_reversion", {})
            reversion_agent = MeanReversionAgent(
                config=AgentConfig(
                    name="MeanReversionAgent",
                    enabled=True,
                    parameters=reversion_config,
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            signal_agents.append(reversion_agent)
            self._event_bus.register_signal_agent("MeanReversionAgent")

        # MACD-v Agent (Volatility-Normalized MACD)
        if agents_config.get("macdv", {}).get("enabled", True):
            from agents.macdv_agent import MACDvAgent

            macdv_config = agents_config.get("macdv", {})
            macdv_agent = MACDvAgent(
                config=AgentConfig(
                    name="MACDvAgent",
                    enabled=True,
                    parameters=macdv_config,
                ),
                event_bus=self._event_bus,
                audit_logger=self._audit_logger,
            )
            signal_agents.append(macdv_agent)
            self._event_bus.register_signal_agent("MACDvAgent")

        logger.info(f"Created {len(signal_agents)} signal agents")
        return signal_agents

    def _create_cio_agent(self) -> Any:
        """Create CIO agent (THE decision maker)."""
        from agents.cio_agent import CIOAgent

        logger.info("Creating CIO agent...")
        return CIOAgent(
            config=AgentConfig(
                name="CIOAgent",
                enabled=True,
                parameters=self._config.agents_config.get("cio", {}),
            ),
            event_bus=self._event_bus,
            audit_logger=self._audit_logger,
        )

    def _create_risk_agent(self) -> Any:
        """Create Risk agent."""
        from agents.risk_agent import RiskAgent

        logger.info("Creating Risk agent...")

        risk_config = self._config.risk_config
        # Get drawdown config (may be nested dict or flat)
        drawdown_config = risk_config.get("drawdown", {})
        max_drawdown_pct = risk_config.get("max_drawdown_pct", 10.0)

        risk_params = {
            "limits": {
                "max_position_size_pct": risk_config.get("max_position_size_pct", 5.0),
                "max_sector_exposure_pct": risk_config.get("max_sector_exposure_pct", 20.0),
                "max_leverage": risk_config.get("max_leverage", 2.0),
                "max_portfolio_var_pct": risk_config.get("max_portfolio_var_pct", 2.0),
                "max_daily_loss_pct": risk_config.get("max_daily_loss_pct", 3.0),
                "max_drawdown_pct": max_drawdown_pct,
            },
            "rate_limits": {
                "max_orders_per_minute": risk_config.get("max_orders_per_minute", 10),
                "min_order_interval_ms": risk_config.get("min_order_interval_ms", 100),
            },
            "sector_map": self._config.sector_map,
            # Pass drawdown config with halt_pct defaulting to max_drawdown_pct
            "drawdown": {
                "warning_pct": drawdown_config.get("warning_pct", 5.0),
                "reduce_pct": drawdown_config.get("reduce_pct", 7.5),
                "halt_pct": drawdown_config.get("halt_pct", max_drawdown_pct),
                "position_reduction_factor": drawdown_config.get("position_reduction_factor", 0.5),
            },
        }

        return RiskAgent(
            config=AgentConfig(
                name="RiskAgent",
                enabled=True,
                parameters=risk_params,
            ),
            event_bus=self._event_bus,
            audit_logger=self._audit_logger,
            broker=self._broker,
        )

    def _create_compliance_agent(self) -> Any:
        """Create Compliance agent (EU/AMF)."""
        from agents.compliance_agent import ComplianceAgent

        logger.info("Creating Compliance agent...")

        compliance_config = self._config.compliance_config
        compliance_params = {
            "jurisdiction": compliance_config.get("jurisdiction", "EU"),
            "restricted_instruments": compliance_config.get("banned_instruments", []),
            "allowed_asset_classes": compliance_config.get("allowed_asset_classes", ["equity", "etf"]),
        }

        return ComplianceAgent(
            config=AgentConfig(
                name="ComplianceAgent",
                enabled=True,
                parameters=compliance_params,
            ),
            event_bus=self._event_bus,
            audit_logger=self._audit_logger,
        )

    def _create_execution_agent(self) -> Any:
        """Create Execution agent (ONLY agent that sends orders)."""
        from agents.execution_agent import ExecutionAgentImpl

        logger.info("Creating Execution agent...")
        return ExecutionAgentImpl(
            config=AgentConfig(
                name="ExecutionAgent",
                enabled=True,
                parameters=self._config.agents_config.get("execution", {}),
            ),
            event_bus=self._event_bus,
            audit_logger=self._audit_logger,
            broker=self._broker,
        )

    def _create_surveillance_agent(self) -> Any | None:
        """Create Surveillance agent (MAR 2014/596/EU)."""
        agents_config = self._config.agents_config

        if not agents_config.get("surveillance", {}).get("enabled", True):
            return None

        from agents.surveillance_agent import SurveillanceAgent

        logger.info("Creating Surveillance agent...")

        surveillance_config = self._config.surveillance_config
        surveillance_params = {
            "wash_trading_detection": surveillance_config.get("wash_trading_detection", True),
            "spoofing_detection": surveillance_config.get("spoofing_detection", True),
            "quote_stuffing_detection": surveillance_config.get("quote_stuffing_detection", True),
            "layering_detection": surveillance_config.get("layering_detection", True),
            "wash_trading_window_seconds": surveillance_config.get("wash_trading_window_seconds", 60),
            "spoofing_cancel_threshold": surveillance_config.get("spoofing_cancel_threshold", 0.8),
            "quote_stuffing_rate_per_second": surveillance_config.get("quote_stuffing_rate_per_second", 10),
        }

        return SurveillanceAgent(
            config=AgentConfig(
                name="SurveillanceAgent",
                enabled=True,
                parameters=surveillance_params,
            ),
            event_bus=self._event_bus,
            audit_logger=self._audit_logger,
        )

    def _create_transaction_reporting_agent(self) -> Any | None:
        """Create Transaction Reporting agent (ESMA RTS 22/23)."""
        agents_config = self._config.agents_config

        if not agents_config.get("transaction_reporting", {}).get("enabled", True):
            return None

        from agents.transaction_reporting_agent import TransactionReportingAgent

        logger.info("Creating Transaction Reporting agent...")

        reporting_config = self._config.transaction_reporting_config
        reporting_params = {
            "enabled": reporting_config.get("enabled", True),
            "reporting_deadline_minutes": reporting_config.get("reporting_deadline_minutes", 15),
            "firm_lei": reporting_config.get("firm_lei", ""),
            "firm_country": reporting_config.get("firm_country", "FR"),
            "default_venue": reporting_config.get("default_venue", "XPAR"),
        }

        return TransactionReportingAgent(
            config=AgentConfig(
                name="TransactionReportingAgent",
                enabled=True,
                parameters=reporting_params,
            ),
            event_bus=self._event_bus,
            audit_logger=self._audit_logger,
        )

    def _log_agent_summary(self, agents: CreatedAgents) -> None:
        """Log summary of created agents."""
        logger.info("All agents created:")
        logger.info(f"  Signal agents: {', '.join(a.name for a in agents.signal_agents)}")
        logger.info("  Decision: CIOAgent")
        logger.info("  Validation: RiskAgent -> ComplianceAgent")
        logger.info("  Execution: ExecutionAgent")
        if agents.surveillance_agent:
            logger.info("  Surveillance: SurveillanceAgent (MAR 2014/596/EU)")
        if agents.transaction_reporting_agent:
            logger.info("  Reporting: TransactionReportingAgent (ESMA RTS 22/23)")
