"""
AI Trading Firm - Core Module
=============================

Core infrastructure components for the multi-agent trading system.
Per CLAUDE.md: Testable, observable, and auditable.
"""

from core.events import (
    Event,
    MarketDataEvent,
    SignalEvent,
    DecisionEvent,
    ValidatedDecisionEvent,
    OrderEvent,
    FillEvent,
    RiskAlertEvent,
)
from core.event_bus import EventBus
from core.agent_base import BaseAgent
from core.broker import IBBroker
from core.logger import AuditLogger
from core.monitoring import (
    MonitoringSystem,
    MetricsCollector,
    AlertManager,
    AnomalyDetector,
    AlertSeverity,
    MetricType,
)

__all__ = [
    # Events
    "Event",
    "MarketDataEvent",
    "SignalEvent",
    "DecisionEvent",
    "ValidatedDecisionEvent",
    "OrderEvent",
    "FillEvent",
    "RiskAlertEvent",
    # Core components
    "EventBus",
    "BaseAgent",
    "IBBroker",
    "AuditLogger",
    # Monitoring
    "MonitoringSystem",
    "MetricsCollector",
    "AlertManager",
    "AnomalyDetector",
    "AlertSeverity",
    "MetricType",
]
