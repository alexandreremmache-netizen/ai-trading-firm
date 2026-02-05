"""
Agent Status Tracker
====================

Real-time agent status tracking component for the trading system dashboard.

Monitors all trading agents including:
- Decision Agent: CIO
- Validation Agents: Risk, Compliance
- Execution Agent
- Signal Agents: Macro, StatArb, Momentum, MarketMaking, MACDv, Sentiment, Forecasting

Features:
- Real-time status tracking (active/idle/error)
- Event processing metrics
- Agent health scoring
- Thread-safe with asyncio locks
- WebSocket-ready export to dict
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent_base import BaseAgent


logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Status of an agent in the system."""
    ACTIVE = "active"          # Agent is actively processing events
    IDLE = "idle"              # Agent is running but not processing
    ERROR = "error"            # Agent has encountered an error
    STOPPED = "stopped"        # Agent is not running
    STARTING = "starting"      # Agent is starting up
    SHUTTING_DOWN = "shutting_down"  # Agent is shutting down


class AgentType(Enum):
    """Classification of agent types in the trading system."""
    DECISION = "decision"      # CIO - the single decision authority
    VALIDATION = "validation"  # Risk, Compliance
    EXECUTION = "execution"    # Execution agent
    SIGNAL = "signal"          # Signal-generating agents
    SURVEILLANCE = "surveillance"  # Compliance monitoring agents
    REPORTING = "reporting"    # Transaction reporting agents


# Map agent names to their types
AGENT_TYPE_MAPPING: dict[str, AgentType] = {
    # Decision Agent
    "CIOAgent": AgentType.DECISION,
    "CIO": AgentType.DECISION,
    # Validation Agents
    "RiskAgent": AgentType.VALIDATION,
    "Risk": AgentType.VALIDATION,
    "ComplianceAgent": AgentType.VALIDATION,
    "Compliance": AgentType.VALIDATION,
    # Execution Agent
    "ExecutionAgent": AgentType.EXECUTION,
    "Execution": AgentType.EXECUTION,
    # Signal Agents
    "MacroAgent": AgentType.SIGNAL,
    "Macro": AgentType.SIGNAL,
    "StatArbAgent": AgentType.SIGNAL,
    "StatArb": AgentType.SIGNAL,
    "MomentumAgent": AgentType.SIGNAL,
    "Momentum": AgentType.SIGNAL,
    "MarketMakingAgent": AgentType.SIGNAL,
    "MarketMaking": AgentType.SIGNAL,
    "SentimentAgent": AgentType.SIGNAL,
    "Sentiment": AgentType.SIGNAL,
    "ForecastingAgent": AgentType.SIGNAL,
    "Forecasting": AgentType.SIGNAL,
    "MACDvAgent": AgentType.SIGNAL,
    "MACDv": AgentType.SIGNAL,
    # Surveillance Agents
    "SurveillanceAgent": AgentType.SURVEILLANCE,
    "Surveillance": AgentType.SURVEILLANCE,
    # Reporting Agents
    "TransactionReportingAgent": AgentType.REPORTING,
    "TransactionReporting": AgentType.REPORTING,
}


@dataclass
class AgentStatusRecord:
    """
    Status record for a single agent.

    Captures comprehensive metrics for monitoring and health assessment.
    """
    agent_name: str
    agent_type: AgentType
    status: AgentStatus = AgentStatus.STOPPED
    last_event_time: datetime | None = None
    events_processed: int = 0
    current_task: str | None = None
    error_count: int = 0
    last_error: str | None = None
    last_error_time: datetime | None = None
    started_at: datetime | None = None
    uptime_seconds: float = 0.0
    # Performance metrics
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    events_per_minute: float = 0.0
    # Health metrics
    health_score: float = 100.0  # 0-100 scale
    consecutive_errors: int = 0
    last_health_check: datetime | None = None
    # Recent event times for throughput calculation
    _recent_event_times: deque = field(default_factory=lambda: deque(maxlen=100))
    # Processing time samples for averaging
    _processing_times: deque = field(default_factory=lambda: deque(maxlen=100))

    def __post_init__(self):
        """Initialize deques if they were not set."""
        if not isinstance(self._recent_event_times, deque):
            object.__setattr__(self, '_recent_event_times', deque(maxlen=100))
        if not isinstance(self._processing_times, deque):
            object.__setattr__(self, '_processing_times', deque(maxlen=100))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "last_event_time": self.last_event_time.isoformat() if self.last_event_time else None,
            "events_processed": self.events_processed,
            "current_task": self.current_task,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
            "max_processing_time_ms": round(self.max_processing_time_ms, 2),
            "events_per_minute": round(self.events_per_minute, 2),
            "health_score": round(self.health_score, 1),
            "consecutive_errors": self.consecutive_errors,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
        }


@dataclass
class AgentHealthMetrics:
    """
    Aggregated health metrics across all agents.

    Provides system-wide health overview for monitoring.
    """
    total_agents: int = 0
    active_agents: int = 0
    idle_agents: int = 0
    error_agents: int = 0
    stopped_agents: int = 0
    avg_health_score: float = 100.0
    min_health_score: float = 100.0
    min_health_agent: str | None = None
    total_events_processed: int = 0
    total_errors: int = 0
    system_health: str = "healthy"  # healthy, degraded, critical
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "total_agents": self.total_agents,
            "active_agents": self.active_agents,
            "idle_agents": self.idle_agents,
            "error_agents": self.error_agents,
            "stopped_agents": self.stopped_agents,
            "avg_health_score": round(self.avg_health_score, 1),
            "min_health_score": round(self.min_health_score, 1),
            "min_health_agent": self.min_health_agent,
            "total_events_processed": self.total_events_processed,
            "total_errors": self.total_errors,
            "system_health": self.system_health,
            "last_updated": self.last_updated.isoformat(),
        }


class AgentStatusTracker:
    """
    Tracks and monitors the status of all trading agents.

    Provides real-time visibility into agent health and performance
    for the trading system dashboard.

    Usage:
        tracker = AgentStatusTracker()

        # Register agents at startup
        tracker.register_agent("CIOAgent", AgentType.DECISION)
        tracker.register_agent("RiskAgent", AgentType.VALIDATION)

        # Update status as agents process events
        tracker.update_status("CIOAgent", AgentStatus.ACTIVE, current_task="Processing signals")
        tracker.record_event_processed("CIOAgent", processing_time_ms=15.5)

        # Get individual agent status
        status = tracker.get_agent_status("CIOAgent")

        # Get all statuses for dashboard
        all_statuses = tracker.get_all_statuses()

        # Get health metrics
        health = tracker.get_agent_health("CIOAgent")
        system_health = tracker.get_system_health()

        # Export for WebSocket streaming
        data = tracker.to_dict()

    Thread Safety:
        All public methods acquire an asyncio lock before modifying state,
        making the tracker safe for concurrent use across multiple agents.
    """

    # Default idle threshold (seconds without events = idle)
    IDLE_THRESHOLD_SECONDS = 30.0

    # Health score thresholds
    HEALTH_CRITICAL_THRESHOLD = 50.0
    HEALTH_DEGRADED_THRESHOLD = 75.0

    # Error impact on health score
    ERROR_HEALTH_PENALTY = 10.0
    CONSECUTIVE_ERROR_MULTIPLIER = 2.0

    # Activity bonus for health score
    ACTIVITY_HEALTH_BONUS = 5.0

    def __init__(
        self,
        idle_threshold_seconds: float = IDLE_THRESHOLD_SECONDS,
    ):
        """
        Initialize the agent status tracker.

        Args:
            idle_threshold_seconds: Seconds without events before agent is considered idle
        """
        self._idle_threshold = idle_threshold_seconds
        self._agents: dict[str, AgentStatusRecord] = {}
        self._lock = asyncio.Lock()

        # Predefined agent list based on CLAUDE.md architecture (17 signal + 6 core = 23 total)
        self._expected_agents = [
            # Core decision/validation/execution agents
            ("CIOAgent", AgentType.DECISION),
            ("RiskAgent", AgentType.VALIDATION),
            ("ComplianceAgent", AgentType.VALIDATION),
            ("ExecutionAgent", AgentType.EXECUTION),
            # Signal agents - Core (always enabled)
            ("MacroAgent", AgentType.SIGNAL),
            ("StatArbAgent", AgentType.SIGNAL),
            ("MomentumAgent", AgentType.SIGNAL),
            ("MarketMakingAgent", AgentType.SIGNAL),
            # Signal agents - LLM (may be disabled to save tokens)
            ("SentimentAgent", AgentType.SIGNAL),
            ("ChartAnalysisAgent", AgentType.SIGNAL),
            ("ForecastingAgent", AgentType.SIGNAL),
            # Signal agents - Phase 6 (new strategies)
            ("SessionAgent", AgentType.SIGNAL),
            ("IndexSpreadAgent", AgentType.SIGNAL),
            ("TTMSqueezeAgent", AgentType.SIGNAL),
            ("EventDrivenAgent", AgentType.SIGNAL),
            ("MeanReversionAgent", AgentType.SIGNAL),
            # Signal agents - Award-winning indicators
            ("MACDvAgent", AgentType.SIGNAL),  # MACD-v (Charles H. Dow Award 2022)
            # Compliance agents
            ("SurveillanceAgent", AgentType.SURVEILLANCE),
            ("TransactionReportingAgent", AgentType.REPORTING),
        ]

        logger.info(f"AgentStatusTracker initialized with idle_threshold={idle_threshold_seconds}s")

    async def register_agent(
        self,
        agent_name: str,
        agent_type: AgentType | None = None,
    ) -> AgentStatusRecord:
        """
        Register an agent for status tracking.

        Args:
            agent_name: Name of the agent
            agent_type: Type of the agent (auto-detected if not provided)

        Returns:
            AgentStatusRecord for the registered agent
        """
        async with self._lock:
            # Auto-detect agent type if not provided
            if agent_type is None:
                agent_type = AGENT_TYPE_MAPPING.get(agent_name, AgentType.SIGNAL)

            # Create or update record
            if agent_name not in self._agents:
                record = AgentStatusRecord(
                    agent_name=agent_name,
                    agent_type=agent_type,
                    status=AgentStatus.STOPPED,
                )
                self._agents[agent_name] = record
                logger.info(f"Registered agent: {agent_name} (type={agent_type.value})")
            else:
                record = self._agents[agent_name]
                record.agent_type = agent_type

            return record

    def register_agent_sync(
        self,
        agent_name: str,
        agent_type: AgentType | None = None,
    ) -> AgentStatusRecord:
        """
        Synchronous version of register_agent for non-async contexts.

        Args:
            agent_name: Name of the agent
            agent_type: Type of the agent (auto-detected if not provided)

        Returns:
            AgentStatusRecord for the registered agent
        """
        # Auto-detect agent type if not provided
        if agent_type is None:
            agent_type = AGENT_TYPE_MAPPING.get(agent_name, AgentType.SIGNAL)

        # Create or update record
        if agent_name not in self._agents:
            record = AgentStatusRecord(
                agent_name=agent_name,
                agent_type=agent_type,
                status=AgentStatus.STOPPED,
            )
            self._agents[agent_name] = record
            logger.info(f"Registered agent (sync): {agent_name} (type={agent_type.value})")
        else:
            record = self._agents[agent_name]
            record.agent_type = agent_type

        return record

    async def register_from_base_agent(self, agent: BaseAgent) -> AgentStatusRecord:
        """
        Register an agent from a BaseAgent instance.

        Args:
            agent: BaseAgent instance to register

        Returns:
            AgentStatusRecord for the registered agent
        """
        agent_name = agent.name
        agent_type = AGENT_TYPE_MAPPING.get(agent_name, AgentType.SIGNAL)
        return await self.register_agent(agent_name, agent_type)

    async def unregister_agent(self, agent_name: str) -> bool:
        """
        Unregister an agent from tracking.

        Args:
            agent_name: Name of the agent to unregister

        Returns:
            True if agent was unregistered, False if not found
        """
        async with self._lock:
            if agent_name in self._agents:
                del self._agents[agent_name]
                logger.info(f"Unregistered agent: {agent_name}")
                return True
            return False

    async def update_status(
        self,
        agent_name: str,
        status: AgentStatus,
        current_task: str | None = None,
        error_message: str | None = None,
    ) -> bool:
        """
        Update the status of an agent.

        Args:
            agent_name: Name of the agent
            status: New status for the agent
            current_task: Description of current task (optional)
            error_message: Error message if status is ERROR (optional)

        Returns:
            True if update was successful, False if agent not found
        """
        async with self._lock:
            if agent_name not in self._agents:
                # Auto-register unknown agents
                agent_type = AGENT_TYPE_MAPPING.get(agent_name, AgentType.SIGNAL)
                self._agents[agent_name] = AgentStatusRecord(
                    agent_name=agent_name,
                    agent_type=agent_type,
                )

            record = self._agents[agent_name]
            now = datetime.now(timezone.utc)

            # Update status
            old_status = record.status
            record.status = status
            record.current_task = current_task

            # Handle status transitions
            if status == AgentStatus.ACTIVE and old_status == AgentStatus.STOPPED:
                record.started_at = now
                record.consecutive_errors = 0

            if status == AgentStatus.ERROR:
                record.error_count += 1
                record.consecutive_errors += 1
                record.last_error = error_message
                record.last_error_time = now
            elif status == AgentStatus.ACTIVE:
                record.consecutive_errors = 0

            if status == AgentStatus.STOPPED:
                record.current_task = None

            # Update uptime
            if record.started_at and status != AgentStatus.STOPPED:
                record.uptime_seconds = (now - record.started_at).total_seconds()

            # Recalculate health score
            self._calculate_health_score(record)

            logger.debug(
                f"Agent {agent_name} status updated: {old_status.value} -> {status.value}"
            )

            return True

    async def record_event_processed(
        self,
        agent_name: str,
        processing_time_ms: float | None = None,
    ) -> bool:
        """
        Record that an agent processed an event.

        Args:
            agent_name: Name of the agent
            processing_time_ms: Time taken to process the event (optional)

        Returns:
            True if recorded successfully, False if agent not found
        """
        async with self._lock:
            if agent_name not in self._agents:
                return False

            record = self._agents[agent_name]
            now = datetime.now(timezone.utc)

            # Update event metrics
            record.events_processed += 1
            record.last_event_time = now
            record._recent_event_times.append(now)

            # Update processing time metrics
            if processing_time_ms is not None:
                record._processing_times.append(processing_time_ms)

                # Calculate average
                times = list(record._processing_times)
                if times:
                    record.avg_processing_time_ms = sum(times) / len(times)
                    record.max_processing_time_ms = max(
                        record.max_processing_time_ms,
                        processing_time_ms
                    )

            # Calculate events per minute
            one_minute_ago = now - timedelta(minutes=1)
            recent = [t for t in record._recent_event_times if t >= one_minute_ago]
            record.events_per_minute = len(recent)

            # Update status to active if was idle
            if record.status == AgentStatus.IDLE:
                record.status = AgentStatus.ACTIVE

            # Reset consecutive errors on successful event
            record.consecutive_errors = 0

            # Recalculate health score
            self._calculate_health_score(record)

            return True

    async def record_error(
        self,
        agent_name: str,
        error_message: str,
    ) -> bool:
        """
        Record an error for an agent.

        Args:
            agent_name: Name of the agent
            error_message: Description of the error

        Returns:
            True if recorded successfully, False if agent not found
        """
        async with self._lock:
            if agent_name not in self._agents:
                return False

            record = self._agents[agent_name]
            now = datetime.now(timezone.utc)

            record.error_count += 1
            record.consecutive_errors += 1
            record.last_error = error_message
            record.last_error_time = now

            # Recalculate health score
            self._calculate_health_score(record)

            logger.warning(f"Agent {agent_name} error recorded: {error_message}")

            return True

    def _calculate_health_score(self, record: AgentStatusRecord) -> float:
        """
        Calculate health score for an agent.

        Health score is based on:
        - Base score of 100
        - Penalty for errors (more for consecutive errors)
        - Penalty for being stopped or in error state
        - Bonus for recent activity

        Args:
            record: Agent status record to calculate score for

        Returns:
            Health score (0-100)
        """
        now = datetime.now(timezone.utc)
        score = 100.0

        # Penalty for errors
        if record.error_count > 0:
            error_penalty = min(
                self.ERROR_HEALTH_PENALTY * record.error_count,
                30.0  # Cap at 30 points
            )
            score -= error_penalty

        # Extra penalty for consecutive errors
        if record.consecutive_errors > 1:
            consecutive_penalty = min(
                self.CONSECUTIVE_ERROR_MULTIPLIER * record.consecutive_errors,
                20.0  # Cap at 20 points
            )
            score -= consecutive_penalty

        # Penalty for status
        if record.status == AgentStatus.STOPPED:
            score -= 25.0
        elif record.status == AgentStatus.ERROR:
            score -= 40.0
        elif record.status == AgentStatus.SHUTTING_DOWN:
            score -= 10.0

        # Penalty for inactivity (idle for too long)
        if record.last_event_time:
            idle_seconds = (now - record.last_event_time).total_seconds()
            if idle_seconds > self._idle_threshold * 10:  # Very idle
                score -= 15.0
            elif idle_seconds > self._idle_threshold * 5:  # Moderately idle
                score -= 5.0

        # Bonus for recent activity
        if record.events_per_minute > 0:
            activity_bonus = min(
                self.ACTIVITY_HEALTH_BONUS * (record.events_per_minute / 10),
                10.0  # Cap at 10 points
            )
            score += activity_bonus

        # Clamp to valid range
        score = max(0.0, min(100.0, score))

        record.health_score = score
        record.last_health_check = now

        return score

    async def get_agent_status(self, agent_name: str) -> AgentStatusRecord | None:
        """
        Get the status record for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            AgentStatusRecord or None if not found
        """
        async with self._lock:
            return self._agents.get(agent_name)

    async def get_all_statuses(self) -> dict[str, AgentStatusRecord]:
        """
        Get status records for all registered agents.

        Returns:
            Dictionary mapping agent names to their status records
        """
        async with self._lock:
            # Update idle status for all agents before returning
            self._update_idle_statuses()
            return dict(self._agents)

    def _update_idle_statuses(self) -> None:
        """Update status to IDLE for agents that haven't processed events recently."""
        now = datetime.now(timezone.utc)
        threshold = now - timedelta(seconds=self._idle_threshold)

        for record in self._agents.values():
            if record.status == AgentStatus.ACTIVE:
                if record.last_event_time is None or record.last_event_time < threshold:
                    record.status = AgentStatus.IDLE

            # Update uptime
            if record.started_at and record.status not in (AgentStatus.STOPPED,):
                record.uptime_seconds = (now - record.started_at).total_seconds()

    async def get_agent_health(self, agent_name: str) -> float:
        """
        Get the health score for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Health score (0-100) or 0 if agent not found
        """
        async with self._lock:
            record = self._agents.get(agent_name)
            if record:
                self._calculate_health_score(record)
                return record.health_score
            return 0.0

    async def get_system_health(self) -> AgentHealthMetrics:
        """
        Get aggregated health metrics for the entire system.

        Returns:
            AgentHealthMetrics with system-wide health overview
        """
        async with self._lock:
            self._update_idle_statuses()

            metrics = AgentHealthMetrics(
                total_agents=len(self._agents),
                last_updated=datetime.now(timezone.utc),
            )

            if not self._agents:
                return metrics

            health_scores = []
            min_health = 100.0
            min_health_agent = None

            for record in self._agents.values():
                # Recalculate health scores
                self._calculate_health_score(record)

                # Count by status
                if record.status == AgentStatus.ACTIVE:
                    metrics.active_agents += 1
                elif record.status == AgentStatus.IDLE:
                    metrics.idle_agents += 1
                elif record.status == AgentStatus.ERROR:
                    metrics.error_agents += 1
                elif record.status == AgentStatus.STOPPED:
                    metrics.stopped_agents += 1

                # Aggregate metrics
                metrics.total_events_processed += record.events_processed
                metrics.total_errors += record.error_count
                health_scores.append(record.health_score)

                # Track minimum health
                if record.health_score < min_health:
                    min_health = record.health_score
                    min_health_agent = record.agent_name

            # Calculate averages
            if health_scores:
                metrics.avg_health_score = sum(health_scores) / len(health_scores)
                metrics.min_health_score = min_health
                metrics.min_health_agent = min_health_agent

            # Determine system health status
            if metrics.avg_health_score >= self.HEALTH_DEGRADED_THRESHOLD:
                metrics.system_health = "healthy"
            elif metrics.avg_health_score >= self.HEALTH_CRITICAL_THRESHOLD:
                metrics.system_health = "degraded"
            else:
                metrics.system_health = "critical"

            # Override to degraded if any agent in error
            if metrics.error_agents > 0:
                if metrics.system_health == "healthy":
                    metrics.system_health = "degraded"

            return metrics

    async def get_agents_by_type(self, agent_type: AgentType) -> list[AgentStatusRecord]:
        """
        Get all agents of a specific type.

        Args:
            agent_type: Type of agents to retrieve

        Returns:
            List of AgentStatusRecord for matching agents
        """
        async with self._lock:
            return [
                record for record in self._agents.values()
                if record.agent_type == agent_type
            ]

    async def get_agents_by_status(self, status: AgentStatus) -> list[AgentStatusRecord]:
        """
        Get all agents with a specific status.

        Args:
            status: Status to filter by

        Returns:
            List of AgentStatusRecord for matching agents
        """
        async with self._lock:
            self._update_idle_statuses()
            return [
                record for record in self._agents.values()
                if record.status == status
            ]

    async def get_signal_agents_status(self) -> list[AgentStatusRecord]:
        """
        Get status of all signal agents.

        Returns:
            List of AgentStatusRecord for signal agents
        """
        return await self.get_agents_by_type(AgentType.SIGNAL)

    async def get_unhealthy_agents(
        self,
        threshold: float = HEALTH_DEGRADED_THRESHOLD,
    ) -> list[AgentStatusRecord]:
        """
        Get agents with health score below threshold.

        Args:
            threshold: Health score threshold (default 75.0)

        Returns:
            List of AgentStatusRecord for unhealthy agents
        """
        async with self._lock:
            unhealthy = []
            for record in self._agents.values():
                self._calculate_health_score(record)
                if record.health_score < threshold:
                    unhealthy.append(record)
            return sorted(unhealthy, key=lambda r: r.health_score)

    async def reset_agent_errors(self, agent_name: str) -> bool:
        """
        Reset error counts for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            True if reset successful, False if agent not found
        """
        async with self._lock:
            record = self._agents.get(agent_name)
            if record:
                record.error_count = 0
                record.consecutive_errors = 0
                record.last_error = None
                record.last_error_time = None
                self._calculate_health_score(record)
                logger.info(f"Reset errors for agent: {agent_name}")
                return True
            return False

    async def reset_all_errors(self) -> int:
        """
        Reset error counts for all agents.

        Returns:
            Number of agents reset
        """
        async with self._lock:
            count = 0
            for record in self._agents.values():
                if record.error_count > 0:
                    record.error_count = 0
                    record.consecutive_errors = 0
                    record.last_error = None
                    record.last_error_time = None
                    self._calculate_health_score(record)
                    count += 1
            logger.info(f"Reset errors for {count} agents")
            return count

    def to_dict(self) -> dict[str, Any]:
        """
        Export tracker state to dictionary for WebSocket streaming.

        Note: This is synchronous for compatibility with simple JSON serialization.
        For async contexts, use to_dict_async().

        Returns:
            Complete tracker state as dict
        """
        self._update_idle_statuses()

        # Calculate system health synchronously
        health_scores = []
        metrics = {
            "total_agents": len(self._agents),
            "active_agents": 0,
            "idle_agents": 0,
            "error_agents": 0,
            "stopped_agents": 0,
            "total_events_processed": 0,
            "total_errors": 0,
        }

        for record in self._agents.values():
            self._calculate_health_score(record)
            health_scores.append(record.health_score)
            metrics["total_events_processed"] += record.events_processed
            metrics["total_errors"] += record.error_count

            if record.status == AgentStatus.ACTIVE:
                metrics["active_agents"] += 1
            elif record.status == AgentStatus.IDLE:
                metrics["idle_agents"] += 1
            elif record.status == AgentStatus.ERROR:
                metrics["error_agents"] += 1
            elif record.status == AgentStatus.STOPPED:
                metrics["stopped_agents"] += 1

        avg_health = sum(health_scores) / len(health_scores) if health_scores else 100.0
        min_health = min(health_scores) if health_scores else 100.0

        # Determine system health
        if avg_health >= self.HEALTH_DEGRADED_THRESHOLD:
            system_health = "healthy"
        elif avg_health >= self.HEALTH_CRITICAL_THRESHOLD:
            system_health = "degraded"
        else:
            system_health = "critical"

        if metrics["error_agents"] > 0 and system_health == "healthy":
            system_health = "degraded"

        return {
            "agents": {
                name: record.to_dict()
                for name, record in self._agents.items()
            },
            "by_type": {
                agent_type.value: [
                    record.to_dict()
                    for record in self._agents.values()
                    if record.agent_type == agent_type
                ]
                for agent_type in AgentType
            },
            "system_health": {
                "status": system_health,
                "avg_health_score": round(avg_health, 1),
                "min_health_score": round(min_health, 1),
                **metrics,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def to_dict_async(self) -> dict[str, Any]:
        """
        Async version of to_dict with proper locking.

        Returns:
            Complete tracker state as dict
        """
        async with self._lock:
            return self.to_dict()

    @property
    def agent_count(self) -> int:
        """Get the number of registered agents."""
        return len(self._agents)

    @property
    def active_agent_count(self) -> int:
        """Get the number of active agents."""
        return sum(
            1 for record in self._agents.values()
            if record.status == AgentStatus.ACTIVE
        )

    @property
    def registered_agent_names(self) -> list[str]:
        """Get list of all registered agent names."""
        return list(self._agents.keys())
