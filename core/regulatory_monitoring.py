"""
Regulatory Change Monitoring and Compliance Report Templates.

This module provides automated monitoring of regulatory changes and
standardized templates for compliance reporting.

Addresses:
- #C42 - Regulatory change monitoring not automated
- #C43 - Compliance report templates outdated
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import hashlib
import json


logger = logging.getLogger(__name__)


class RegulatorType(Enum):
    """Types of financial regulators."""
    AMF = "amf"  # Autorite des Marches Financiers (France)
    ESMA = "esma"  # European Securities and Markets Authority
    FCA = "fca"  # Financial Conduct Authority (UK)
    SEC = "sec"  # Securities and Exchange Commission (US)
    CFTC = "cftc"  # Commodity Futures Trading Commission (US)
    BIS = "bis"  # Bank for International Settlements
    ECB = "ecb"  # European Central Bank


# =============================================================================
# CIRCUIT BREAKER MONITORING (REG-P1-2)
# =============================================================================

class CircuitBreakerStatus(Enum):
    """Exchange circuit breaker status states (REG-P1-2)."""
    NORMAL = "normal"  # Trading normally
    LULD_PAUSE = "luld_pause"  # Limit Up-Limit Down pause
    HALT = "halt"  # Trading halt
    MWCB_LEVEL1 = "mwcb_level1"  # Market-wide circuit breaker Level 1 (7%)
    MWCB_LEVEL2 = "mwcb_level2"  # Market-wide circuit breaker Level 2 (13%)
    MWCB_LEVEL3 = "mwcb_level3"  # Market-wide circuit breaker Level 3 (20%)
    VOLATILITY_AUCTION = "volatility_auction"
    UNKNOWN = "unknown"


@dataclass
class ExchangeCircuitBreakerState:
    """Current state of exchange circuit breaker (REG-P1-2)."""
    exchange_mic: str
    status: CircuitBreakerStatus
    triggered_at: Optional[datetime] = None
    expected_resume: Optional[datetime] = None
    trigger_reason: str = ""
    affected_symbols: List[str] = field(default_factory=list)
    halt_duration_minutes: Optional[int] = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ExchangeCircuitBreakerMonitor:
    """
    Monitors exchange-level circuit breakers (REG-P1-2).

    Tracks market-wide circuit breakers (MWCB), Limit Up-Limit Down (LULD),
    and individual security trading halts across exchanges.
    """

    # MWCB thresholds and halt durations (US markets)
    MWCB_THRESHOLDS = {
        "level1": {"threshold_pct": 7.0, "halt_minutes": 15},
        "level2": {"threshold_pct": 13.0, "halt_minutes": 15},
        "level3": {"threshold_pct": 20.0, "halt_minutes": None},  # Market closed for day
    }

    # LULD price bands by tier
    LULD_BANDS = {
        "tier1": {"band_pct": 5.0},  # S&P 500, Russell 1000
        "tier2": {"band_pct": 10.0},  # Other NMS stocks
    }

    def __init__(self):
        self._exchange_states: Dict[str, ExchangeCircuitBreakerState] = {}
        self._symbol_halts: Dict[str, ExchangeCircuitBreakerState] = {}
        self._halt_history: List[Dict[str, Any]] = []
        self._alert_callbacks: List[Callable[[ExchangeCircuitBreakerState], None]] = []

    def register_alert_callback(self, callback: Callable[[ExchangeCircuitBreakerState], None]):
        """Register callback for circuit breaker alerts."""
        self._alert_callbacks.append(callback)

    def update_exchange_status(
        self,
        exchange_mic: str,
        status: CircuitBreakerStatus,
        trigger_reason: str = "",
        affected_symbols: Optional[List[str]] = None,
        expected_resume: Optional[datetime] = None
    ) -> ExchangeCircuitBreakerState:
        """
        Update exchange circuit breaker status (REG-P1-2).

        Args:
            exchange_mic: Market Identifier Code of exchange
            status: New circuit breaker status
            trigger_reason: Reason for status change
            affected_symbols: List of affected symbols (for halts)
            expected_resume: Expected trading resume time

        Returns:
            Updated ExchangeCircuitBreakerState
        """
        now = datetime.now(timezone.utc)

        # Get halt duration based on status
        halt_duration = None
        if status == CircuitBreakerStatus.MWCB_LEVEL1:
            halt_duration = self.MWCB_THRESHOLDS["level1"]["halt_minutes"]
        elif status == CircuitBreakerStatus.MWCB_LEVEL2:
            halt_duration = self.MWCB_THRESHOLDS["level2"]["halt_minutes"]
        elif status == CircuitBreakerStatus.MWCB_LEVEL3:
            halt_duration = self.MWCB_THRESHOLDS["level3"]["halt_minutes"]

        state = ExchangeCircuitBreakerState(
            exchange_mic=exchange_mic,
            status=status,
            triggered_at=now if status != CircuitBreakerStatus.NORMAL else None,
            expected_resume=expected_resume,
            trigger_reason=trigger_reason,
            affected_symbols=affected_symbols or [],
            halt_duration_minutes=halt_duration,
            last_updated=now
        )

        # Record state change
        old_state = self._exchange_states.get(exchange_mic)
        if old_state and old_state.status != status:
            self._halt_history.append({
                "exchange_mic": exchange_mic,
                "old_status": old_state.status.value,
                "new_status": status.value,
                "timestamp": now.isoformat(),
                "trigger_reason": trigger_reason,
            })

            # Alert on non-normal status
            if status != CircuitBreakerStatus.NORMAL:
                logger.warning(
                    f"Circuit breaker triggered: {exchange_mic} -> {status.value} "
                    f"({trigger_reason})"
                )
                for callback in self._alert_callbacks:
                    try:
                        callback(state)
                    except Exception as e:
                        logger.error(f"Error in circuit breaker callback: {e}")

        self._exchange_states[exchange_mic] = state
        return state

    def check_exchange_circuit_breaker_status(
        self,
        exchange_mic: str
    ) -> ExchangeCircuitBreakerState:
        """
        Check current circuit breaker status for an exchange (REG-P1-2).

        Args:
            exchange_mic: Market Identifier Code

        Returns:
            Current ExchangeCircuitBreakerState
        """
        if exchange_mic not in self._exchange_states:
            # Initialize with normal status if not tracked
            self._exchange_states[exchange_mic] = ExchangeCircuitBreakerState(
                exchange_mic=exchange_mic,
                status=CircuitBreakerStatus.NORMAL
            )

        state = self._exchange_states[exchange_mic]

        # Check if halt should have expired
        if (state.status != CircuitBreakerStatus.NORMAL and
                state.expected_resume and
                datetime.now(timezone.utc) > state.expected_resume):
            # Auto-resume to normal (in production, would verify with exchange)
            logger.info(f"Circuit breaker {exchange_mic} auto-resuming to normal")
            state.status = CircuitBreakerStatus.NORMAL
            state.last_updated = datetime.now(timezone.utc)

        return state

    def is_trading_allowed(self, exchange_mic: str, symbol: str = "") -> tuple[bool, str]:
        """
        Check if trading is allowed given circuit breaker status (REG-P1-2).

        Args:
            exchange_mic: Exchange MIC
            symbol: Optional symbol to check for individual halts

        Returns:
            (is_allowed, reason)
        """
        # Check exchange-wide status
        state = self.check_exchange_circuit_breaker_status(exchange_mic)

        if state.status == CircuitBreakerStatus.MWCB_LEVEL3:
            return False, "Market closed due to Level 3 circuit breaker"

        if state.status in [
            CircuitBreakerStatus.MWCB_LEVEL1,
            CircuitBreakerStatus.MWCB_LEVEL2,
            CircuitBreakerStatus.HALT
        ]:
            return False, f"Trading halted: {state.status.value} - {state.trigger_reason}"

        # Check symbol-specific halt
        if symbol and symbol in self._symbol_halts:
            symbol_state = self._symbol_halts[symbol]
            if symbol_state.status != CircuitBreakerStatus.NORMAL:
                return False, f"Symbol {symbol} halted: {symbol_state.trigger_reason}"

        return True, "Trading allowed"

    def update_symbol_halt(
        self,
        symbol: str,
        exchange_mic: str,
        is_halted: bool,
        reason: str = "",
        expected_resume: Optional[datetime] = None
    ) -> None:
        """Update trading halt status for a specific symbol."""
        status = CircuitBreakerStatus.HALT if is_halted else CircuitBreakerStatus.NORMAL

        self._symbol_halts[symbol] = ExchangeCircuitBreakerState(
            exchange_mic=exchange_mic,
            status=status,
            triggered_at=datetime.now(timezone.utc) if is_halted else None,
            expected_resume=expected_resume,
            trigger_reason=reason,
            affected_symbols=[symbol]
        )

        if is_halted:
            logger.warning(f"Symbol {symbol} halted on {exchange_mic}: {reason}")

    def get_all_halted_symbols(self) -> List[str]:
        """Get list of all currently halted symbols."""
        return [
            symbol for symbol, state in self._symbol_halts.items()
            if state.status != CircuitBreakerStatus.NORMAL
        ]

    def get_halt_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get circuit breaker history for the last N hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [
            h for h in self._halt_history
            if datetime.fromisoformat(h["timestamp"]) > cutoff
        ]

    def get_circuit_breaker_summary(self) -> Dict[str, Any]:
        """Get summary of all circuit breaker states."""
        halted_exchanges = [
            mic for mic, state in self._exchange_states.items()
            if state.status != CircuitBreakerStatus.NORMAL
        ]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "exchanges_tracked": len(self._exchange_states),
            "exchanges_halted": len(halted_exchanges),
            "halted_exchanges": halted_exchanges,
            "symbols_halted": len(self.get_all_halted_symbols()),
            "recent_halts_24h": len(self.get_halt_history(24)),
            "exchange_states": {
                mic: state.status.value
                for mic, state in self._exchange_states.items()
            }
        }


# =============================================================================
# CFTC POSITION LIMITS (REG-P1-3)
# =============================================================================

@dataclass
class CFTCPositionLimit:
    """CFTC position limit for a commodity (REG-P1-3)."""
    commodity_code: str
    commodity_name: str
    spot_month_limit: int  # Contracts
    single_month_limit: int  # Contracts
    all_months_limit: int  # Contracts
    exchange: str
    effective_date: datetime
    is_federal_limit: bool = True  # True = federal, False = exchange-set


@dataclass
class PositionLimitCheck:
    """Result of position limit check (REG-P1-3)."""
    commodity_code: str
    current_position: int
    spot_month_position: int
    limit_type: str  # "spot_month", "single_month", "all_months"
    limit_value: int
    utilization_pct: float
    is_within_limit: bool
    headroom: int  # Contracts remaining before limit
    warning_threshold_pct: float = 80.0
    is_warning: bool = False


class CFTCPositionLimitMonitor:
    """
    Monitors CFTC position limits for commodities (REG-P1-3).

    Checks position limits in real-time before each order to ensure
    compliance with CFTC federal and exchange-set position limits.
    """

    # Default CFTC federal position limits (as of 2024)
    # These are illustrative - actual limits change and should be loaded from config
    DEFAULT_LIMITS = {
        "CL": CFTCPositionLimit(
            commodity_code="CL",
            commodity_name="Crude Oil (WTI)",
            spot_month_limit=6000,
            single_month_limit=15000,
            all_months_limit=25000,
            exchange="NYMEX",
            effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc)
        ),
        "NG": CFTCPositionLimit(
            commodity_code="NG",
            commodity_name="Natural Gas",
            spot_month_limit=1000,
            single_month_limit=4000,
            all_months_limit=8000,
            exchange="NYMEX",
            effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc)
        ),
        "GC": CFTCPositionLimit(
            commodity_code="GC",
            commodity_name="Gold",
            spot_month_limit=3000,
            single_month_limit=6000,
            all_months_limit=12000,
            exchange="COMEX",
            effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc)
        ),
        "SI": CFTCPositionLimit(
            commodity_code="SI",
            commodity_name="Silver",
            spot_month_limit=1500,
            single_month_limit=6000,
            all_months_limit=12000,
            exchange="COMEX",
            effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc)
        ),
        "ZC": CFTCPositionLimit(
            commodity_code="ZC",
            commodity_name="Corn",
            spot_month_limit=600,
            single_month_limit=33000,
            all_months_limit=33000,
            exchange="CBOT",
            effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc)
        ),
        "ZW": CFTCPositionLimit(
            commodity_code="ZW",
            commodity_name="Wheat",
            spot_month_limit=600,
            single_month_limit=12000,
            all_months_limit=12000,
            exchange="CBOT",
            effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc)
        ),
        "ZS": CFTCPositionLimit(
            commodity_code="ZS",
            commodity_name="Soybeans",
            spot_month_limit=600,
            single_month_limit=15000,
            all_months_limit=15000,
            exchange="CBOT",
            effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc)
        ),
    }

    def __init__(self, warning_threshold_pct: float = 80.0):
        self._limits: Dict[str, CFTCPositionLimit] = dict(self.DEFAULT_LIMITS)
        self._positions: Dict[str, Dict[str, int]] = {}  # commodity -> {month: position}
        self._warning_threshold = warning_threshold_pct
        self._limit_breaches: List[Dict[str, Any]] = []

    def set_position_limit(self, limit: CFTCPositionLimit) -> None:
        """Set or update position limit for a commodity."""
        self._limits[limit.commodity_code] = limit
        logger.info(f"Set CFTC position limit for {limit.commodity_code}: "
                    f"spot={limit.spot_month_limit}, all={limit.all_months_limit}")

    def update_position(
        self,
        commodity_code: str,
        contract_month: str,
        position: int
    ) -> None:
        """
        Update current position for a commodity/month.

        Args:
            commodity_code: e.g., "CL", "NG", "GC"
            contract_month: e.g., "2024-03", "2024-04"
            position: Net position in contracts (positive = long)
        """
        if commodity_code not in self._positions:
            self._positions[commodity_code] = {}
        self._positions[commodity_code][contract_month] = position

    def check_position_limits(
        self,
        commodity_code: str,
        is_spot_month: bool = False
    ) -> List[PositionLimitCheck]:
        """
        Check all position limits for a commodity (REG-P1-3).

        Args:
            commodity_code: Commodity code to check
            is_spot_month: Whether we're in the spot month

        Returns:
            List of PositionLimitCheck results
        """
        checks = []
        limit = self._limits.get(commodity_code)
        if not limit:
            logger.warning(f"No CFTC limit defined for {commodity_code}")
            return checks

        positions = self._positions.get(commodity_code, {})
        total_position = sum(abs(p) for p in positions.values())

        # Get spot month position
        spot_position = 0
        if positions:
            # In production, determine which month is spot month
            spot_position = abs(list(positions.values())[0]) if is_spot_month else 0

        # Check spot month limit
        if is_spot_month:
            spot_util = (spot_position / limit.spot_month_limit * 100) if limit.spot_month_limit > 0 else 0
            checks.append(PositionLimitCheck(
                commodity_code=commodity_code,
                current_position=total_position,
                spot_month_position=spot_position,
                limit_type="spot_month",
                limit_value=limit.spot_month_limit,
                utilization_pct=spot_util,
                is_within_limit=spot_position <= limit.spot_month_limit,
                headroom=limit.spot_month_limit - spot_position,
                warning_threshold_pct=self._warning_threshold,
                is_warning=spot_util >= self._warning_threshold
            ))

        # Check all-months limit
        all_util = (total_position / limit.all_months_limit * 100) if limit.all_months_limit > 0 else 0
        checks.append(PositionLimitCheck(
            commodity_code=commodity_code,
            current_position=total_position,
            spot_month_position=spot_position,
            limit_type="all_months",
            limit_value=limit.all_months_limit,
            utilization_pct=all_util,
            is_within_limit=total_position <= limit.all_months_limit,
            headroom=limit.all_months_limit - total_position,
            warning_threshold_pct=self._warning_threshold,
            is_warning=all_util >= self._warning_threshold
        ))

        return checks

    def validate_order_against_limits(
        self,
        commodity_code: str,
        order_quantity: int,
        is_spot_month: bool = False
    ) -> tuple[bool, str, List[PositionLimitCheck]]:
        """
        Validate an order against CFTC position limits before submission (REG-P1-3).

        This should be called before EVERY order in a commodity to ensure
        real-time compliance.

        Args:
            commodity_code: Commodity code
            order_quantity: Contracts to buy (+) or sell (-)
            is_spot_month: Whether order is in spot month

        Returns:
            (is_allowed, reason, list of limit checks)
        """
        limit = self._limits.get(commodity_code)
        if not limit:
            return True, f"No CFTC limit defined for {commodity_code}", []

        positions = self._positions.get(commodity_code, {})
        current_total = sum(abs(p) for p in positions.values())
        new_total = current_total + abs(order_quantity)

        # Check all-months limit
        if new_total > limit.all_months_limit:
            breach = {
                "commodity": commodity_code,
                "limit_type": "all_months",
                "current": current_total,
                "order_qty": order_quantity,
                "would_be": new_total,
                "limit": limit.all_months_limit,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self._limit_breaches.append(breach)
            logger.error(f"CFTC limit breach: {commodity_code} all-months "
                         f"({new_total} > {limit.all_months_limit})")
            return False, f"Order would exceed CFTC all-months limit: {new_total} > {limit.all_months_limit}", []

        # Check spot month limit if applicable
        if is_spot_month:
            current_spot = abs(list(positions.values())[0]) if positions else 0
            new_spot = current_spot + abs(order_quantity)
            if new_spot > limit.spot_month_limit:
                breach = {
                    "commodity": commodity_code,
                    "limit_type": "spot_month",
                    "current": current_spot,
                    "order_qty": order_quantity,
                    "would_be": new_spot,
                    "limit": limit.spot_month_limit,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                self._limit_breaches.append(breach)
                logger.error(f"CFTC limit breach: {commodity_code} spot-month "
                             f"({new_spot} > {limit.spot_month_limit})")
                return False, f"Order would exceed CFTC spot-month limit: {new_spot} > {limit.spot_month_limit}", []

        # Run full check for reporting
        checks = self.check_position_limits(commodity_code, is_spot_month)

        # Log warnings
        for check in checks:
            if check.is_warning:
                logger.warning(
                    f"CFTC position warning: {commodity_code} {check.limit_type} "
                    f"at {check.utilization_pct:.1f}% utilization"
                )

        return True, "Order within CFTC position limits", checks

    def get_position_limit_summary(self) -> Dict[str, Any]:
        """Get summary of all position limits and utilizations."""
        summaries = {}

        for code, limit in self._limits.items():
            positions = self._positions.get(code, {})
            total_pos = sum(abs(p) for p in positions.values())
            util = (total_pos / limit.all_months_limit * 100) if limit.all_months_limit > 0 else 0

            summaries[code] = {
                "commodity_name": limit.commodity_name,
                "exchange": limit.exchange,
                "current_position": total_pos,
                "all_months_limit": limit.all_months_limit,
                "spot_month_limit": limit.spot_month_limit,
                "utilization_pct": util,
                "is_warning": util >= self._warning_threshold,
                "headroom": limit.all_months_limit - total_pos
            }

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "commodities_tracked": len(self._limits),
            "commodities_with_positions": len(self._positions),
            "limit_breaches_total": len(self._limit_breaches),
            "summaries": summaries
        }

    def get_breach_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get position limit breach history."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [
            b for b in self._limit_breaches
            if datetime.fromisoformat(b["timestamp"]) > cutoff
        ]


class RegulatoryDomain(Enum):
    """Regulatory domains/categories."""
    MARKET_ABUSE = "market_abuse"
    TRANSACTION_REPORTING = "transaction_reporting"
    BEST_EXECUTION = "best_execution"
    POSITION_LIMITS = "position_limits"
    CAPITAL_REQUIREMENTS = "capital_requirements"
    ALGORITHMIC_TRADING = "algorithmic_trading"
    DATA_PROTECTION = "data_protection"
    AML_KYC = "aml_kyc"
    RECORD_KEEPING = "record_keeping"
    CLIENT_PROTECTION = "client_protection"


class ChangeImpact(Enum):
    """Impact level of regulatory changes."""
    CRITICAL = "critical"  # Requires immediate action
    HIGH = "high"  # Requires action within 30 days
    MEDIUM = "medium"  # Requires action within 90 days
    LOW = "low"  # Informational, no immediate action
    INFORMATIONAL = "informational"  # No action required


class ChangeStatus(Enum):
    """Status of regulatory change tracking."""
    PENDING_REVIEW = "pending_review"
    UNDER_REVIEW = "under_review"
    IMPACT_ASSESSED = "impact_assessed"
    IMPLEMENTATION_PLANNED = "implementation_planned"
    IMPLEMENTING = "implementing"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"


@dataclass
class RegulatoryChange:
    """Represents a regulatory change or update."""
    change_id: str
    title: str
    regulator: RegulatorType
    domain: RegulatoryDomain
    description: str
    effective_date: datetime
    publication_date: datetime
    impact: ChangeImpact
    status: ChangeStatus = ChangeStatus.PENDING_REVIEW
    source_url: str = ""
    affected_modules: List[str] = field(default_factory=list)
    implementation_notes: str = ""
    assigned_to: str = ""
    review_deadline: Optional[datetime] = None
    implementation_deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "change_id": self.change_id,
            "title": self.title,
            "regulator": self.regulator.value,
            "domain": self.domain.value,
            "description": self.description,
            "effective_date": self.effective_date.isoformat(),
            "publication_date": self.publication_date.isoformat(),
            "impact": self.impact.value,
            "status": self.status.value,
            "source_url": self.source_url,
            "affected_modules": self.affected_modules,
            "implementation_notes": self.implementation_notes,
            "assigned_to": self.assigned_to,
            "review_deadline": self.review_deadline.isoformat() if self.review_deadline else None,
            "implementation_deadline": self.implementation_deadline.isoformat() if self.implementation_deadline else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def days_until_effective(self) -> int:
        """Calculate days until effective date."""
        delta = self.effective_date - datetime.now(timezone.utc)
        return max(0, delta.days)

    def is_overdue(self) -> bool:
        """Check if implementation is overdue."""
        if self.status == ChangeStatus.VERIFIED:
            return False
        if self.implementation_deadline:
            return datetime.now(timezone.utc) > self.implementation_deadline
        return datetime.now(timezone.utc) > self.effective_date


@dataclass
class RegulatoryFeed:
    """Configuration for a regulatory feed source."""
    feed_id: str
    regulator: RegulatorType
    feed_url: str
    feed_type: str  # rss, api, scrape
    check_interval_hours: int = 24
    last_checked: Optional[datetime] = None
    enabled: bool = True
    auth_required: bool = False
    auth_config: Dict[str, str] = field(default_factory=dict)


class RegulatoryChangeMonitor:
    """
    Monitors regulatory changes from various sources.

    Provides automated tracking and alerting for regulatory updates
    that may affect trading operations.
    """

    def __init__(self):
        self.feeds: Dict[str, RegulatoryFeed] = {}
        self.changes: Dict[str, RegulatoryChange] = {}
        self.subscribers: Dict[RegulatoryDomain, List[Callable]] = {}
        self.seen_hashes: Set[str] = set()
        self._setup_default_feeds()

    def _setup_default_feeds(self):
        """Configure default regulatory feed sources."""
        default_feeds = [
            RegulatoryFeed(
                feed_id="esma_news",
                regulator=RegulatorType.ESMA,
                feed_url="https://www.esma.europa.eu/press-news/esma-news/feed",
                feed_type="rss",
                check_interval_hours=12
            ),
            RegulatoryFeed(
                feed_id="amf_news",
                regulator=RegulatorType.AMF,
                feed_url="https://www.amf-france.org/en/news-publications/news-releases/feed",
                feed_type="rss",
                check_interval_hours=12
            ),
            RegulatoryFeed(
                feed_id="fca_news",
                regulator=RegulatorType.FCA,
                feed_url="https://www.fca.org.uk/news/rss.xml",
                feed_type="rss",
                check_interval_hours=12
            )
        ]
        for feed in default_feeds:
            self.feeds[feed.feed_id] = feed

    def add_feed(self, feed: RegulatoryFeed):
        """Add a regulatory feed source."""
        self.feeds[feed.feed_id] = feed
        logger.info(f"Added regulatory feed: {feed.feed_id}")

    def subscribe(self, domain: RegulatoryDomain, callback: Callable[[RegulatoryChange], None]):
        """Subscribe to changes in a regulatory domain."""
        if domain not in self.subscribers:
            self.subscribers[domain] = []
        self.subscribers[domain].append(callback)

    def _generate_change_id(self, title: str, regulator: str, date: datetime) -> str:
        """Generate unique change ID."""
        content = f"{title}:{regulator}:{date.isoformat()}"
        hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"REG-{hash_val.upper()}"

    def _classify_domain(self, title: str, description: str) -> RegulatoryDomain:
        """Classify regulatory change into a domain based on content."""
        text = f"{title} {description}".lower()

        domain_keywords = {
            RegulatoryDomain.MARKET_ABUSE: ["mar", "market abuse", "insider", "manipulation"],
            RegulatoryDomain.TRANSACTION_REPORTING: ["emir", "reporting", "transaction report", "trade repository"],
            RegulatoryDomain.BEST_EXECUTION: ["best execution", "rts 27", "rts 28", "execution quality"],
            RegulatoryDomain.POSITION_LIMITS: ["position limit", "commodity derivative"],
            RegulatoryDomain.CAPITAL_REQUIREMENTS: ["capital", "crr", "crd", "prudential"],
            RegulatoryDomain.ALGORITHMIC_TRADING: ["algorithmic", "algo trading", "rts 6", "hft"],
            RegulatoryDomain.DATA_PROTECTION: ["gdpr", "data protection", "privacy"],
            RegulatoryDomain.AML_KYC: ["aml", "kyc", "money laundering", "terrorist financing"],
            RegulatoryDomain.RECORD_KEEPING: ["record keeping", "retention", "rts 25"],
            RegulatoryDomain.CLIENT_PROTECTION: ["client", "investor protection", "suitability"]
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in text for kw in keywords):
                return domain

        return RegulatoryDomain.TRANSACTION_REPORTING  # Default

    def _assess_impact(self, change: RegulatoryChange) -> ChangeImpact:
        """Assess the impact level of a regulatory change."""
        days_until = change.days_until_effective()

        # Critical if already effective or within 7 days
        if days_until <= 7:
            return ChangeImpact.CRITICAL

        # High if within 30 days
        if days_until <= 30:
            return ChangeImpact.HIGH

        # Medium if within 90 days
        if days_until <= 90:
            return ChangeImpact.MEDIUM

        # Otherwise low
        return ChangeImpact.LOW

    def register_change(self, change: RegulatoryChange) -> bool:
        """
        Register a new regulatory change.

        Returns True if this is a new change, False if duplicate.
        """
        # Check for duplicates
        content_hash = hashlib.md5(
            f"{change.title}:{change.regulator.value}:{change.publication_date.isoformat()}".encode()
        ).hexdigest()

        if content_hash in self.seen_hashes:
            return False

        self.seen_hashes.add(content_hash)

        # Auto-assess impact if not set
        if change.impact == ChangeImpact.INFORMATIONAL:
            change.impact = self._assess_impact(change)

        # Set review deadline based on impact
        if not change.review_deadline:
            review_days = {
                ChangeImpact.CRITICAL: 1,
                ChangeImpact.HIGH: 7,
                ChangeImpact.MEDIUM: 14,
                ChangeImpact.LOW: 30,
                ChangeImpact.INFORMATIONAL: 60
            }
            change.review_deadline = datetime.now(timezone.utc) + timedelta(
                days=review_days.get(change.impact, 30)
            )

        self.changes[change.change_id] = change

        # Notify subscribers
        if change.domain in self.subscribers:
            for callback in self.subscribers[change.domain]:
                try:
                    callback(change)
                except Exception as e:
                    logger.error(f"Error notifying subscriber: {e}")

        logger.info(f"Registered regulatory change: {change.change_id} - {change.title}")
        return True

    def update_status(self, change_id: str, status: ChangeStatus, notes: str = ""):
        """Update the status of a regulatory change."""
        if change_id not in self.changes:
            raise ValueError(f"Change not found: {change_id}")

        change = self.changes[change_id]
        old_status = change.status
        change.status = status
        change.updated_at = datetime.now(timezone.utc)

        if notes:
            change.implementation_notes += f"\n[{datetime.now(timezone.utc).isoformat()}] {notes}"

        logger.info(f"Updated change {change_id} status: {old_status.value} -> {status.value}")

    def get_pending_changes(self,
                           impact_filter: Optional[ChangeImpact] = None,
                           domain_filter: Optional[RegulatoryDomain] = None) -> List[RegulatoryChange]:
        """Get list of pending regulatory changes."""
        changes = []

        for change in self.changes.values():
            if change.status in [ChangeStatus.VERIFIED, ChangeStatus.IMPLEMENTED]:
                continue

            if impact_filter and change.impact != impact_filter:
                continue

            if domain_filter and change.domain != domain_filter:
                continue

            changes.append(change)

        # Sort by effective date
        changes.sort(key=lambda c: c.effective_date)
        return changes

    def get_overdue_changes(self) -> List[RegulatoryChange]:
        """Get list of overdue regulatory changes."""
        return [c for c in self.changes.values() if c.is_overdue()]

    def get_change_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked regulatory changes."""
        by_status = {}
        by_impact = {}
        by_domain = {}

        for change in self.changes.values():
            by_status[change.status.value] = by_status.get(change.status.value, 0) + 1
            by_impact[change.impact.value] = by_impact.get(change.impact.value, 0) + 1
            by_domain[change.domain.value] = by_domain.get(change.domain.value, 0) + 1

        overdue = self.get_overdue_changes()

        return {
            "total_changes": len(self.changes),
            "by_status": by_status,
            "by_impact": by_impact,
            "by_domain": by_domain,
            "overdue_count": len(overdue),
            "overdue_changes": [c.change_id for c in overdue],
            "feeds_configured": len(self.feeds),
            "active_feeds": sum(1 for f in self.feeds.values() if f.enabled)
        }


# ============================================================================
# Compliance Report Templates
# ============================================================================

class ReportFormat(Enum):
    """Report output formats."""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    EXCEL = "excel"
    XML = "xml"


@dataclass
class ReportSection:
    """Section within a compliance report."""
    section_id: str
    title: str
    content: str
    data: Dict[str, Any] = field(default_factory=dict)
    subsections: List['ReportSection'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "section_id": self.section_id,
            "title": self.title,
            "content": self.content,
            "data": self.data,
            "subsections": [s.to_dict() for s in self.subsections]
        }


@dataclass
class ComplianceReportTemplate:
    """Template for compliance reports."""
    template_id: str
    name: str
    description: str
    regulation: str  # e.g., "MiFID II", "EMIR", "MAR"
    report_type: str  # e.g., "periodic", "ad-hoc", "regulatory"
    frequency: str  # e.g., "daily", "weekly", "monthly", "quarterly", "annual"
    sections: List[str] = field(default_factory=list)
    required_data: List[str] = field(default_factory=list)
    output_formats: List[ReportFormat] = field(default_factory=list)
    version: str = "1.0"
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "regulation": self.regulation,
            "report_type": self.report_type,
            "frequency": self.frequency,
            "sections": self.sections,
            "required_data": self.required_data,
            "output_formats": [f.value for f in self.output_formats],
            "version": self.version,
            "last_updated": self.last_updated.isoformat()
        }


class ComplianceReportGenerator:
    """
    Generates compliance reports using standardized templates.

    Provides templates for various EU regulatory requirements including
    MiFID II, EMIR, MAR, and internal compliance reporting.
    """

    def __init__(self):
        self.templates: Dict[str, ComplianceReportTemplate] = {}
        self._setup_default_templates()

    def _setup_default_templates(self):
        """Set up default compliance report templates."""
        templates = [
            # MiFID II RTS 27 Best Execution Report
            ComplianceReportTemplate(
                template_id="rts27_best_execution",
                name="RTS 27 Best Execution Report",
                description="Quarterly report on execution quality per MiFID II RTS 27",
                regulation="MiFID II",
                report_type="regulatory",
                frequency="quarterly",
                sections=[
                    "executive_summary",
                    "venue_statistics",
                    "price_quality",
                    "speed_execution",
                    "likelihood_execution",
                    "settlement_quality",
                    "transaction_costs"
                ],
                required_data=[
                    "executions_by_venue",
                    "execution_times",
                    "price_improvements",
                    "failed_settlements",
                    "transaction_costs"
                ],
                output_formats=[ReportFormat.PDF, ReportFormat.XML, ReportFormat.EXCEL]
            ),

            # MiFID II RTS 28 Venue Analysis Report
            ComplianceReportTemplate(
                template_id="rts28_venue_analysis",
                name="RTS 28 Top 5 Venues Report",
                description="Annual report on top execution venues per MiFID II RTS 28",
                regulation="MiFID II",
                report_type="regulatory",
                frequency="annual",
                sections=[
                    "methodology",
                    "top_venues_by_class",
                    "order_routing_decisions",
                    "conflicts_of_interest",
                    "passive_aggressive_analysis"
                ],
                required_data=[
                    "venue_volumes",
                    "order_routing_policies",
                    "inducements",
                    "fill_types"
                ],
                output_formats=[ReportFormat.PDF, ReportFormat.HTML]
            ),

            # EMIR Trade Reporting
            ComplianceReportTemplate(
                template_id="emir_trade_report",
                name="EMIR Trade Report",
                description="Daily transaction report for EMIR compliance",
                regulation="EMIR",
                report_type="regulatory",
                frequency="daily",
                sections=[
                    "reporting_summary",
                    "counterparty_details",
                    "trade_details",
                    "valuation_data",
                    "collateral_data"
                ],
                required_data=[
                    "trades",
                    "counterparties",
                    "valuations",
                    "collateral_movements"
                ],
                output_formats=[ReportFormat.XML]
            ),

            # MAR Suspicious Transaction Report
            ComplianceReportTemplate(
                template_id="mar_stor",
                name="Suspicious Transaction and Order Report (STOR)",
                description="Report for suspicious market activity per MAR Article 16",
                regulation="MAR",
                report_type="ad-hoc",
                frequency="as_needed",
                sections=[
                    "incident_summary",
                    "transaction_details",
                    "suspicion_indicators",
                    "supporting_evidence",
                    "risk_assessment"
                ],
                required_data=[
                    "suspicious_orders",
                    "market_data",
                    "detection_alerts",
                    "historical_patterns"
                ],
                output_formats=[ReportFormat.PDF, ReportFormat.XML]
            ),

            # Daily Compliance Summary
            ComplianceReportTemplate(
                template_id="daily_compliance_summary",
                name="Daily Compliance Summary",
                description="Internal daily compliance status report",
                regulation="Internal",
                report_type="periodic",
                frequency="daily",
                sections=[
                    "trading_activity",
                    "limit_utilization",
                    "breach_summary",
                    "pending_actions",
                    "upcoming_deadlines"
                ],
                required_data=[
                    "trades",
                    "positions",
                    "risk_limits",
                    "alerts"
                ],
                output_formats=[ReportFormat.HTML, ReportFormat.JSON]
            ),

            # Monthly Regulatory Report
            ComplianceReportTemplate(
                template_id="monthly_regulatory_summary",
                name="Monthly Regulatory Summary",
                description="Monthly summary of regulatory compliance status",
                regulation="Multi-regulation",
                report_type="periodic",
                frequency="monthly",
                sections=[
                    "executive_summary",
                    "mifid_compliance",
                    "emir_compliance",
                    "mar_compliance",
                    "incident_summary",
                    "action_items"
                ],
                required_data=[
                    "reporting_status",
                    "breaches",
                    "incidents",
                    "remediation_status"
                ],
                output_formats=[ReportFormat.PDF, ReportFormat.HTML]
            ),

            # Risk Report
            ComplianceReportTemplate(
                template_id="daily_risk_report",
                name="Daily Risk Report",
                description="Daily risk metrics and limit utilization report",
                regulation="Internal",
                report_type="periodic",
                frequency="daily",
                sections=[
                    "portfolio_summary",
                    "var_metrics",
                    "limit_utilization",
                    "stress_test_results",
                    "concentration_analysis"
                ],
                required_data=[
                    "positions",
                    "var_calculations",
                    "risk_limits",
                    "stress_scenarios"
                ],
                output_formats=[ReportFormat.HTML, ReportFormat.PDF, ReportFormat.JSON]
            ),

            # Algorithmic Trading Report (RTS 6)
            ComplianceReportTemplate(
                template_id="rts6_algo_trading",
                name="RTS 6 Algorithmic Trading Report",
                description="Report on algorithmic trading systems per MiFID II RTS 6",
                regulation="MiFID II",
                report_type="regulatory",
                frequency="annual",
                sections=[
                    "system_overview",
                    "algorithm_inventory",
                    "kill_switch_tests",
                    "pre_trade_controls",
                    "incident_log"
                ],
                required_data=[
                    "algorithms",
                    "system_tests",
                    "control_parameters",
                    "incidents"
                ],
                output_formats=[ReportFormat.PDF]
            )
        ]

        for template in templates:
            self.templates[template.template_id] = template

    def get_template(self, template_id: str) -> Optional[ComplianceReportTemplate]:
        """Get a report template by ID."""
        return self.templates.get(template_id)

    def list_templates(self,
                       regulation_filter: Optional[str] = None,
                       frequency_filter: Optional[str] = None) -> List[ComplianceReportTemplate]:
        """List available report templates."""
        templates = list(self.templates.values())

        if regulation_filter:
            templates = [t for t in templates if t.regulation == regulation_filter]

        if frequency_filter:
            templates = [t for t in templates if t.frequency == frequency_filter]

        return templates

    def generate_report(self,
                       template_id: str,
                       data: Dict[str, Any],
                       output_format: ReportFormat = ReportFormat.JSON,
                       report_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate a compliance report from a template.

        Args:
            template_id: ID of the template to use
            data: Data to populate the report
            output_format: Desired output format
            report_date: Date for the report (defaults to today)

        Returns:
            Generated report content
        """
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")

        if output_format not in template.output_formats:
            raise ValueError(f"Format {output_format.value} not supported for template {template_id}")

        report_date = report_date or datetime.now(timezone.utc)

        # Validate required data
        missing_data = [d for d in template.required_data if d not in data]
        if missing_data:
            logger.warning(f"Missing required data for report: {missing_data}")

        # Build report structure
        report = {
            "report_id": f"{template_id}_{report_date.strftime('%Y%m%d_%H%M%S')}",
            "template_id": template_id,
            "template_name": template.name,
            "template_version": template.version,
            "regulation": template.regulation,
            "report_type": template.report_type,
            "report_date": report_date.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sections": {}
        }

        # Populate sections
        for section in template.sections:
            section_data = self._build_section(section, data)
            report["sections"][section] = section_data

        # Add metadata
        report["metadata"] = {
            "data_completeness": len([d for d in template.required_data if d in data]) / len(template.required_data) if template.required_data else 1.0,
            "missing_data": missing_data,
            "output_format": output_format.value
        }

        logger.info(f"Generated report: {report['report_id']}")
        return report

    def _build_section(self, section_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build a report section with available data."""
        # Map section names to data extraction logic
        section_builders = {
            "executive_summary": self._build_executive_summary,
            "trading_activity": self._build_trading_activity,
            "limit_utilization": self._build_limit_utilization,
            "breach_summary": self._build_breach_summary,
            "var_metrics": self._build_var_metrics,
            "venue_statistics": self._build_venue_statistics
        }

        builder = section_builders.get(section_name, self._build_generic_section)
        return builder(section_name, data)

    def _build_executive_summary(self, section_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build executive summary section."""
        return {
            "title": "Executive Summary",
            "content": "Summary of key compliance metrics and status.",
            "data": {
                "total_trades": data.get("trades", {}).get("count", 0),
                "total_volume": data.get("trades", {}).get("volume", 0),
                "breaches": data.get("breaches", {}).get("count", 0),
                "alerts": data.get("alerts", {}).get("count", 0)
            }
        }

    def _build_trading_activity(self, section_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build trading activity section."""
        trades = data.get("trades", {})
        return {
            "title": "Trading Activity",
            "content": "Summary of trading activity during the reporting period.",
            "data": {
                "total_orders": trades.get("order_count", 0),
                "executed_orders": trades.get("executed_count", 0),
                "cancelled_orders": trades.get("cancelled_count", 0),
                "total_volume": trades.get("volume", 0),
                "by_instrument_type": trades.get("by_type", {}),
                "by_venue": trades.get("by_venue", {})
            }
        }

    def _build_limit_utilization(self, section_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build limit utilization section."""
        limits = data.get("risk_limits", {})
        return {
            "title": "Limit Utilization",
            "content": "Current utilization of risk and position limits.",
            "data": {
                "var_utilization": limits.get("var_utilization", 0),
                "position_utilization": limits.get("position_utilization", 0),
                "leverage_utilization": limits.get("leverage_utilization", 0),
                "concentration_utilization": limits.get("concentration_utilization", 0)
            }
        }

    def _build_breach_summary(self, section_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build breach summary section."""
        breaches = data.get("breaches", {})
        return {
            "title": "Breach Summary",
            "content": "Summary of limit breaches and violations.",
            "data": {
                "total_breaches": breaches.get("count", 0),
                "by_type": breaches.get("by_type", {}),
                "resolved": breaches.get("resolved", 0),
                "pending": breaches.get("pending", 0)
            }
        }

    def _build_var_metrics(self, section_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build VaR metrics section."""
        var_data = data.get("var_calculations", {})
        return {
            "title": "Value at Risk Metrics",
            "content": "Daily VaR and risk metrics.",
            "data": {
                "var_95": var_data.get("var_95", 0),
                "var_99": var_data.get("var_99", 0),
                "cvar_95": var_data.get("cvar_95", 0),
                "cvar_99": var_data.get("cvar_99", 0)
            }
        }

    def _build_venue_statistics(self, section_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build venue statistics section."""
        venues = data.get("executions_by_venue", {})
        return {
            "title": "Venue Statistics",
            "content": "Execution statistics by trading venue.",
            "data": {
                "venues": venues,
                "total_venues": len(venues)
            }
        }

    def _build_generic_section(self, section_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build a generic section."""
        return {
            "title": section_name.replace("_", " ").title(),
            "content": f"Data for {section_name}",
            "data": data.get(section_name, {})
        }

    def export_template_catalog(self) -> str:
        """Export catalog of all templates as JSON."""
        catalog = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "template_count": len(self.templates),
            "templates": [t.to_dict() for t in self.templates.values()]
        }
        return json.dumps(catalog, indent=2)


# ============================================================================
# Combined Regulatory Monitoring System
# ============================================================================

# =============================================================================
# REGULATORY CHANGE TRACKING (P3)
# =============================================================================

class ChangeTrackingStatus(Enum):
    """Status of regulatory change tracking (P3)."""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    UNDER_ANALYSIS = "under_analysis"
    IMPACT_ASSESSED = "impact_assessed"
    ACTION_REQUIRED = "action_required"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEFERRED = "deferred"


@dataclass
class RegulatoryChangeTracking:
    """Detailed tracking of a regulatory change (P3)."""
    change_id: str
    title: str
    regulator: RegulatorType
    domain: RegulatoryDomain
    effective_date: datetime
    tracking_status: ChangeTrackingStatus = ChangeTrackingStatus.NEW
    assigned_team: str = ""
    assigned_individual: str = ""
    impact_assessment: str = ""
    required_actions: List[str] = field(default_factory=list)
    completed_actions: List[str] = field(default_factory=list)
    notes: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    due_date: Optional[datetime] = None

    def add_note(self, author: str, content: str) -> None:
        """Add a tracking note."""
        self.notes.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "author": author,
            "content": content,
        })
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_id": self.change_id,
            "title": self.title,
            "regulator": self.regulator.value,
            "domain": self.domain.value,
            "effective_date": self.effective_date.isoformat(),
            "tracking_status": self.tracking_status.value,
            "assigned_team": self.assigned_team,
            "assigned_individual": self.assigned_individual,
            "impact_assessment": self.impact_assessment,
            "required_actions": self.required_actions,
            "completed_actions": self.completed_actions,
            "notes": self.notes,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class RegulatoryChangeTracker:
    """
    Tracks regulatory changes through their lifecycle (P3).

    Provides detailed tracking of regulatory changes from identification
    through implementation.
    """

    def __init__(self):
        self._tracking: Dict[str, RegulatoryChangeTracking] = {}

    def start_tracking(
        self,
        change: RegulatoryChange,
        assigned_team: str = "",
        assigned_individual: str = "",
    ) -> RegulatoryChangeTracking:
        """Start tracking a regulatory change (P3)."""
        tracking = RegulatoryChangeTracking(
            change_id=change.change_id,
            title=change.title,
            regulator=change.regulator,
            domain=change.domain,
            effective_date=change.effective_date,
            assigned_team=assigned_team,
            assigned_individual=assigned_individual,
            due_date=change.effective_date - timedelta(days=30),  # Default: 30 days before
        )
        self._tracking[change.change_id] = tracking
        logger.info(f"Started tracking regulatory change: {change.change_id}")
        return tracking

    def update_status(
        self,
        change_id: str,
        new_status: ChangeTrackingStatus,
        updated_by: str,
        note: str = "",
    ) -> Optional[RegulatoryChangeTracking]:
        """Update tracking status (P3)."""
        tracking = self._tracking.get(change_id)
        if not tracking:
            return None

        old_status = tracking.tracking_status
        tracking.tracking_status = new_status
        tracking.updated_at = datetime.now(timezone.utc)

        if note:
            tracking.add_note(updated_by, f"Status changed: {old_status.value} -> {new_status.value}. {note}")

        logger.info(f"Tracking status updated: {change_id} -> {new_status.value}")
        return tracking

    def set_impact_assessment(
        self,
        change_id: str,
        assessment: str,
        required_actions: List[str],
        assessed_by: str,
    ) -> Optional[RegulatoryChangeTracking]:
        """Set impact assessment for a change (P3)."""
        tracking = self._tracking.get(change_id)
        if not tracking:
            return None

        tracking.impact_assessment = assessment
        tracking.required_actions = required_actions
        tracking.tracking_status = ChangeTrackingStatus.IMPACT_ASSESSED
        tracking.add_note(assessed_by, f"Impact assessment completed. {len(required_actions)} actions identified.")

        return tracking

    def complete_action(
        self,
        change_id: str,
        action: str,
        completed_by: str,
    ) -> Optional[RegulatoryChangeTracking]:
        """Mark an action as completed (P3)."""
        tracking = self._tracking.get(change_id)
        if not tracking:
            return None

        if action in tracking.required_actions and action not in tracking.completed_actions:
            tracking.completed_actions.append(action)
            tracking.add_note(completed_by, f"Action completed: {action}")

            # Check if all actions completed
            if len(tracking.completed_actions) == len(tracking.required_actions):
                tracking.tracking_status = ChangeTrackingStatus.COMPLETED
                logger.info(f"All actions completed for change: {change_id}")

        return tracking

    def get_tracking(self, change_id: str) -> Optional[RegulatoryChangeTracking]:
        """Get tracking details for a change (P3)."""
        return self._tracking.get(change_id)

    def get_overdue_changes(self) -> List[RegulatoryChangeTracking]:
        """Get changes that are overdue (P3)."""
        now = datetime.now(timezone.utc)
        return [
            t for t in self._tracking.values()
            if t.tracking_status not in [ChangeTrackingStatus.COMPLETED, ChangeTrackingStatus.DEFERRED]
            and t.due_date and t.due_date < now
        ]

    def get_changes_by_status(self, status: ChangeTrackingStatus) -> List[RegulatoryChangeTracking]:
        """Get changes by tracking status (P3)."""
        return [t for t in self._tracking.values() if t.tracking_status == status]

    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked changes (P3)."""
        by_status = {}
        for status in ChangeTrackingStatus:
            by_status[status.value] = len(self.get_changes_by_status(status))

        overdue = self.get_overdue_changes()

        return {
            "total_tracked": len(self._tracking),
            "by_status": by_status,
            "overdue_count": len(overdue),
            "overdue_changes": [t.change_id for t in overdue],
        }


# =============================================================================
# COMPLIANCE DASHBOARD METRICS (P3)
# =============================================================================

@dataclass
class ComplianceDashboardMetric:
    """A single metric for the compliance dashboard (P3)."""
    metric_id: str
    name: str
    category: str
    value: float
    unit: str
    threshold_warning: float
    threshold_critical: float
    status: str = "normal"  # normal, warning, critical
    trend: str = "stable"  # improving, stable, degrading
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def update(self, new_value: float) -> None:
        """Update metric value and status."""
        old_value = self.value
        self.value = new_value
        self.last_updated = datetime.now(timezone.utc)

        # Update status
        if new_value >= self.threshold_critical:
            self.status = "critical"
        elif new_value >= self.threshold_warning:
            self.status = "warning"
        else:
            self.status = "normal"

        # Update trend
        if new_value < old_value:
            self.trend = "improving"
        elif new_value > old_value:
            self.trend = "degrading"
        else:
            self.trend = "stable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "category": self.category,
            "value": self.value,
            "unit": self.unit,
            "threshold_warning": self.threshold_warning,
            "threshold_critical": self.threshold_critical,
            "status": self.status,
            "trend": self.trend,
            "last_updated": self.last_updated.isoformat(),
        }


class ComplianceDashboard:
    """
    Compliance dashboard with key metrics (P3).

    Provides real-time compliance metrics for monitoring.
    """

    def __init__(self):
        self._metrics: Dict[str, ComplianceDashboardMetric] = {}
        self._setup_default_metrics()

    def _setup_default_metrics(self) -> None:
        """Setup default compliance metrics (P3)."""
        default_metrics = [
            ComplianceDashboardMetric(
                metric_id="transaction_reporting_rate",
                name="Transaction Reporting Rate",
                category="Reporting",
                value=100.0,
                unit="%",
                threshold_warning=95.0,
                threshold_critical=90.0,
            ),
            ComplianceDashboardMetric(
                metric_id="reporting_timeliness",
                name="Reporting Timeliness",
                category="Reporting",
                value=100.0,
                unit="%",
                threshold_warning=98.0,
                threshold_critical=95.0,
            ),
            ComplianceDashboardMetric(
                metric_id="position_limit_utilization",
                name="Position Limit Utilization",
                category="Risk",
                value=0.0,
                unit="%",
                threshold_warning=80.0,
                threshold_critical=95.0,
            ),
            ComplianceDashboardMetric(
                metric_id="market_abuse_alerts",
                name="Open Market Abuse Alerts",
                category="Surveillance",
                value=0.0,
                unit="alerts",
                threshold_warning=5.0,
                threshold_critical=10.0,
            ),
            ComplianceDashboardMetric(
                metric_id="regulatory_changes_pending",
                name="Pending Regulatory Changes",
                category="Regulatory",
                value=0.0,
                unit="changes",
                threshold_warning=5.0,
                threshold_critical=10.0,
            ),
            ComplianceDashboardMetric(
                metric_id="overdue_items",
                name="Overdue Compliance Items",
                category="Operations",
                value=0.0,
                unit="items",
                threshold_warning=1.0,
                threshold_critical=3.0,
            ),
            ComplianceDashboardMetric(
                metric_id="best_execution_score",
                name="Best Execution Score",
                category="Execution",
                value=100.0,
                unit="%",
                threshold_warning=95.0,
                threshold_critical=90.0,
            ),
        ]

        for metric in default_metrics:
            self._metrics[metric.metric_id] = metric

    def update_metric(self, metric_id: str, value: float) -> Optional[ComplianceDashboardMetric]:
        """Update a dashboard metric (P3)."""
        metric = self._metrics.get(metric_id)
        if metric:
            metric.update(value)
            logger.debug(f"Dashboard metric updated: {metric_id} = {value}")
        return metric

    def add_metric(self, metric: ComplianceDashboardMetric) -> None:
        """Add a custom metric to the dashboard (P3)."""
        self._metrics[metric.metric_id] = metric
        logger.info(f"Custom metric added: {metric.metric_id}")

    def get_metric(self, metric_id: str) -> Optional[ComplianceDashboardMetric]:
        """Get a specific metric (P3)."""
        return self._metrics.get(metric_id)

    def get_all_metrics(self) -> List[ComplianceDashboardMetric]:
        """Get all dashboard metrics (P3)."""
        return list(self._metrics.values())

    def get_metrics_by_category(self, category: str) -> List[ComplianceDashboardMetric]:
        """Get metrics by category (P3)."""
        return [m for m in self._metrics.values() if m.category == category]

    def get_metrics_by_status(self, status: str) -> List[ComplianceDashboardMetric]:
        """Get metrics by status (P3)."""
        return [m for m in self._metrics.values() if m.status == status]

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get dashboard summary (P3)."""
        metrics = list(self._metrics.values())
        critical = [m for m in metrics if m.status == "critical"]
        warning = [m for m in metrics if m.status == "warning"]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_metrics": len(metrics),
            "critical_count": len(critical),
            "warning_count": len(warning),
            "normal_count": len(metrics) - len(critical) - len(warning),
            "overall_status": "critical" if critical else ("warning" if warning else "normal"),
            "critical_metrics": [m.to_dict() for m in critical],
            "warning_metrics": [m.to_dict() for m in warning],
            "all_metrics": [m.to_dict() for m in metrics],
        }


# =============================================================================
# ALERT ESCALATION PATHS (P3)
# =============================================================================

class EscalationLevel(Enum):
    """Escalation levels for alerts (P3)."""
    L1_ANALYST = "l1_analyst"
    L2_SUPERVISOR = "l2_supervisor"
    L3_MANAGER = "l3_manager"
    L4_DIRECTOR = "l4_director"
    L5_EXECUTIVE = "l5_executive"


@dataclass
class EscalationContact:
    """Contact for escalation (P3)."""
    contact_id: str
    name: str
    role: str
    email: str
    phone: str
    level: EscalationLevel
    available_hours: str = "09:00-17:00"
    backup_contact_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contact_id": self.contact_id,
            "name": self.name,
            "role": self.role,
            "email": self.email,
            "phone": self.phone,
            "level": self.level.value,
            "available_hours": self.available_hours,
            "backup_contact_id": self.backup_contact_id,
        }


@dataclass
class EscalationPath:
    """Escalation path definition (P3)."""
    path_id: str
    name: str
    alert_type: str  # e.g., "market_abuse", "position_limit", "reporting_failure"
    levels: List[Dict[str, Any]]  # [{level, contacts, timeout_minutes}]
    auto_escalate: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "name": self.name,
            "alert_type": self.alert_type,
            "levels": self.levels,
            "auto_escalate": self.auto_escalate,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class EscalationInstance:
    """An active escalation instance (P3)."""
    instance_id: str
    path_id: str
    alert_id: str
    alert_type: str
    alert_details: Dict[str, Any]
    current_level: EscalationLevel
    started_at: datetime
    last_escalation_at: datetime
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    status: str = "active"  # active, acknowledged, resolved, expired

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "path_id": self.path_id,
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "alert_details": self.alert_details,
            "current_level": self.current_level.value,
            "started_at": self.started_at.isoformat(),
            "last_escalation_at": self.last_escalation_at.isoformat(),
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "status": self.status,
        }


class AlertEscalationManager:
    """
    Manages alert escalation paths (P3).

    Provides structured escalation for compliance alerts.
    """

    def __init__(self):
        self._contacts: Dict[str, EscalationContact] = {}
        self._paths: Dict[str, EscalationPath] = {}
        self._instances: Dict[str, EscalationInstance] = {}
        self._instance_counter = 0
        self._setup_default_paths()

    def _setup_default_paths(self) -> None:
        """Setup default escalation paths (P3)."""
        default_paths = [
            EscalationPath(
                path_id="market_abuse_escalation",
                name="Market Abuse Alert Escalation",
                alert_type="market_abuse",
                levels=[
                    {"level": EscalationLevel.L1_ANALYST, "timeout_minutes": 15},
                    {"level": EscalationLevel.L2_SUPERVISOR, "timeout_minutes": 30},
                    {"level": EscalationLevel.L3_MANAGER, "timeout_minutes": 60},
                    {"level": EscalationLevel.L4_DIRECTOR, "timeout_minutes": 120},
                ],
            ),
            EscalationPath(
                path_id="position_limit_escalation",
                name="Position Limit Breach Escalation",
                alert_type="position_limit",
                levels=[
                    {"level": EscalationLevel.L2_SUPERVISOR, "timeout_minutes": 5},
                    {"level": EscalationLevel.L3_MANAGER, "timeout_minutes": 15},
                    {"level": EscalationLevel.L4_DIRECTOR, "timeout_minutes": 30},
                ],
            ),
            EscalationPath(
                path_id="reporting_failure_escalation",
                name="Reporting Failure Escalation",
                alert_type="reporting_failure",
                levels=[
                    {"level": EscalationLevel.L1_ANALYST, "timeout_minutes": 10},
                    {"level": EscalationLevel.L2_SUPERVISOR, "timeout_minutes": 30},
                    {"level": EscalationLevel.L3_MANAGER, "timeout_minutes": 60},
                ],
            ),
            EscalationPath(
                path_id="circuit_breaker_escalation",
                name="Circuit Breaker Alert Escalation",
                alert_type="circuit_breaker",
                levels=[
                    {"level": EscalationLevel.L2_SUPERVISOR, "timeout_minutes": 1},
                    {"level": EscalationLevel.L3_MANAGER, "timeout_minutes": 5},
                    {"level": EscalationLevel.L4_DIRECTOR, "timeout_minutes": 15},
                    {"level": EscalationLevel.L5_EXECUTIVE, "timeout_minutes": 30},
                ],
            ),
        ]

        for path in default_paths:
            self._paths[path.path_id] = path

    def add_contact(self, contact: EscalationContact) -> None:
        """Add an escalation contact (P3)."""
        self._contacts[contact.contact_id] = contact
        logger.info(f"Escalation contact added: {contact.contact_id}")

    def add_path(self, path: EscalationPath) -> None:
        """Add a custom escalation path (P3)."""
        self._paths[path.path_id] = path
        logger.info(f"Escalation path added: {path.path_id}")

    def start_escalation(
        self,
        alert_type: str,
        alert_id: str,
        alert_details: Dict[str, Any],
    ) -> EscalationInstance:
        """
        Start an escalation for an alert (P3).

        Args:
            alert_type: Type of alert (must match a path)
            alert_id: Unique alert identifier
            alert_details: Details about the alert

        Returns:
            EscalationInstance
        """
        # Find matching path
        path = None
        for p in self._paths.values():
            if p.alert_type == alert_type:
                path = p
                break

        if not path:
            # Use default path
            path = EscalationPath(
                path_id="default_escalation",
                name="Default Escalation",
                alert_type=alert_type,
                levels=[
                    {"level": EscalationLevel.L1_ANALYST, "timeout_minutes": 30},
                    {"level": EscalationLevel.L2_SUPERVISOR, "timeout_minutes": 60},
                ],
            )

        self._instance_counter += 1
        instance_id = f"ESC-{self._instance_counter:06d}"
        now = datetime.now(timezone.utc)

        instance = EscalationInstance(
            instance_id=instance_id,
            path_id=path.path_id,
            alert_id=alert_id,
            alert_type=alert_type,
            alert_details=alert_details,
            current_level=path.levels[0]["level"],
            started_at=now,
            last_escalation_at=now,
        )

        self._instances[instance_id] = instance
        logger.warning(f"Escalation started: {instance_id} for alert {alert_id}")
        return instance

    def acknowledge_escalation(
        self,
        instance_id: str,
        acknowledged_by: str,
    ) -> Optional[EscalationInstance]:
        """Acknowledge an escalation (P3)."""
        instance = self._instances.get(instance_id)
        if not instance:
            return None

        instance.acknowledged_by = acknowledged_by
        instance.acknowledged_at = datetime.now(timezone.utc)
        instance.status = "acknowledged"

        logger.info(f"Escalation acknowledged: {instance_id} by {acknowledged_by}")
        return instance

    def resolve_escalation(
        self,
        instance_id: str,
        resolved_by: str,
    ) -> Optional[EscalationInstance]:
        """Resolve an escalation (P3)."""
        instance = self._instances.get(instance_id)
        if not instance:
            return None

        instance.resolved_by = resolved_by
        instance.resolved_at = datetime.now(timezone.utc)
        instance.status = "resolved"

        logger.info(f"Escalation resolved: {instance_id} by {resolved_by}")
        return instance

    def escalate_to_next_level(self, instance_id: str) -> Optional[EscalationInstance]:
        """Escalate to the next level (P3)."""
        instance = self._instances.get(instance_id)
        if not instance or instance.status != "active":
            return None

        path = self._paths.get(instance.path_id)
        if not path:
            return None

        # Find current level index
        current_idx = None
        for i, level_info in enumerate(path.levels):
            if level_info["level"] == instance.current_level:
                current_idx = i
                break

        if current_idx is None or current_idx >= len(path.levels) - 1:
            logger.warning(f"Escalation {instance_id} at maximum level")
            return instance

        # Move to next level
        next_level = path.levels[current_idx + 1]["level"]
        instance.current_level = next_level
        instance.last_escalation_at = datetime.now(timezone.utc)

        logger.warning(f"Escalation {instance_id} escalated to {next_level.value}")
        return instance

    def check_escalations_for_timeout(self) -> List[EscalationInstance]:
        """Check for escalations that need to be escalated due to timeout (P3)."""
        now = datetime.now(timezone.utc)
        escalated = []

        for instance in self._instances.values():
            if instance.status != "active":
                continue

            path = self._paths.get(instance.path_id)
            if not path or not path.auto_escalate:
                continue

            # Find current level timeout
            for level_info in path.levels:
                if level_info["level"] == instance.current_level:
                    timeout_minutes = level_info.get("timeout_minutes", 60)
                    time_since_escalation = (now - instance.last_escalation_at).total_seconds() / 60

                    if time_since_escalation >= timeout_minutes:
                        self.escalate_to_next_level(instance.instance_id)
                        escalated.append(instance)
                    break

        return escalated

    def get_active_escalations(self) -> List[EscalationInstance]:
        """Get all active escalations (P3)."""
        return [i for i in self._instances.values() if i.status == "active"]

    def get_escalation_summary(self) -> Dict[str, Any]:
        """Get escalation summary (P3)."""
        instances = list(self._instances.values())
        active = [i for i in instances if i.status == "active"]
        acknowledged = [i for i in instances if i.status == "acknowledged"]

        return {
            "total_escalations": len(instances),
            "active_count": len(active),
            "acknowledged_count": len(acknowledged),
            "resolved_count": len([i for i in instances if i.status == "resolved"]),
            "active_by_level": {
                level.value: len([i for i in active if i.current_level == level])
                for level in EscalationLevel
            },
            "paths_configured": len(self._paths),
            "contacts_configured": len(self._contacts),
        }


class RegulatoryComplianceSystem:
    """
    Unified system for regulatory change monitoring and compliance reporting.

    Combines change monitoring with report generation for comprehensive
    regulatory compliance management.

    Includes:
    - REG-P1-2: Exchange circuit breaker monitoring
    - REG-P1-3: CFTC position limit monitoring
    - P3: Regulatory change tracking
    - P3: Compliance dashboard metrics
    - P3: Alert escalation paths
    """

    def __init__(self):
        self.change_monitor = RegulatoryChangeMonitor()
        self.report_generator = ComplianceReportGenerator()
        # REG-P1-2: Circuit breaker monitoring
        self.circuit_breaker_monitor = ExchangeCircuitBreakerMonitor()
        # REG-P1-3: CFTC position limit monitoring
        self.cftc_position_monitor = CFTCPositionLimitMonitor()
        # P3: Regulatory change tracking
        self.change_tracker = RegulatoryChangeTracker()
        # P3: Compliance dashboard
        self.dashboard = ComplianceDashboard()
        # P3: Alert escalation
        self.escalation_manager = AlertEscalationManager()

    def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status including circuit breakers and position limits."""
        change_summary = self.change_monitor.get_change_summary()
        circuit_breaker_summary = self.circuit_breaker_monitor.get_circuit_breaker_summary()
        position_limit_summary = self.cftc_position_monitor.get_position_limit_summary()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regulatory_changes": change_summary,
            "available_templates": len(self.report_generator.templates),
            "pending_changes": len(self.change_monitor.get_pending_changes()),
            "overdue_changes": len(self.change_monitor.get_overdue_changes()),
            # REG-P1-2: Circuit breaker status
            "circuit_breakers": {
                "exchanges_halted": circuit_breaker_summary["exchanges_halted"],
                "symbols_halted": circuit_breaker_summary["symbols_halted"],
            },
            # REG-P1-3: Position limit status
            "position_limits": {
                "commodities_tracked": position_limit_summary["commodities_tracked"],
                "breach_count": position_limit_summary["limit_breaches_total"],
            }
        }

    def check_exchange_circuit_breaker_status(self, exchange_mic: str) -> ExchangeCircuitBreakerState:
        """
        Check circuit breaker status for an exchange (REG-P1-2).

        Args:
            exchange_mic: Market Identifier Code of exchange

        Returns:
            Current circuit breaker state
        """
        return self.circuit_breaker_monitor.check_exchange_circuit_breaker_status(exchange_mic)

    def is_trading_allowed_on_exchange(self, exchange_mic: str, symbol: str = "") -> tuple[bool, str]:
        """
        Check if trading is allowed given circuit breaker status (REG-P1-2).

        Args:
            exchange_mic: Exchange MIC
            symbol: Optional symbol to check

        Returns:
            (is_allowed, reason)
        """
        return self.circuit_breaker_monitor.is_trading_allowed(exchange_mic, symbol)

    def validate_commodity_order(
        self,
        commodity_code: str,
        order_quantity: int,
        is_spot_month: bool = False
    ) -> tuple[bool, str, List[PositionLimitCheck]]:
        """
        Validate a commodity order against CFTC position limits (REG-P1-3).

        This should be called before EVERY commodity futures/options order.

        Args:
            commodity_code: e.g., "CL", "NG", "GC"
            order_quantity: Number of contracts
            is_spot_month: Whether order is in spot month

        Returns:
            (is_allowed, reason, list of limit checks)
        """
        return self.cftc_position_monitor.validate_order_against_limits(
            commodity_code, order_quantity, is_spot_month
        )

    def update_commodity_position(
        self,
        commodity_code: str,
        contract_month: str,
        position: int
    ) -> None:
        """
        Update commodity position for CFTC limit tracking (REG-P1-3).

        Args:
            commodity_code: e.g., "CL", "NG"
            contract_month: e.g., "2024-03"
            position: Net position in contracts
        """
        self.cftc_position_monitor.update_position(commodity_code, contract_month, position)

    def generate_compliance_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for compliance dashboard."""
        pending = self.change_monitor.get_pending_changes()
        overdue = self.change_monitor.get_overdue_changes()
        circuit_breaker_summary = self.circuit_breaker_monitor.get_circuit_breaker_summary()
        position_limit_summary = self.cftc_position_monitor.get_position_limit_summary()

        return {
            "summary": {
                "total_tracked_changes": len(self.change_monitor.changes),
                "pending_changes": len(pending),
                "overdue_changes": len(overdue),
                "critical_changes": len([c for c in pending if c.impact == ChangeImpact.CRITICAL])
            },
            "pending_changes": [c.to_dict() for c in pending[:10]],
            "overdue_changes": [c.to_dict() for c in overdue],
            "upcoming_deadlines": self._get_upcoming_deadlines(),
            "templates_available": len(self.report_generator.templates),
            # REG-P1-2: Circuit breaker dashboard data
            "circuit_breakers": circuit_breaker_summary,
            # REG-P1-3: Position limit dashboard data
            "position_limits": position_limit_summary
        }

    def _get_upcoming_deadlines(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get regulatory deadlines in the next N days."""
        cutoff = datetime.now(timezone.utc) + timedelta(days=days)
        deadlines = []

        for change in self.change_monitor.changes.values():
            if change.effective_date <= cutoff and change.status != ChangeStatus.VERIFIED:
                deadlines.append({
                    "change_id": change.change_id,
                    "title": change.title,
                    "effective_date": change.effective_date.isoformat(),
                    "days_remaining": change.days_until_effective(),
                    "impact": change.impact.value
                })

        deadlines.sort(key=lambda d: d["days_remaining"])
        return deadlines

    # =========================================================================
    # REGULATORY CHANGE TRACKING (P3)
    # =========================================================================

    def track_regulatory_change(
        self,
        change_id: str,
        assigned_team: str = "",
        assigned_individual: str = "",
    ) -> Optional[RegulatoryChangeTracking]:
        """
        Start tracking a regulatory change (P3).

        Args:
            change_id: ID of the registered change to track
            assigned_team: Team responsible for implementation
            assigned_individual: Individual responsible

        Returns:
            RegulatoryChangeTracking or None if change not found
        """
        change = self.change_monitor.changes.get(change_id)
        if not change:
            return None

        return self.change_tracker.start_tracking(change, assigned_team, assigned_individual)

    def update_change_tracking_status(
        self,
        change_id: str,
        new_status: ChangeTrackingStatus,
        updated_by: str,
        note: str = "",
    ) -> Optional[RegulatoryChangeTracking]:
        """Update regulatory change tracking status (P3)."""
        return self.change_tracker.update_status(change_id, new_status, updated_by, note)

    def get_change_tracking_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked regulatory changes (P3)."""
        return self.change_tracker.get_tracking_summary()

    def get_overdue_regulatory_changes(self) -> List[RegulatoryChangeTracking]:
        """Get overdue regulatory changes (P3)."""
        return self.change_tracker.get_overdue_changes()

    # =========================================================================
    # COMPLIANCE DASHBOARD METRICS (P3)
    # =========================================================================

    def update_dashboard_metric(self, metric_id: str, value: float) -> Optional[ComplianceDashboardMetric]:
        """Update a compliance dashboard metric (P3)."""
        return self.dashboard.update_metric(metric_id, value)

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get compliance dashboard summary (P3)."""
        return self.dashboard.get_dashboard_summary()

    def get_critical_metrics(self) -> List[ComplianceDashboardMetric]:
        """Get metrics in critical status (P3)."""
        return self.dashboard.get_metrics_by_status("critical")

    def refresh_dashboard_metrics(self) -> None:
        """
        Refresh dashboard metrics from current system state (P3).

        Updates metrics based on current compliance status.
        """
        # Update pending regulatory changes metric
        pending_changes = len(self.change_monitor.get_pending_changes())
        self.dashboard.update_metric("regulatory_changes_pending", float(pending_changes))

        # Update overdue items
        overdue_count = len(self.change_monitor.get_overdue_changes())
        overdue_count += len(self.change_tracker.get_overdue_changes())
        self.dashboard.update_metric("overdue_items", float(overdue_count))

        # Update position limit utilization (from CFTC monitor)
        summary = self.cftc_position_monitor.get_position_limit_summary()
        max_util = max(
            (s.get("utilization_pct", 0) for s in summary.get("summaries", {}).values()),
            default=0
        )
        self.dashboard.update_metric("position_limit_utilization", float(max_util))

        logger.debug("Dashboard metrics refreshed")

    # =========================================================================
    # ALERT ESCALATION PATHS (P3)
    # =========================================================================

    def start_alert_escalation(
        self,
        alert_type: str,
        alert_id: str,
        alert_details: Dict[str, Any],
    ) -> EscalationInstance:
        """
        Start an escalation for an alert (P3).

        Args:
            alert_type: Type of alert (market_abuse, position_limit, etc.)
            alert_id: Unique alert identifier
            alert_details: Details about the alert

        Returns:
            EscalationInstance
        """
        return self.escalation_manager.start_escalation(alert_type, alert_id, alert_details)

    def acknowledge_alert(self, instance_id: str, acknowledged_by: str) -> Optional[EscalationInstance]:
        """Acknowledge an escalated alert (P3)."""
        return self.escalation_manager.acknowledge_escalation(instance_id, acknowledged_by)

    def resolve_alert(self, instance_id: str, resolved_by: str) -> Optional[EscalationInstance]:
        """Resolve an escalated alert (P3)."""
        return self.escalation_manager.resolve_escalation(instance_id, resolved_by)

    def check_escalation_timeouts(self) -> List[EscalationInstance]:
        """Check and auto-escalate timed out alerts (P3)."""
        return self.escalation_manager.check_escalations_for_timeout()

    def get_active_escalations(self) -> List[EscalationInstance]:
        """Get all active alert escalations (P3)."""
        return self.escalation_manager.get_active_escalations()

    def get_escalation_summary(self) -> Dict[str, Any]:
        """Get alert escalation summary (P3)."""
        return self.escalation_manager.get_escalation_summary()

    def add_escalation_contact(self, contact: EscalationContact) -> None:
        """Add an escalation contact (P3)."""
        self.escalation_manager.add_contact(contact)

    def get_full_compliance_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive compliance dashboard data (P3).

        Combines all compliance metrics, tracking, and escalation data.
        """
        # Refresh metrics first
        self.refresh_dashboard_metrics()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "compliance_status": self.get_compliance_status(),
            "dashboard_metrics": self.get_dashboard_metrics(),
            "change_tracking": self.get_change_tracking_summary(),
            "escalations": self.get_escalation_summary(),
            "circuit_breakers": self.circuit_breaker_monitor.get_circuit_breaker_summary(),
            "position_limits": self.cftc_position_monitor.get_position_limit_summary(),
            "upcoming_deadlines": self._get_upcoming_deadlines(30),
        }
