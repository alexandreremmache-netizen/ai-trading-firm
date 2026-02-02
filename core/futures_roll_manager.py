"""
Futures Roll Manager
====================

Manages futures contract rolling for continuous exposure.
Handles roll schedules, roll date detection, price adjustment,
and roll notifications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from enum import Enum
from typing import Any, Callable

from core.contract_specs import CONTRACT_SPECS, ContractSpec, AssetClass


logger = logging.getLogger(__name__)


class RollMethod(Enum):
    """Method for rolling futures contracts."""
    CALENDAR = "calendar"  # Roll on calendar date
    VOLUME = "volume"  # Roll when back month volume exceeds front
    OPEN_INTEREST = "open_interest"  # Roll when OI shifts
    FIRST_NOTICE = "first_notice"  # Roll before first notice day


class RollStatus(Enum):
    """Current roll status for a contract."""
    NOT_DUE = "not_due"  # Normal trading
    APPROACHING = "approaching"  # Within roll window
    ROLLING = "rolling"  # Actively rolling
    COMPLETED = "completed"  # Roll completed


@dataclass(frozen=True)
class RollSchedule:
    """Roll schedule configuration for a contract."""
    symbol: str
    roll_days_before_expiry: int  # Days before expiry to start roll
    roll_window_days: int  # Number of days to complete roll
    roll_method: RollMethod
    contract_months: tuple[str, ...]  # Valid contract months

    # Month code to month number mapping
    MONTH_CODES = {
        "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
        "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12
    }

    def get_next_contract_month(self, current_month: str, current_year: int) -> tuple[str, int]:
        """
        Get next contract month after current.

        Args:
            current_month: Current month code (e.g., "H")
            current_year: Current year

        Returns:
            Tuple of (next_month_code, year)
        """
        current_month_num = self.MONTH_CODES.get(current_month, 1)

        # Find next valid contract month
        for month_code in self.contract_months:
            month_num = self.MONTH_CODES[month_code]
            if month_num > current_month_num:
                return month_code, current_year

        # Wrap to first month of next year
        return self.contract_months[0], current_year + 1


# =============================================================================
# DEFAULT ROLL SCHEDULES
# =============================================================================

DEFAULT_ROLL_SCHEDULES: dict[str, RollSchedule] = {
    # Index Futures - Roll 5 days before expiry
    "ES": RollSchedule(
        symbol="ES",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("H", "M", "U", "Z"),
    ),
    "NQ": RollSchedule(
        symbol="NQ",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("H", "M", "U", "Z"),
    ),
    "YM": RollSchedule(
        symbol="YM",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("H", "M", "U", "Z"),
    ),
    "RTY": RollSchedule(
        symbol="RTY",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("H", "M", "U", "Z"),
    ),

    # Energy Futures - Roll 5 days before expiry
    "CL": RollSchedule(
        symbol="CL",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"),
    ),
    "NG": RollSchedule(
        symbol="NG",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"),
    ),
    "RB": RollSchedule(
        symbol="RB",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"),
    ),
    "HO": RollSchedule(
        symbol="HO",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"),
    ),

    # Precious Metals - Roll 7 days before expiry
    "GC": RollSchedule(
        symbol="GC",
        roll_days_before_expiry=7,
        roll_window_days=4,
        roll_method=RollMethod.CALENDAR,
        contract_months=("G", "J", "M", "Q", "V", "Z"),
    ),
    "SI": RollSchedule(
        symbol="SI",
        roll_days_before_expiry=7,
        roll_window_days=4,
        roll_method=RollMethod.CALENDAR,
        contract_months=("H", "K", "N", "U", "Z"),
    ),
    "PL": RollSchedule(
        symbol="PL",
        roll_days_before_expiry=7,
        roll_window_days=4,
        roll_method=RollMethod.CALENDAR,
        contract_months=("F", "J", "N", "V"),
    ),
    "HG": RollSchedule(
        symbol="HG",
        roll_days_before_expiry=7,
        roll_window_days=4,
        roll_method=RollMethod.CALENDAR,
        contract_months=("H", "K", "N", "U", "Z"),
    ),

    # Agriculture - Roll 10 days before expiry
    "ZC": RollSchedule(
        symbol="ZC",
        roll_days_before_expiry=10,
        roll_window_days=5,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("H", "K", "N", "U", "Z"),
    ),
    "ZW": RollSchedule(
        symbol="ZW",
        roll_days_before_expiry=10,
        roll_window_days=5,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("H", "K", "N", "U", "Z"),
    ),
    "ZS": RollSchedule(
        symbol="ZS",
        roll_days_before_expiry=10,
        roll_window_days=5,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("F", "H", "K", "N", "Q", "U", "X"),
    ),
    "ZM": RollSchedule(
        symbol="ZM",
        roll_days_before_expiry=10,
        roll_window_days=5,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("F", "H", "K", "N", "Q", "U", "V", "Z"),
    ),
    "ZL": RollSchedule(
        symbol="ZL",
        roll_days_before_expiry=10,
        roll_window_days=5,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("F", "H", "K", "N", "Q", "U", "V", "Z"),
    ),

    # Bonds - Roll 5 days before expiry
    "ZB": RollSchedule(
        symbol="ZB",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("H", "M", "U", "Z"),
    ),
    "ZN": RollSchedule(
        symbol="ZN",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("H", "M", "U", "Z"),
    ),
    "ZF": RollSchedule(
        symbol="ZF",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("H", "M", "U", "Z"),
    ),
}


class FNDViolationLevel(Enum):
    """Level of FND (First Notice Date) violation risk."""
    SAFE = "safe"  # No risk
    WARNING = "warning"  # Approaching FND
    CRITICAL = "critical"  # At or past FND
    VIOLATION = "violation"  # Past FND with position - immediate action required


@dataclass
class FNDStatus:
    """First Notice Date status for a contract."""
    contract_code: str
    is_physical_delivery: bool
    first_notice_date: date | None
    days_to_fnd: int | None
    violation_level: FNDViolationLevel
    message: str
    requires_action: bool


@dataclass
class ContractInfo:
    """Information about a specific contract."""
    symbol: str  # Base symbol (e.g., "ES")
    contract_code: str  # Full code (e.g., "ESH25")
    month_code: str  # Month code (e.g., "H")
    year: int  # Full year (e.g., 2025)
    expiry_date: date
    first_notice_date: date | None = None
    last_trading_date: date | None = None

    @property
    def is_expired(self) -> bool:
        """Check if contract is expired."""
        return date.today() > self.expiry_date

    @property
    def days_to_expiry(self) -> int:
        """Days until expiry."""
        return (self.expiry_date - date.today()).days

    @property
    def days_to_fnd(self) -> int | None:
        """Days until First Notice Date."""
        if self.first_notice_date is None:
            return None
        return (self.first_notice_date - date.today()).days

    @property
    def is_physical_delivery(self) -> bool:
        """Check if this is a physical delivery contract."""
        spec = CONTRACT_SPECS.get(self.symbol)
        return spec is not None and spec.settlement_type == "physical"

    @property
    def is_past_fnd(self) -> bool:
        """Check if we're past First Notice Date."""
        if self.first_notice_date is None:
            return False
        return date.today() >= self.first_notice_date

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "contract_code": self.contract_code,
            "month_code": self.month_code,
            "year": self.year,
            "expiry_date": self.expiry_date.isoformat(),
            "first_notice_date": self.first_notice_date.isoformat() if self.first_notice_date else None,
            "days_to_expiry": self.days_to_expiry,
            "days_to_fnd": self.days_to_fnd,
            "is_expired": self.is_expired,
            "is_physical_delivery": self.is_physical_delivery,
            "is_past_fnd": self.is_past_fnd,
        }


@dataclass
class RollEvent:
    """Event representing a roll signal or completion."""
    event_id: str
    timestamp: datetime
    symbol: str
    event_type: str  # "roll_signal", "roll_start", "roll_complete"
    from_contract: str
    to_contract: str
    price_adjustment: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "event_type": self.event_type,
            "from_contract": self.from_contract,
            "to_contract": self.to_contract,
            "price_adjustment": self.price_adjustment,
            "details": self.details,
        }


def parse_contract_code(contract_code: str) -> tuple[str, str, int] | None:
    """
    Parse futures contract code into root symbol, month code, and year.

    Handles multiple formats (#F2):
    - "ESZ4"   -> ("ES", "Z", 2024)   2-letter symbol + month + 1-digit year
    - "ESZ24"  -> ("ES", "Z", 2024)   2-letter symbol + month + 2-digit year
    - "NGF25"  -> ("NG", "F", 2025)   2-letter symbol + month + 2-digit year
    - "RTYZ4"  -> ("RTY", "Z", 2024)  3-letter symbol + month + 1-digit year
    - "RTYZ24" -> ("RTY", "Z", 2024)  3-letter symbol + month + 2-digit year

    Args:
        contract_code: Full contract code string

    Returns:
        Tuple of (root_symbol, month_code, year) or None if parsing fails
    """
    MONTH_CODES = set("FGHJKMNQUVXZ")

    if not contract_code or len(contract_code) < 3:
        return None

    contract_code = contract_code.upper().strip()

    # Try to find the month code position
    # Month codes: F, G, H, J, K, M, N, Q, U, V, X, Z
    # The month code should be followed by 1 or 2 digit year

    # Try 3-letter root first (less common)
    for root_len in [3, 2]:
        if len(contract_code) < root_len + 2:
            continue

        potential_root = contract_code[:root_len]
        remainder = contract_code[root_len:]

        if len(remainder) >= 1 and remainder[0] in MONTH_CODES:
            month_code = remainder[0]
            year_part = remainder[1:]

            if year_part.isdigit() and len(year_part) in [1, 2]:
                # Valid year suffix
                current_year = date.today().year
                current_decade = (current_year // 10) * 10
                current_century = (current_year // 100) * 100

                if len(year_part) == 1:
                    # Single digit year (e.g., "4" -> 2024 or 2034)
                    year = current_decade + int(year_part)
                    # Adjust for wrap-around
                    if year < current_year - 5:
                        year += 10
                else:
                    # Two digit year (e.g., "24" -> 2024, "25" -> 2025)
                    year_suffix = int(year_part)
                    year = current_century + year_suffix
                    # Handle century boundary
                    if year < current_year - 50:
                        year += 100

                # Verify root is a known symbol (if possible)
                if potential_root in CONTRACT_SPECS or potential_root in DEFAULT_ROLL_SCHEDULES:
                    return potential_root, month_code, year
                # If not found in known symbols, still return for unknown/new contracts
                elif root_len == 2:  # Default to 2-letter if 3-letter failed
                    return potential_root, month_code, year

    return None


def format_contract_code(root_symbol: str, month_code: str, year: int, two_digit_year: bool = False) -> str:
    """
    Format a contract code from components.

    Args:
        root_symbol: Root symbol (e.g., "ES", "NG", "RTY")
        month_code: Month code (e.g., "Z", "F")
        year: Full year (e.g., 2024)
        two_digit_year: If True, use 2-digit year (e.g., "24"), else 1-digit

    Returns:
        Formatted contract code (e.g., "ESZ4" or "ESZ24")
    """
    if two_digit_year:
        year_suffix = str(year)[-2:]
    else:
        year_suffix = str(year)[-1]
    return f"{root_symbol}{month_code}{year_suffix}"


class FuturesRollManager:
    """
    Manages futures contract rolling.

    Features:
    - Automatic roll date detection
    - Roll signal generation
    - Continuous contract price adjustment
    - Roll event notifications
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize roll manager.

        Args:
            config: Configuration with:
                - enabled: Enable auto-roll detection (default: True)
                - auto_roll: Automatically generate roll signals (default: True)
                - custom_schedules: Override default roll schedules
        """
        self._config = config or {}
        self._enabled = self._config.get("enabled", True)
        self._auto_roll = self._config.get("auto_roll", True)

        # Roll schedules
        self._schedules: dict[str, RollSchedule] = dict(DEFAULT_ROLL_SCHEDULES)
        custom = self._config.get("custom_schedules", {})
        self._schedules.update(custom)

        # Current contract tracking
        self._current_contracts: dict[str, ContractInfo] = {}
        self._roll_status: dict[str, RollStatus] = {}

        # Price adjustments for continuous contracts
        self._price_adjustments: dict[str, float] = {}

        # Event callbacks
        self._event_callbacks: list[Callable[[RollEvent], None]] = []

        # Event counter
        self._event_counter = 0

        # Roll history
        self._roll_history: list[RollEvent] = []

        logger.info(f"FuturesRollManager initialized: {len(self._schedules)} schedules loaded")

    def register_callback(self, callback: Callable[[RollEvent], None]) -> None:
        """Register callback for roll events."""
        self._event_callbacks.append(callback)

    def set_current_contract(
        self,
        symbol: str,
        contract_code: str,
        expiry_date: date,
        first_notice_date: date | None = None
    ) -> None:
        """
        Set the current active contract for a symbol.

        Args:
            symbol: Base symbol (e.g., "ES", "NG", "RTY")
            contract_code: Full contract code (e.g., "ESH25", "NGF25", "RTYZ24")
            expiry_date: Contract expiry date
            first_notice_date: First notice date for physical delivery
        """
        # Parse contract code using improved parser (#F2)
        parsed = parse_contract_code(contract_code)

        if parsed is None:
            # Fallback to simple parsing for compatibility
            logger.warning(f"Could not parse contract code {contract_code}, using fallback")
            month_code = ""
            year = date.today().year
            if len(contract_code) >= 3:
                # Try to extract month from position based on symbol length
                symbol_len = len(symbol)
                if len(contract_code) > symbol_len:
                    month_code = contract_code[symbol_len]
                    year_part = contract_code[symbol_len + 1:]
                    if year_part.isdigit():
                        if len(year_part) == 1:
                            year = (year // 10) * 10 + int(year_part)
                        else:
                            year = (year // 100) * 100 + int(year_part)

            # Raise error if parsing completely failed
            if not month_code:
                raise ValueError(f"Could not parse contract code: {contract_code}")
        else:
            _, month_code, year = parsed

        contract_info = ContractInfo(
            symbol=symbol,
            contract_code=contract_code,
            month_code=month_code,
            year=year,
            expiry_date=expiry_date,
            first_notice_date=first_notice_date,
        )

        self._current_contracts[symbol] = contract_info
        self._roll_status[symbol] = RollStatus.NOT_DUE

        logger.info(f"Set current contract: {symbol} -> {contract_code} (month={month_code}, year={year}), expires {expiry_date}")

    def check_roll_status(self, symbol: str) -> RollStatus:
        """
        Check if a contract needs to be rolled.

        Args:
            symbol: Base symbol

        Returns:
            Current roll status
        """
        if symbol not in self._current_contracts:
            return RollStatus.NOT_DUE

        if symbol not in self._schedules:
            return RollStatus.NOT_DUE

        contract = self._current_contracts[symbol]
        schedule = self._schedules[symbol]

        days_to_expiry = contract.days_to_expiry

        # Check for physical delivery contracts (use first notice date)
        if schedule.roll_method == RollMethod.FIRST_NOTICE and contract.first_notice_date:
            days_to_notice = (contract.first_notice_date - date.today()).days
            reference_days = min(days_to_expiry, days_to_notice)
        else:
            reference_days = days_to_expiry

        # Determine status
        if reference_days <= 0:
            status = RollStatus.COMPLETED  # Expired
        elif reference_days <= schedule.roll_window_days:
            status = RollStatus.ROLLING
        elif reference_days <= schedule.roll_days_before_expiry:
            status = RollStatus.APPROACHING
        else:
            status = RollStatus.NOT_DUE

        # Check for status change
        old_status = self._roll_status.get(symbol, RollStatus.NOT_DUE)
        if status != old_status:
            self._roll_status[symbol] = status
            self._handle_status_change(symbol, old_status, status)

        return status

    def _handle_status_change(
        self,
        symbol: str,
        old_status: RollStatus,
        new_status: RollStatus
    ) -> None:
        """Handle roll status change."""
        contract = self._current_contracts[symbol]

        if new_status == RollStatus.APPROACHING:
            # Generate roll signal
            self._generate_roll_event(
                symbol=symbol,
                event_type="roll_signal",
                details={
                    "days_to_expiry": contract.days_to_expiry,
                    "reason": "approaching_expiry",
                }
            )

        elif new_status == RollStatus.ROLLING:
            # Roll has started
            self._generate_roll_event(
                symbol=symbol,
                event_type="roll_start",
                details={
                    "days_to_expiry": contract.days_to_expiry,
                }
            )

    def _generate_roll_event(
        self,
        symbol: str,
        event_type: str,
        details: dict[str, Any] | None = None
    ) -> RollEvent:
        """Generate and dispatch a roll event."""
        self._event_counter += 1
        event_id = f"ROLL-{self._event_counter:06d}"

        contract = self._current_contracts[symbol]
        schedule = self._schedules.get(symbol)

        # Determine next contract
        if schedule:
            next_month, next_year = schedule.get_next_contract_month(
                contract.month_code, contract.year
            )
            to_contract = f"{symbol}{next_month}{str(next_year)[-1]}"
        else:
            to_contract = "unknown"

        event = RollEvent(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            event_type=event_type,
            from_contract=contract.contract_code,
            to_contract=to_contract,
            price_adjustment=self._price_adjustments.get(symbol, 0.0),
            details=details or {},
        )

        # Store in history
        self._roll_history.append(event)

        # Dispatch to callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Roll event callback error: {e}")

        logger.info(f"Roll event: {event_type} for {symbol} ({contract.contract_code} -> {to_contract})")

        return event

    def complete_roll(
        self,
        symbol: str,
        new_contract_code: str,
        new_expiry_date: date,
        roll_price_adjustment: float = 0.0,
        first_notice_date: date | None = None
    ) -> RollEvent:
        """
        Complete a roll to a new contract.

        Args:
            symbol: Base symbol
            new_contract_code: New contract code
            new_expiry_date: New expiry date
            roll_price_adjustment: Price adjustment for continuous pricing
            first_notice_date: First notice date for new contract

        Returns:
            Roll completion event
        """
        old_contract = self._current_contracts.get(symbol)
        old_code = old_contract.contract_code if old_contract else "unknown"

        # Update price adjustment
        current_adj = self._price_adjustments.get(symbol, 0.0)
        self._price_adjustments[symbol] = current_adj + roll_price_adjustment

        # Set new contract
        self.set_current_contract(
            symbol=symbol,
            contract_code=new_contract_code,
            expiry_date=new_expiry_date,
            first_notice_date=first_notice_date,
        )

        # Update status
        self._roll_status[symbol] = RollStatus.COMPLETED

        # Generate completion event
        event = RollEvent(
            event_id=f"ROLL-{self._event_counter + 1:06d}",
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            event_type="roll_complete",
            from_contract=old_code,
            to_contract=new_contract_code,
            price_adjustment=roll_price_adjustment,
            details={
                "total_adjustment": self._price_adjustments[symbol],
            }
        )

        self._event_counter += 1
        self._roll_history.append(event)

        # Dispatch callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Roll event callback error: {e}")

        logger.info(f"Roll complete: {symbol} {old_code} -> {new_contract_code}, adj={roll_price_adjustment:.4f}")

        return event

    def get_continuous_price(self, symbol: str, raw_price: float) -> float:
        """
        Get continuous contract adjusted price.

        Args:
            symbol: Contract symbol
            raw_price: Raw price from current contract

        Returns:
            Adjusted continuous price
        """
        adjustment = self._price_adjustments.get(symbol, 0.0)
        return raw_price + adjustment

    def get_raw_price(self, symbol: str, continuous_price: float) -> float:
        """
        Convert continuous price back to raw price.

        Args:
            symbol: Contract symbol
            continuous_price: Adjusted continuous price

        Returns:
            Raw price in current contract
        """
        adjustment = self._price_adjustments.get(symbol, 0.0)
        return continuous_price - adjustment

    def get_current_contract(self, symbol: str) -> ContractInfo | None:
        """Get current contract info for symbol."""
        return self._current_contracts.get(symbol)

    def get_roll_schedule(self, symbol: str) -> RollSchedule | None:
        """Get roll schedule for symbol."""
        return self._schedules.get(symbol)

    def get_contracts_approaching_roll(self) -> list[ContractInfo]:
        """Get list of contracts approaching roll."""
        approaching = []
        for symbol, contract in self._current_contracts.items():
            status = self.check_roll_status(symbol)
            if status in [RollStatus.APPROACHING, RollStatus.ROLLING]:
                approaching.append(contract)
        return approaching

    def get_roll_history(
        self,
        symbol: str | None = None,
        limit: int = 100
    ) -> list[RollEvent]:
        """
        Get roll event history.

        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum events to return

        Returns:
            List of roll events
        """
        events = self._roll_history

        if symbol:
            events = [e for e in events if e.symbol == symbol]

        return events[-limit:]

    def check_all_contracts(self) -> dict[str, RollStatus]:
        """
        Check roll status for all tracked contracts.

        Returns:
            Dictionary of symbol to roll status
        """
        statuses = {}
        for symbol in self._current_contracts:
            statuses[symbol] = self.check_roll_status(symbol)
        return statuses

    def check_fnd_status(
        self,
        symbol: str,
        has_position: bool = False,
        warning_days: int = 5
    ) -> FNDStatus:
        """
        Check First Notice Date status for a contract.

        CRITICAL: Physical delivery contracts must be closed before FND
        to avoid delivery obligations. This is especially important for
        retail/prop traders who cannot take physical delivery.

        Args:
            symbol: Contract symbol
            has_position: Whether we currently have a position
            warning_days: Days before FND to start warning

        Returns:
            FNDStatus with violation level and required action
        """
        contract = self._current_contracts.get(symbol)

        if contract is None:
            return FNDStatus(
                contract_code="unknown",
                is_physical_delivery=False,
                first_notice_date=None,
                days_to_fnd=None,
                violation_level=FNDViolationLevel.SAFE,
                message="Contract not tracked",
                requires_action=False,
            )

        # Check if physical delivery
        if not contract.is_physical_delivery:
            return FNDStatus(
                contract_code=contract.contract_code,
                is_physical_delivery=False,
                first_notice_date=contract.first_notice_date,
                days_to_fnd=contract.days_to_fnd,
                violation_level=FNDViolationLevel.SAFE,
                message="Cash-settled contract, no FND risk",
                requires_action=False,
            )

        # No FND set - warn about potential risk
        if contract.first_notice_date is None:
            return FNDStatus(
                contract_code=contract.contract_code,
                is_physical_delivery=True,
                first_notice_date=None,
                days_to_fnd=None,
                violation_level=FNDViolationLevel.WARNING,
                message="Physical delivery contract with unknown FND - verify expiry schedule",
                requires_action=True,
            )

        days_to_fnd = contract.days_to_fnd

        # Determine violation level
        if days_to_fnd is None or days_to_fnd > warning_days:
            level = FNDViolationLevel.SAFE
            message = f"FND is {days_to_fnd} days away"
            requires_action = False

        elif days_to_fnd > 0:
            level = FNDViolationLevel.WARNING
            message = f"FND in {days_to_fnd} days - close position before FND"
            requires_action = has_position

        elif days_to_fnd == 0:
            level = FNDViolationLevel.CRITICAL
            message = "TODAY is First Notice Date - IMMEDIATE action required"
            requires_action = has_position

        else:  # Past FND
            if has_position:
                level = FNDViolationLevel.VIOLATION
                message = f"VIOLATION: Position held {-days_to_fnd} days past FND - delivery risk!"
                requires_action = True
            else:
                level = FNDViolationLevel.CRITICAL
                message = f"Past FND by {-days_to_fnd} days - do not open new positions"
                requires_action = False

        return FNDStatus(
            contract_code=contract.contract_code,
            is_physical_delivery=True,
            first_notice_date=contract.first_notice_date,
            days_to_fnd=days_to_fnd,
            violation_level=level,
            message=message,
            requires_action=requires_action,
        )

    def can_hold_position(self, symbol: str) -> tuple[bool, str]:
        """
        Check if it's safe to hold a position in this contract.

        Enforces FND rules for physical delivery contracts.

        Args:
            symbol: Contract symbol

        Returns:
            Tuple of (can_hold, reason)
        """
        fnd_status = self.check_fnd_status(symbol, has_position=True)

        if fnd_status.violation_level == FNDViolationLevel.VIOLATION:
            return False, fnd_status.message

        if fnd_status.violation_level == FNDViolationLevel.CRITICAL:
            return False, fnd_status.message

        return True, "OK"

    def can_open_position(self, symbol: str) -> tuple[bool, str]:
        """
        Check if it's safe to open a new position in this contract.

        More restrictive than can_hold_position - prevents opening
        positions too close to FND.

        Args:
            symbol: Contract symbol

        Returns:
            Tuple of (can_open, reason)
        """
        contract = self._current_contracts.get(symbol)

        if contract is None:
            return True, "Contract not tracked"

        # Check expiry
        if contract.is_expired:
            return False, f"Contract {contract.contract_code} has expired"

        # Check FND for physical delivery
        fnd_status = self.check_fnd_status(symbol, has_position=False, warning_days=3)

        if fnd_status.violation_level in [FNDViolationLevel.CRITICAL, FNDViolationLevel.VIOLATION]:
            return False, f"Cannot open position: {fnd_status.message}"

        if fnd_status.violation_level == FNDViolationLevel.WARNING:
            logger.warning(f"Opening position in {symbol} near FND: {fnd_status.message}")

        return True, "OK"

    def get_fnd_violations(
        self,
        positions: dict[str, float]
    ) -> list[FNDStatus]:
        """
        Check all positions for FND violations.

        Args:
            positions: Dictionary of symbol -> position quantity

        Returns:
            List of FNDStatus for contracts with violations/warnings
        """
        violations = []

        for symbol, qty in positions.items():
            if qty == 0:
                continue

            fnd_status = self.check_fnd_status(symbol, has_position=True)

            if fnd_status.violation_level in [
                FNDViolationLevel.WARNING,
                FNDViolationLevel.CRITICAL,
                FNDViolationLevel.VIOLATION,
            ]:
                violations.append(fnd_status)

        return violations

    def get_status(self) -> dict[str, Any]:
        """Get manager status for monitoring."""
        approaching = self.get_contracts_approaching_roll()

        return {
            "enabled": self._enabled,
            "auto_roll": self._auto_roll,
            "contracts_tracked": len(self._current_contracts),
            "schedules_loaded": len(self._schedules),
            "contracts_approaching_roll": len(approaching),
            "approaching_symbols": [c.symbol for c in approaching],
            "total_roll_events": len(self._roll_history),
            "contract_status": {
                symbol: {
                    "contract": contract.contract_code,
                    "days_to_expiry": contract.days_to_expiry,
                    "days_to_fnd": contract.days_to_fnd,
                    "is_physical": contract.is_physical_delivery,
                    "roll_status": self._roll_status.get(symbol, RollStatus.NOT_DUE).value,
                }
                for symbol, contract in self._current_contracts.items()
            },
        }


def calculate_roll_date(
    expiry_date: date,
    roll_days_before: int,
    method: RollMethod = RollMethod.CALENDAR
) -> date:
    """
    Calculate roll date for a contract.

    Args:
        expiry_date: Contract expiry date
        roll_days_before: Days before expiry to roll
        method: Roll method

    Returns:
        Recommended roll date
    """
    roll_date = expiry_date - timedelta(days=roll_days_before)

    # Adjust for weekends
    while roll_date.weekday() >= 5:  # Saturday or Sunday
        roll_date -= timedelta(days=1)

    return roll_date


def get_contract_expiry(symbol: str, month_code: str, year: int) -> date | None:
    """
    Estimate contract expiry date.

    Note: This is an approximation. Use exchange data for production.

    Args:
        symbol: Base symbol
        month_code: Contract month code
        year: Contract year

    Returns:
        Estimated expiry date
    """
    month_map = {
        "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
        "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12
    }

    month = month_map.get(month_code)
    if month is None:
        return None

    # Default: 3rd Friday of the month (common for financials)
    first_day = date(year, month, 1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday() + 7) % 7)
    third_friday = first_friday + timedelta(weeks=2)

    # Adjust for specific contracts
    spec = CONTRACT_SPECS.get(symbol)
    if spec and spec.asset_class == AssetClass.ENERGY:
        # Energy typically expires around 20th of month before delivery
        return date(year, month, 20) - timedelta(days=3)
    elif spec and spec.asset_class == AssetClass.AGRICULTURE:
        # Ag typically expires mid-month before delivery
        return date(year, month, 15) - timedelta(days=1)

    return third_friday


def estimate_first_notice_date(symbol: str, contract_month: str, contract_year: int) -> date | None:
    """
    P2-18: Automatically estimate First Notice Date (FND) for physical delivery contracts.

    FND varies by exchange and commodity type:
    - CME Energy (CL, NG): Last business day of month prior to delivery month
    - CME Metals (GC, SI): Last business day of month prior to delivery month
    - CME Grains (ZC, ZW, ZS): Last business day of month prior to delivery month
    - COMEX (HG): 2 business days before 1st delivery day
    - Cash-settled contracts (ES, NQ, etc.): No FND (returns None)

    Args:
        symbol: Futures symbol (e.g., "CL", "GC")
        contract_month: Contract month code (F, G, H, J, K, M, N, Q, U, V, X, Z)
        contract_year: Contract year

    Returns:
        Estimated FND date, or None for cash-settled contracts
    """
    # Month code to month number mapping
    month_map = {
        "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
        "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12
    }

    delivery_month = month_map.get(contract_month.upper())
    if delivery_month is None:
        return None

    spec = CONTRACT_SPECS.get(symbol)

    # Cash-settled contracts have no FND
    CASH_SETTLED = {"ES", "NQ", "RTY", "YM", "EMD", "VIX", "VX"}
    if symbol.upper() in CASH_SETTLED:
        return None

    if spec and not spec.is_physical_delivery:
        return None

    # Get the delivery month's first day
    delivery_date = date(contract_year, delivery_month, 1)

    # FND rules by asset class
    ENERGY_SYMBOLS = {"CL", "NG", "RB", "HO", "BZ"}
    METALS_SYMBOLS = {"GC", "SI", "HG", "PL", "PA"}
    GRAINS_SYMBOLS = {"ZC", "ZW", "ZS", "ZM", "ZL", "ZO"}
    SOFTS_SYMBOLS = {"KC", "SB", "CC", "CT", "OJ"}
    LIVESTOCK_SYMBOLS = {"LE", "HE", "GF"}

    symbol_upper = symbol.upper()

    if symbol_upper in ENERGY_SYMBOLS:
        # Energy: Last business day of month BEFORE delivery month
        # If delivery is March, FND is last business day of February
        prior_month = delivery_month - 1
        prior_year = contract_year
        if prior_month == 0:
            prior_month = 12
            prior_year -= 1

        # Find last business day of prior month
        if prior_month in (1, 3, 5, 7, 8, 10, 12):
            last_day = 31
        elif prior_month in (4, 6, 9, 11):
            last_day = 30
        else:  # February
            if (prior_year % 4 == 0 and prior_year % 100 != 0) or (prior_year % 400 == 0):
                last_day = 29
            else:
                last_day = 28

        fnd = date(prior_year, prior_month, last_day)
        # Adjust for weekends (move to Friday)
        while fnd.weekday() >= 5:
            fnd -= timedelta(days=1)
        return fnd

    elif symbol_upper in METALS_SYMBOLS:
        # Metals: Last business day of month before delivery
        prior_month = delivery_month - 1
        prior_year = contract_year
        if prior_month == 0:
            prior_month = 12
            prior_year -= 1

        if prior_month in (1, 3, 5, 7, 8, 10, 12):
            last_day = 31
        elif prior_month in (4, 6, 9, 11):
            last_day = 30
        else:
            if (prior_year % 4 == 0 and prior_year % 100 != 0) or (prior_year % 400 == 0):
                last_day = 29
            else:
                last_day = 28

        fnd = date(prior_year, prior_month, last_day)
        while fnd.weekday() >= 5:
            fnd -= timedelta(days=1)
        return fnd

    elif symbol_upper in GRAINS_SYMBOLS:
        # Grains: Last business day of month before delivery
        prior_month = delivery_month - 1
        prior_year = contract_year
        if prior_month == 0:
            prior_month = 12
            prior_year -= 1

        if prior_month in (1, 3, 5, 7, 8, 10, 12):
            last_day = 31
        elif prior_month in (4, 6, 9, 11):
            last_day = 30
        else:
            if (prior_year % 4 == 0 and prior_year % 100 != 0) or (prior_year % 400 == 0):
                last_day = 29
            else:
                last_day = 28

        fnd = date(prior_year, prior_month, last_day)
        while fnd.weekday() >= 5:
            fnd -= timedelta(days=1)
        return fnd

    elif symbol_upper in SOFTS_SYMBOLS or symbol_upper in LIVESTOCK_SYMBOLS:
        # Softs/Livestock: Typically 5 business days before delivery month
        fnd = delivery_date - timedelta(days=7)
        while fnd.weekday() >= 5:
            fnd -= timedelta(days=1)
        return fnd

    # Default for unknown physical delivery: 3 business days before delivery month
    fnd = delivery_date - timedelta(days=5)
    while fnd.weekday() >= 5:
        fnd -= timedelta(days=1)
    return fnd


def get_fnd_with_auto_estimate(
    symbol: str,
    contract_month: str,
    contract_year: int,
    manual_fnd: date | None = None
) -> date | None:
    """
    P2-18: Get FND with automatic estimation fallback.

    Args:
        symbol: Futures symbol
        contract_month: Contract month code
        contract_year: Contract year
        manual_fnd: Manually specified FND (takes precedence if provided)

    Returns:
        FND date (manual if provided, else estimated, else None for cash-settled)
    """
    if manual_fnd is not None:
        return manual_fnd

    return estimate_first_notice_date(symbol, contract_month, contract_year)


# =============================================================================
# COMPLETE ROLL CALENDAR (#F4)
# =============================================================================

# Extended roll schedules for all contract types
EXTENDED_ROLL_SCHEDULES: dict[str, RollSchedule] = {
    **DEFAULT_ROLL_SCHEDULES,
    # Additional index contracts
    "RTY": RollSchedule(
        symbol="RTY",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("H", "M", "U", "Z"),
    ),
    # Additional energy contracts
    "RB": RollSchedule(
        symbol="RB",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"),
    ),
    "HO": RollSchedule(
        symbol="HO",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"),
    ),
    # Additional metals
    "SI": RollSchedule(
        symbol="SI",
        roll_days_before_expiry=7,
        roll_window_days=4,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("H", "K", "N", "U", "Z"),
    ),
    "HG": RollSchedule(
        symbol="HG",
        roll_days_before_expiry=7,
        roll_window_days=4,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("H", "K", "N", "U", "Z"),
    ),
    "PL": RollSchedule(
        symbol="PL",
        roll_days_before_expiry=7,
        roll_window_days=4,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("F", "J", "N", "V"),
    ),
    # Additional grains
    "ZM": RollSchedule(
        symbol="ZM",
        roll_days_before_expiry=10,
        roll_window_days=5,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("F", "H", "K", "N", "Q", "U", "V", "Z"),
    ),
    "ZL": RollSchedule(
        symbol="ZL",
        roll_days_before_expiry=10,
        roll_window_days=5,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("F", "H", "K", "N", "Q", "U", "V", "Z"),
    ),
    # Treasury futures
    "ZF": RollSchedule(
        symbol="ZF",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("H", "M", "U", "Z"),
    ),
    # Currency futures
    "6E": RollSchedule(
        symbol="6E",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("H", "M", "U", "Z"),
    ),
    "6J": RollSchedule(
        symbol="6J",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("H", "M", "U", "Z"),
    ),
    "6B": RollSchedule(
        symbol="6B",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("H", "M", "U", "Z"),
    ),
    "6A": RollSchedule(
        symbol="6A",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("H", "M", "U", "Z"),
    ),
    "6C": RollSchedule(
        symbol="6C",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("H", "M", "U", "Z"),
    ),
    "6S": RollSchedule(
        symbol="6S",
        roll_days_before_expiry=5,
        roll_window_days=3,
        roll_method=RollMethod.CALENDAR,
        contract_months=("H", "M", "U", "Z"),
    ),
    # Livestock
    "LE": RollSchedule(
        symbol="LE",
        roll_days_before_expiry=10,
        roll_window_days=5,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("G", "J", "M", "Q", "V", "Z"),
    ),
    "HE": RollSchedule(
        symbol="HE",
        roll_days_before_expiry=10,
        roll_window_days=5,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("G", "J", "K", "M", "N", "Q", "V", "Z"),
    ),
    # Softs
    "KC": RollSchedule(
        symbol="KC",
        roll_days_before_expiry=7,
        roll_window_days=4,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("H", "K", "N", "U", "Z"),
    ),
    "SB": RollSchedule(
        symbol="SB",
        roll_days_before_expiry=7,
        roll_window_days=4,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("H", "K", "N", "V"),
    ),
    "CC": RollSchedule(
        symbol="CC",
        roll_days_before_expiry=7,
        roll_window_days=4,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("H", "K", "N", "U", "Z"),
    ),
    "CT": RollSchedule(
        symbol="CT",
        roll_days_before_expiry=7,
        roll_window_days=4,
        roll_method=RollMethod.FIRST_NOTICE,
        contract_months=("H", "K", "N", "V", "Z"),
    ),
}


def get_complete_roll_calendar(year: int) -> list[dict]:
    """
    Generate complete roll calendar for a year (#F4).

    Args:
        year: Calendar year

    Returns:
        List of roll events with dates and symbols
    """
    calendar = []

    for symbol, schedule in EXTENDED_ROLL_SCHEDULES.items():
        for month_code in schedule.contract_months:
            expiry = get_contract_expiry(symbol, month_code, year)
            if expiry:
                roll_date = calculate_roll_date(
                    expiry, schedule.roll_days_before_expiry, schedule.roll_method
                )
                calendar.append({
                    "symbol": symbol,
                    "contract_month": month_code,
                    "year": year,
                    "contract_code": f"{symbol}{month_code}{year % 100:02d}",
                    "expiry_date": expiry.isoformat(),
                    "roll_start_date": roll_date.isoformat(),
                    "roll_end_date": (roll_date + timedelta(days=schedule.roll_window_days)).isoformat(),
                    "roll_method": schedule.roll_method.value,
                })

    # Sort by roll date
    calendar.sort(key=lambda x: x["roll_start_date"])
    return calendar


# =============================================================================
# BASIS/CALENDAR SPREAD TRACKING (#F5)
# =============================================================================

@dataclass
class BasisSpread:
    """Tracks basis between spot and futures or between contract months (#F5)."""
    symbol: str
    front_contract: str
    back_contract: str
    front_price: float
    back_price: float
    timestamp: datetime
    spread: float = field(init=False)
    spread_pct: float = field(init=False)
    annualized_spread: float = field(init=False)
    days_between: int = 0

    def __post_init__(self):
        self.spread = self.back_price - self.front_price
        self.spread_pct = (self.spread / self.front_price * 100) if self.front_price > 0 else 0
        # Annualize the spread
        if self.days_between > 0:
            self.annualized_spread = self.spread_pct * (365 / self.days_between)
        else:
            self.annualized_spread = 0


class BasisTracker:
    """
    Tracks basis and calendar spreads for futures (#F5).

    Monitors:
    - Spot vs front-month basis (contango/backwardation)
    - Calendar spreads between contract months
    - Roll yield opportunities
    """

    def __init__(self):
        self._basis_history: dict[str, list[BasisSpread]] = {}
        self._spot_prices: dict[str, float] = {}
        self._futures_prices: dict[str, dict[str, float]] = {}  # symbol -> {contract: price}
        self._alerts: list[dict] = []

    def update_spot_price(self, symbol: str, price: float) -> None:
        """Update spot price for a symbol."""
        self._spot_prices[symbol] = price

    def update_futures_price(self, symbol: str, contract: str, price: float) -> None:
        """Update futures price for a contract."""
        if symbol not in self._futures_prices:
            self._futures_prices[symbol] = {}
        self._futures_prices[symbol][contract] = price

    def calculate_spot_basis(self, symbol: str, front_contract: str) -> BasisSpread | None:
        """
        Calculate basis between spot and front month (#F5).

        Returns:
            BasisSpread or None if prices not available
        """
        spot = self._spot_prices.get(symbol)
        futures = self._futures_prices.get(symbol, {}).get(front_contract)

        if spot is None or futures is None:
            return None

        basis = BasisSpread(
            symbol=symbol,
            front_contract="SPOT",
            back_contract=front_contract,
            front_price=spot,
            back_price=futures,
            timestamp=datetime.now(timezone.utc),
            days_between=30,  # Approximate for front month
        )

        # Track history
        if symbol not in self._basis_history:
            self._basis_history[symbol] = []
        self._basis_history[symbol].append(basis)

        # Trim history to last 100 observations
        self._basis_history[symbol] = self._basis_history[symbol][-100:]

        return basis

    def calculate_calendar_spread(
        self,
        symbol: str,
        front_contract: str,
        back_contract: str,
        days_between: int
    ) -> BasisSpread | None:
        """
        Calculate calendar spread between two contract months (#F5).

        Args:
            symbol: Base symbol
            front_contract: Front month contract code
            back_contract: Back month contract code
            days_between: Approximate days between expiries

        Returns:
            BasisSpread or None if prices not available
        """
        prices = self._futures_prices.get(symbol, {})
        front_price = prices.get(front_contract)
        back_price = prices.get(back_contract)

        if front_price is None or back_price is None:
            return None

        spread = BasisSpread(
            symbol=symbol,
            front_contract=front_contract,
            back_contract=back_contract,
            front_price=front_price,
            back_price=back_price,
            timestamp=datetime.now(timezone.utc),
            days_between=days_between,
        )

        # Track
        key = f"{symbol}:{front_contract}-{back_contract}"
        if key not in self._basis_history:
            self._basis_history[key] = []
        self._basis_history[key].append(spread)
        self._basis_history[key] = self._basis_history[key][-100:]

        return spread

    def get_term_structure(self, symbol: str) -> dict:
        """
        Get full term structure for a symbol (#F5).

        Returns prices and spreads across all contract months.
        """
        prices = self._futures_prices.get(symbol, {})
        if not prices:
            return {"error": "no_prices_available"}

        # Sort contracts by month code
        sorted_contracts = sorted(prices.items(), key=lambda x: parse_contract_month(x[0]))

        term_structure = []
        for i, (contract, price) in enumerate(sorted_contracts):
            entry = {
                "contract": contract,
                "price": price,
            }
            if i > 0:
                prev_contract, prev_price = sorted_contracts[i - 1]
                entry["spread_vs_prev"] = price - prev_price
                entry["spread_pct"] = (price - prev_price) / prev_price * 100
            term_structure.append(entry)

        # Determine market state
        if len(term_structure) >= 2:
            first_spread = term_structure[1].get("spread_vs_prev", 0)
            if first_spread > 0:
                market_state = "contango"
            elif first_spread < 0:
                market_state = "backwardation"
            else:
                market_state = "flat"
        else:
            market_state = "unknown"

        return {
            "symbol": symbol,
            "term_structure": term_structure,
            "market_state": market_state,
            "n_contracts": len(term_structure),
        }

    def detect_roll_yield_opportunity(
        self,
        symbol: str,
        threshold_bps: float = 50.0
    ) -> dict | None:
        """
        Detect roll yield opportunities (#F5).

        Identifies when calendar spread offers significant carry.

        Args:
            symbol: Symbol to analyze
            threshold_bps: Minimum annualized spread in bps to flag

        Returns:
            Opportunity details or None
        """
        term = self.get_term_structure(symbol)
        if "error" in term:
            return None

        opportunities = []
        for i, entry in enumerate(term.get("term_structure", [])):
            spread_pct = entry.get("spread_pct", 0)
            if abs(spread_pct * 100) > threshold_bps:  # Convert to bps
                opportunities.append({
                    "contract": entry["contract"],
                    "spread_bps": spread_pct * 100,
                    "direction": "sell_front_buy_back" if spread_pct > 0 else "buy_front_sell_back",
                })

        if opportunities:
            return {
                "symbol": symbol,
                "market_state": term["market_state"],
                "opportunities": opportunities,
            }
        return None

    def get_basis_statistics(self, symbol: str, lookback: int = 30) -> dict:
        """Get basis statistics for monitoring."""
        history = self._basis_history.get(symbol, [])[-lookback:]
        if not history:
            return {"error": "no_history"}

        spreads = [b.spread_pct for b in history]
        return {
            "symbol": symbol,
            "current_basis_pct": spreads[-1] if spreads else None,
            "avg_basis_pct": sum(spreads) / len(spreads),
            "min_basis_pct": min(spreads),
            "max_basis_pct": max(spreads),
            "std_basis_pct": (sum((s - sum(spreads)/len(spreads))**2 for s in spreads) / len(spreads)) ** 0.5,
            "observations": len(spreads),
        }


def parse_contract_month(contract_code: str) -> tuple[int, str]:
    """Parse contract code to (year, month_code) for sorting."""
    # Extract month code and year
    month_codes = "FGHJKMNQUVXZ"
    for i, char in enumerate(contract_code):
        if char in month_codes:
            month_code = char
            year_str = contract_code[i+1:]
            year = int(year_str) if year_str.isdigit() else 0
            if year < 100:
                year += 2000
            return (year, month_codes.index(month_code))
    return (0, 0)


# =============================================================================
# DELIVERY MONTH VALIDATION (#F6)
# =============================================================================

class DeliveryMonthValidator:
    """
    Validates delivery month matches between internal specs and IB (#F6).

    Prevents mismatches that could lead to wrong contract trading.
    """

    # Known IB contract month codes by symbol
    IB_CONTRACT_MONTHS: dict[str, tuple[str, ...]] = {
        "ES": ("H", "M", "U", "Z"),
        "NQ": ("H", "M", "U", "Z"),
        "YM": ("H", "M", "U", "Z"),
        "RTY": ("H", "M", "U", "Z"),
        "CL": ("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"),
        "NG": ("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"),
        "GC": ("G", "J", "M", "Q", "V", "Z"),
        "SI": ("H", "K", "N", "U", "Z"),
        "HG": ("H", "K", "N", "U", "Z"),
        "ZC": ("H", "K", "N", "U", "Z"),
        "ZW": ("H", "K", "N", "U", "Z"),
        "ZS": ("F", "H", "K", "N", "Q", "U", "X"),
        "ZB": ("H", "M", "U", "Z"),
        "ZN": ("H", "M", "U", "Z"),
        "ZF": ("H", "M", "U", "Z"),
        "6E": ("H", "M", "U", "Z"),
        "6J": ("H", "M", "U", "Z"),
        "6B": ("H", "M", "U", "Z"),
    }

    @classmethod
    def validate_contract_month(
        cls,
        symbol: str,
        month_code: str,
        source: str = "internal"
    ) -> dict:
        """
        Validate if month code is valid for symbol (#F6).

        Args:
            symbol: Contract symbol
            month_code: Month code to validate
            source: Source of the month code

        Returns:
            Validation result
        """
        ib_months = cls.IB_CONTRACT_MONTHS.get(symbol)
        schedule = EXTENDED_ROLL_SCHEDULES.get(symbol)

        result = {
            "symbol": symbol,
            "month_code": month_code,
            "source": source,
            "valid": False,
            "warnings": [],
        }

        if ib_months is None:
            result["warnings"].append(f"Unknown IB contract months for {symbol}")
        elif month_code not in ib_months:
            result["warnings"].append(f"Month {month_code} not valid for {symbol} in IB")
            result["valid_ib_months"] = ib_months
            return result

        if schedule is None:
            result["warnings"].append(f"No roll schedule defined for {symbol}")
        elif month_code not in schedule.contract_months:
            result["warnings"].append(f"Month {month_code} not in roll schedule for {symbol}")
            result["valid_schedule_months"] = schedule.contract_months
            return result

        # Both validations passed
        result["valid"] = True
        return result

    @classmethod
    def validate_roll_schedule(cls, symbol: str) -> dict:
        """
        Validate roll schedule matches IB contract months (#F6).

        Args:
            symbol: Contract symbol to validate

        Returns:
            Validation result with any mismatches
        """
        ib_months = cls.IB_CONTRACT_MONTHS.get(symbol)
        schedule = EXTENDED_ROLL_SCHEDULES.get(symbol)

        if ib_months is None:
            return {
                "symbol": symbol,
                "valid": False,
                "error": "no_ib_months_defined",
            }

        if schedule is None:
            return {
                "symbol": symbol,
                "valid": False,
                "error": "no_roll_schedule_defined",
            }

        ib_set = set(ib_months)
        schedule_set = set(schedule.contract_months)

        # Find mismatches
        in_ib_not_schedule = ib_set - schedule_set
        in_schedule_not_ib = schedule_set - ib_set

        return {
            "symbol": symbol,
            "valid": len(in_schedule_not_ib) == 0,  # All schedule months must be in IB
            "ib_months": ib_months,
            "schedule_months": schedule.contract_months,
            "in_ib_not_schedule": tuple(in_ib_not_schedule),
            "in_schedule_not_ib": tuple(in_schedule_not_ib),
            "warnings": [
                f"Schedule includes month {m} not tradeable in IB"
                for m in in_schedule_not_ib
            ],
        }

    @classmethod
    def validate_all_schedules(cls) -> list[dict]:
        """Validate all roll schedules against IB (#F6)."""
        results = []
        for symbol in EXTENDED_ROLL_SCHEDULES.keys():
            result = cls.validate_roll_schedule(symbol)
            if not result["valid"] or result.get("warnings"):
                results.append(result)
        return results


# =============================================================================
# EXPIRATION WARNING SYSTEM (#F7)
# =============================================================================

@dataclass
class ExpirationWarning:
    """Warning about approaching contract expiration (#F7)."""
    symbol: str
    contract_code: str
    days_to_expiry: int
    days_to_fnd: int | None
    warning_level: str  # "INFO", "WARNING", "CRITICAL"
    message: str
    timestamp: datetime
    position_size: int = 0
    recommended_action: str = ""


class ExpirationWarningSystem:
    """
    Monitors and alerts on approaching expirations (#F7).

    Provides tiered warnings:
    - INFO: 14+ days out
    - WARNING: 7-14 days out
    - CRITICAL: <7 days out
    """

    def __init__(
        self,
        warning_days: int = 14,
        critical_days: int = 7,
        fnd_buffer_days: int = 3
    ):
        self._warning_days = warning_days
        self._critical_days = critical_days
        self._fnd_buffer_days = fnd_buffer_days
        self._warnings: list[ExpirationWarning] = []
        self._acknowledged: set[str] = set()  # Contract codes acknowledged

    def check_expiration(
        self,
        symbol: str,
        contract_code: str,
        days_to_expiry: int,
        days_to_fnd: int | None = None,
        position_size: int = 0,
        is_physical_delivery: bool = False
    ) -> ExpirationWarning | None:
        """
        Check if expiration warning should be issued (#F7).

        Args:
            symbol: Base symbol
            contract_code: Full contract code
            days_to_expiry: Days until expiration
            days_to_fnd: Days until first notice (physical delivery)
            position_size: Current position size
            is_physical_delivery: Whether contract has physical delivery

        Returns:
            ExpirationWarning if warning needed
        """
        # Determine effective days (use FND for physical delivery)
        if is_physical_delivery and days_to_fnd is not None:
            effective_days = days_to_fnd
            expiry_type = "first_notice_date"
        else:
            effective_days = days_to_expiry
            expiry_type = "expiration"

        # Determine warning level
        if effective_days < 0:
            level = "CRITICAL"
            message = f"Contract {contract_code} has EXPIRED"
            action = "CLOSE POSITION IMMEDIATELY"
        elif effective_days < self._critical_days:
            level = "CRITICAL"
            message = f"Contract {contract_code} expires in {effective_days} days"
            action = "Roll or close position before expiry"
        elif effective_days < self._warning_days:
            level = "WARNING"
            message = f"Contract {contract_code} approaching {expiry_type} in {effective_days} days"
            action = "Plan roll strategy"
        elif effective_days < self._warning_days * 2:
            level = "INFO"
            message = f"Contract {contract_code} {expiry_type} in {effective_days} days"
            action = "Monitor position"
        else:
            return None  # No warning needed

        # Physical delivery gets escalated
        if is_physical_delivery and level != "CRITICAL" and effective_days < self._warning_days:
            level = "WARNING"
            message += " (PHYSICAL DELIVERY CONTRACT)"
            action = "Roll before first notice date"

        # Position-based escalation
        if position_size != 0 and level == "INFO":
            level = "WARNING"
            action = f"Position of {position_size} contracts needs attention"

        warning = ExpirationWarning(
            symbol=symbol,
            contract_code=contract_code,
            days_to_expiry=days_to_expiry,
            days_to_fnd=days_to_fnd,
            warning_level=level,
            message=message,
            timestamp=datetime.now(timezone.utc),
            position_size=position_size,
            recommended_action=action,
        )

        self._warnings.append(warning)
        return warning

    def check_all_positions(
        self,
        positions: dict[str, int],  # contract_code -> position_size
        roll_manager: FuturesRollManager
    ) -> list[ExpirationWarning]:
        """
        Check all positions for expiration warnings (#F7).

        Args:
            positions: Map of contract codes to position sizes
            roll_manager: Roll manager for contract info

        Returns:
            List of warnings
        """
        warnings = []

        for contract_code, size in positions.items():
            if size == 0:
                continue

            # Get contract info from roll manager
            parsed = parse_contract_code(contract_code)
            if not parsed:
                continue

            symbol, month, year = parsed
            contract = roll_manager._current_contracts.get(symbol)

            if contract:
                warning = self.check_expiration(
                    symbol=symbol,
                    contract_code=contract_code,
                    days_to_expiry=contract.days_to_expiry,
                    days_to_fnd=contract.days_to_fnd,
                    position_size=size,
                    is_physical_delivery=contract.is_physical_delivery,
                )
                if warning:
                    warnings.append(warning)

        return warnings

    def acknowledge_warning(self, contract_code: str) -> None:
        """Acknowledge a warning to suppress repeat alerts."""
        self._acknowledged.add(contract_code)

    def get_active_warnings(self, level: str | None = None) -> list[ExpirationWarning]:
        """
        Get active warnings, optionally filtered by level.

        Args:
            level: Filter by level ("INFO", "WARNING", "CRITICAL") or None for all

        Returns:
            List of active warnings
        """
        warnings = [w for w in self._warnings if w.contract_code not in self._acknowledged]
        if level:
            warnings = [w for w in warnings if w.warning_level == level]
        return warnings

    def get_summary(self) -> dict:
        """Get warning summary for monitoring."""
        active = self.get_active_warnings()
        return {
            "total_warnings": len(active),
            "critical": len([w for w in active if w.warning_level == "CRITICAL"]),
            "warning": len([w for w in active if w.warning_level == "WARNING"]),
            "info": len([w for w in active if w.warning_level == "INFO"]),
            "acknowledged": len(self._acknowledged),
        }


# =============================================================================
# SETTLEMENT PRICE HANDLING (#F8)
# =============================================================================

@dataclass
class SettlementPrice:
    """Settlement price data for a contract (#F8)."""
    symbol: str
    contract_code: str
    settlement_date: date
    settlement_price: float
    settlement_type: str  # "daily", "final"
    volume: int | None = None
    open_interest: int | None = None
    source: str = "exchange"


class SettlementPriceManager:
    """
    Manages settlement prices for futures (#F8).

    Handles:
    - Daily settlement prices for mark-to-market
    - Final settlement prices for expiring contracts
    - Settlement price history
    """

    def __init__(self):
        self._settlements: dict[str, list[SettlementPrice]] = {}  # contract_code -> history
        self._daily_settlements: dict[str, SettlementPrice] = {}  # contract_code -> latest
        self._final_settlements: dict[str, SettlementPrice] = {}

    def record_daily_settlement(
        self,
        symbol: str,
        contract_code: str,
        settlement_date: date,
        settlement_price: float,
        volume: int | None = None,
        open_interest: int | None = None
    ) -> SettlementPrice:
        """
        Record daily settlement price (#F8).

        Args:
            symbol: Base symbol
            contract_code: Full contract code
            settlement_date: Date of settlement
            settlement_price: Settlement price
            volume: Trading volume
            open_interest: Open interest

        Returns:
            SettlementPrice record
        """
        settlement = SettlementPrice(
            symbol=symbol,
            contract_code=contract_code,
            settlement_date=settlement_date,
            settlement_price=settlement_price,
            settlement_type="daily",
            volume=volume,
            open_interest=open_interest,
        )

        # Store in history
        if contract_code not in self._settlements:
            self._settlements[contract_code] = []
        self._settlements[contract_code].append(settlement)

        # Update latest
        self._daily_settlements[contract_code] = settlement

        return settlement

    def record_final_settlement(
        self,
        symbol: str,
        contract_code: str,
        settlement_date: date,
        settlement_price: float
    ) -> SettlementPrice:
        """
        Record final settlement price at expiration (#F8).

        Args:
            symbol: Base symbol
            contract_code: Full contract code
            settlement_date: Final settlement date
            settlement_price: Final settlement price

        Returns:
            SettlementPrice record
        """
        settlement = SettlementPrice(
            symbol=symbol,
            contract_code=contract_code,
            settlement_date=settlement_date,
            settlement_price=settlement_price,
            settlement_type="final",
        )

        self._final_settlements[contract_code] = settlement

        # Also add to history
        if contract_code not in self._settlements:
            self._settlements[contract_code] = []
        self._settlements[contract_code].append(settlement)

        logger.info(f"Final settlement recorded: {contract_code} @ {settlement_price}")

        return settlement

    def get_latest_settlement(self, contract_code: str) -> SettlementPrice | None:
        """Get most recent settlement price for a contract."""
        return self._daily_settlements.get(contract_code)

    def get_final_settlement(self, contract_code: str) -> SettlementPrice | None:
        """Get final settlement price if contract has expired."""
        return self._final_settlements.get(contract_code)

    def calculate_mark_to_market(
        self,
        contract_code: str,
        position: int,
        entry_price: float
    ) -> dict | None:
        """
        Calculate mark-to-market P&L using settlement price (#F8).

        Args:
            contract_code: Contract code
            position: Number of contracts (positive=long, negative=short)
            entry_price: Average entry price

        Returns:
            MTM calculation or None if no settlement
        """
        settlement = self.get_latest_settlement(contract_code)
        if settlement is None:
            return None

        # Get contract spec for multiplier
        parsed = parse_contract_code(contract_code)
        if parsed:
            symbol = parsed[0]
            spec = CONTRACT_SPECS.get(symbol)
            multiplier = spec.multiplier if spec else 1.0
        else:
            multiplier = 1.0

        price_change = settlement.settlement_price - entry_price
        mtm_pnl = position * price_change * multiplier

        return {
            "contract_code": contract_code,
            "position": position,
            "entry_price": entry_price,
            "settlement_price": settlement.settlement_price,
            "settlement_date": settlement.settlement_date.isoformat(),
            "price_change": price_change,
            "multiplier": multiplier,
            "mtm_pnl": mtm_pnl,
        }

    def get_settlement_history(
        self,
        contract_code: str,
        lookback_days: int = 30
    ) -> list[dict]:
        """Get settlement price history for a contract."""
        history = self._settlements.get(contract_code, [])
        cutoff = date.today() - timedelta(days=lookback_days)

        return [
            {
                "date": s.settlement_date.isoformat(),
                "price": s.settlement_price,
                "type": s.settlement_type,
                "volume": s.volume,
                "open_interest": s.open_interest,
            }
            for s in history
            if s.settlement_date >= cutoff
        ]

    def calculate_variation_margin(
        self,
        positions: dict[str, tuple[int, float]],  # contract_code -> (position, entry_price)
    ) -> dict:
        """
        Calculate total variation margin requirement (#F8).

        Args:
            positions: Map of contract codes to (position, entry_price) tuples

        Returns:
            Variation margin calculation
        """
        total_mtm = 0
        mtm_by_contract = {}

        for contract_code, (position, entry_price) in positions.items():
            mtm = self.calculate_mark_to_market(contract_code, position, entry_price)
            if mtm:
                mtm_by_contract[contract_code] = mtm
                total_mtm += mtm["mtm_pnl"]

        return {
            "total_variation_margin": total_mtm,
            "margin_call_needed": total_mtm < 0,
            "margin_excess": max(0, total_mtm),
            "by_contract": mtm_by_contract,
        }

    def get_status(self) -> dict:
        """Get manager status for monitoring."""
        return {
            "contracts_tracked": len(self._daily_settlements),
            "final_settlements": len(self._final_settlements),
            "total_settlement_records": sum(len(h) for h in self._settlements.values()),
        }
