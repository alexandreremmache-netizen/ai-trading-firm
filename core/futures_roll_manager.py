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
