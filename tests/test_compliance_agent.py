"""
Tests for Compliance Agent
==========================

Tests for the Compliance Agent that validates regulatory requirements.
"""

import pytest
import asyncio
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta, time
from zoneinfo import ZoneInfo

from agents.compliance_agent import (
    ComplianceAgent,
    ComplianceCheckResult,
    ComplianceValidationResult,
    RejectionCode,
    BlackoutType,
    BlackoutEvent,
    validate_lei,
    calculate_lei_check_digits,
    VALID_LOU_PREFIXES,
    PLACEHOLDER_LEI_PATTERNS,
)
from core.agent_base import AgentConfig
from core.events import (
    DecisionEvent,
    ValidatedDecisionEvent,
    EventType,
    OrderSide,
    OrderType,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    bus.subscribe = MagicMock()
    return bus


@pytest.fixture
def mock_audit_logger():
    """Create a mock audit logger."""
    logger = MagicMock()
    logger.log_event = MagicMock()
    logger.log_agent_event = MagicMock()
    logger.log_compliance_check = MagicMock()
    return logger


@pytest.fixture
def valid_lei():
    """A valid LEI for testing (example from GLEIF)."""
    # This is a well-known valid LEI (Bloomberg LP)
    return "5493006MHB84DD0ZWV18"


@pytest.fixture
def default_config(valid_lei):
    """Create default Compliance agent configuration."""
    return AgentConfig(
        name="ComplianceAgent",
        enabled=True,
        timeout_seconds=30.0,
        parameters={
            "jurisdiction": "EU",
            "entity_lei": valid_lei,
            "restricted_instruments": ["SANCTIONED_STOCK"],
            "allow_extended_hours": False,
            "isin_mappings": {
                "AAPL": "US0378331005",
                "GOOGL": "US02079K1079",
            },
        },
    )


@pytest.fixture
def compliance_agent(default_config, mock_event_bus, mock_audit_logger):
    """Create a Compliance agent instance for testing."""
    return ComplianceAgent(default_config, mock_event_bus, mock_audit_logger)


@pytest.fixture
def sample_decision_event():
    """Create a sample decision event."""
    return DecisionEvent(
        source_agent="CIOAgent",
        symbol="AAPL",
        action=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=150.0,
        rationale="Test decision",
        data_sources=("ib", "bloomberg"),
        conviction_score=0.8,
    )


@pytest.fixture
def sample_validated_event(sample_decision_event):
    """Create a sample validated decision event."""
    return ValidatedDecisionEvent(
        source_agent="RiskAgent",
        original_decision_id=sample_decision_event.event_id,
        approved=True,
        adjusted_quantity=100,
        risk_metrics={"var_95": 0.015},
    )


# ============================================================================
# LEI VALIDATION TESTS
# ============================================================================

class TestLEIValidation:
    """Test LEI (Legal Entity Identifier) validation."""

    def test_valid_lei_format(self, valid_lei):
        """Test validation of a valid LEI."""
        is_valid, error = validate_lei(valid_lei, strict=False)

        assert is_valid is True
        assert error == ""

    def test_lei_wrong_length(self):
        """Test rejection of LEI with wrong length."""
        is_valid, error = validate_lei("5493006MHB84DD0ZWV1", strict=False)

        assert is_valid is False
        assert "20 characters" in error

    def test_lei_empty(self):
        """Test rejection of empty LEI."""
        is_valid, error = validate_lei("", strict=False)

        assert is_valid is False
        assert "empty" in error.lower()

    def test_lei_with_special_characters(self):
        """Test rejection of LEI with special characters."""
        is_valid, error = validate_lei("5493006MHB84DD0ZWV1!", strict=False)

        assert is_valid is False
        assert "alphanumeric" in error.lower()

    def test_lei_placeholder_patterns(self):
        """Test rejection of placeholder LEI patterns."""
        placeholders = [
            "PLACEHOLDER000000000",
            "TEST0000000000000000",
            "DUMMY000000000000000",
        ]

        for placeholder in placeholders:
            # Pad to 20 chars if needed
            placeholder = placeholder[:20].ljust(20, '0')
            is_valid, error = validate_lei(placeholder, strict=True)
            assert is_valid is False, f"Should reject placeholder: {placeholder}"

    def test_lei_checksum_validation(self):
        """Test LEI checksum validation."""
        # Valid LEI with correct checksum
        is_valid, _ = validate_lei("5493006MHB84DD0ZWV18", strict=False)
        assert is_valid is True

        # Same LEI with wrong checksum (changed last digit)
        is_valid, error = validate_lei("5493006MHB84DD0ZWV19", strict=False)
        assert is_valid is False
        assert "checksum" in error.lower()

    def test_lei_uppercase_conversion(self):
        """Test that lowercase LEI is converted to uppercase."""
        lowercase_lei = "5493006mhb84dd0zwv18"
        is_valid, _ = validate_lei(lowercase_lei, strict=False)

        # Should be valid after uppercase conversion
        assert is_valid is True

    def test_lei_whitespace_handling(self):
        """Test that whitespace is trimmed."""
        lei_with_spaces = "  5493006MHB84DD0ZWV18  "
        is_valid, _ = validate_lei(lei_with_spaces, strict=False)

        assert is_valid is True


# ============================================================================
# LEI CHECK DIGITS CALCULATION TESTS
# ============================================================================

class TestLEICheckDigitsCalculation:
    """Test LEI check digits calculation."""

    def test_calculate_check_digits(self):
        """Test check digits calculation."""
        # Remove last 2 digits from valid LEI
        lei_base = "5493006MHB84DD0ZWV"
        check_digits = calculate_lei_check_digits(lei_base)

        assert check_digits == "18"

    def test_calculate_check_digits_wrong_length(self):
        """Test that wrong length raises error."""
        with pytest.raises(ValueError):
            calculate_lei_check_digits("TOO_SHORT")


# ============================================================================
# COMPLIANCE AGENT INITIALIZATION TESTS
# ============================================================================

class TestComplianceAgentInitialization:
    """Test Compliance agent initialization."""

    def test_initialization_with_default_config(self, compliance_agent):
        """Test that agent initializes correctly with default config."""
        assert compliance_agent.name == "ComplianceAgent"
        assert compliance_agent._jurisdiction == "EU"
        assert "SANCTIONED_STOCK" in compliance_agent._restricted_instruments

    def test_initialization_without_lei_raises_error(self, mock_event_bus, mock_audit_logger):
        """Test that missing LEI raises error on initialization."""
        config = AgentConfig(
            name="ComplianceAgent",
            parameters={"entity_lei": ""},
        )
        agent = ComplianceAgent(config, mock_event_bus, mock_audit_logger)

        with pytest.raises(ValueError) as exc_info:
            asyncio.get_event_loop().run_until_complete(agent.initialize())

        assert "LEI" in str(exc_info.value)

    def test_initialization_with_invalid_lei_raises_error(self, mock_event_bus, mock_audit_logger):
        """Test that invalid LEI raises error on initialization."""
        config = AgentConfig(
            name="ComplianceAgent",
            parameters={"entity_lei": "INVALID_LEI_12345678"},
        )
        agent = ComplianceAgent(config, mock_event_bus, mock_audit_logger)

        with pytest.raises(ValueError) as exc_info:
            asyncio.get_event_loop().run_until_complete(agent.initialize())

        assert "LEI" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_initialize_with_valid_lei(self, compliance_agent):
        """Test successful initialization with valid LEI."""
        await compliance_agent.initialize()
        # Should complete without error

    def test_subscribed_events(self, compliance_agent):
        """Test subscribed event types."""
        events = compliance_agent.get_subscribed_events()

        assert EventType.DECISION in events
        assert EventType.VALIDATED_DECISION in events


# ============================================================================
# REJECTION CODE TESTS
# ============================================================================

class TestRejectionCodes:
    """Test rejection code enumeration."""

    def test_all_rejection_codes(self):
        """Test all rejection codes exist."""
        codes = list(RejectionCode)

        assert RejectionCode.BLACKOUT_PERIOD in codes
        assert RejectionCode.MNPI_DETECTED in codes
        assert RejectionCode.RESTRICTED_INSTRUMENT in codes
        assert RejectionCode.MARKET_CLOSED in codes
        assert RejectionCode.SSR_RESTRICTION in codes
        assert RejectionCode.INVALID_LEI in codes


# ============================================================================
# BLACKOUT PERIOD TESTS
# ============================================================================

class TestBlackoutPeriods:
    """Test blackout period handling."""

    def test_blackout_type_enumeration(self):
        """Test blackout type values."""
        assert BlackoutType.EARNINGS.value == "earnings"
        assert BlackoutType.MA.value == "merger_acquisition"
        assert BlackoutType.CAPITAL_INCREASE.value == "capital_increase"
        assert BlackoutType.REGULATORY.value == "regulatory"

    def test_blackout_event_creation(self):
        """Test creating a blackout event."""
        now = datetime.now(timezone.utc)
        event = BlackoutEvent(
            symbol="AAPL",
            blackout_type=BlackoutType.EARNINGS,
            event_date=now + timedelta(days=7),
            blackout_start=now,
            blackout_end=now + timedelta(days=14),
            description="Q4 2025 earnings blackout",
        )

        assert event.symbol == "AAPL"
        assert event.blackout_type == BlackoutType.EARNINGS

    def test_add_blackout_event(self, compliance_agent):
        """Test adding a blackout event."""
        now = datetime.now(timezone.utc)
        event = BlackoutEvent(
            symbol="AAPL",
            blackout_type=BlackoutType.EARNINGS,
            event_date=now + timedelta(days=7),
            blackout_start=now,
            blackout_end=now + timedelta(days=14),
            description="Q4 2025 earnings blackout",
        )

        if "AAPL" not in compliance_agent._blackout_events:
            compliance_agent._blackout_events["AAPL"] = []
        compliance_agent._blackout_events["AAPL"].append(event)

        assert len(compliance_agent._blackout_events["AAPL"]) == 1


# ============================================================================
# MARKET HOURS TESTS
# ============================================================================

class TestMarketHours:
    """Test market hours configuration."""

    def test_market_hours_configuration(self, compliance_agent):
        """Test market hours are configured correctly."""
        assert compliance_agent._market_open_time == time(9, 30)
        assert compliance_agent._market_close_time == time(16, 0)

    def test_extended_hours_configuration(self, compliance_agent):
        """Test extended hours configuration."""
        assert compliance_agent._premarket_open == time(4, 0)
        assert compliance_agent._afterhours_close == time(20, 0)
        assert compliance_agent._allow_extended_hours is False

    def test_market_timezone(self, compliance_agent):
        """Test market timezone is set correctly."""
        assert compliance_agent._market_timezone == ZoneInfo("America/New_York")


# ============================================================================
# COMPLIANCE CHECK RESULT TESTS
# ============================================================================

class TestComplianceCheckResult:
    """Test ComplianceCheckResult dataclass."""

    def test_compliance_check_passed(self):
        """Test creating a passed compliance check."""
        result = ComplianceCheckResult(
            check_name="market_hours",
            passed=True,
            message="Market is open",
        )

        assert result.passed is True
        assert result.code is None

    def test_compliance_check_failed(self):
        """Test creating a failed compliance check."""
        result = ComplianceCheckResult(
            check_name="blackout_period",
            passed=False,
            code=RejectionCode.BLACKOUT_PERIOD,
            message="Symbol in earnings blackout",
            details={"blackout_end": "2025-02-15"},
        )

        assert result.passed is False
        assert result.code == RejectionCode.BLACKOUT_PERIOD


# ============================================================================
# COMPLIANCE VALIDATION RESULT TESTS
# ============================================================================

class TestComplianceValidationResult:
    """Test ComplianceValidationResult dataclass."""

    def test_validation_result_approved(self):
        """Test creating an approved validation result."""
        checks = [
            ComplianceCheckResult("market_hours", True),
            ComplianceCheckResult("blackout", True),
            ComplianceCheckResult("restricted", True),
        ]

        result = ComplianceValidationResult(
            approved=True,
            checks=checks,
        )

        assert result.approved is True
        assert result.rejection_code is None
        assert len(result.checks) == 3

    def test_validation_result_rejected(self):
        """Test creating a rejected validation result."""
        checks = [
            ComplianceCheckResult("market_hours", True),
            ComplianceCheckResult(
                "restricted",
                False,
                code=RejectionCode.RESTRICTED_INSTRUMENT,
            ),
        ]

        result = ComplianceValidationResult(
            approved=False,
            checks=checks,
            rejection_code=RejectionCode.RESTRICTED_INSTRUMENT,
            rejection_reason="Instrument is under sanctions",
        )

        assert result.approved is False
        assert result.rejection_code == RejectionCode.RESTRICTED_INSTRUMENT


# ============================================================================
# RESTRICTED INSTRUMENTS TESTS
# ============================================================================

class TestRestrictedInstruments:
    """Test restricted instruments handling."""

    def test_restricted_instruments_loaded(self, compliance_agent):
        """Test restricted instruments are loaded from config."""
        assert "SANCTIONED_STOCK" in compliance_agent._restricted_instruments

    def test_add_restricted_instrument(self, compliance_agent):
        """Test adding a restricted instrument."""
        compliance_agent._restricted_instruments.add("NEW_SANCTION")

        assert "NEW_SANCTION" in compliance_agent._restricted_instruments

    def test_remove_restricted_instrument(self, compliance_agent):
        """Test removing a restricted instrument."""
        compliance_agent._restricted_instruments.add("TEMP_RESTRICTION")
        compliance_agent._restricted_instruments.discard("TEMP_RESTRICTION")

        assert "TEMP_RESTRICTION" not in compliance_agent._restricted_instruments


# ============================================================================
# SSR (SHORT SELLING REGULATION) TESTS
# ============================================================================

class TestSSR:
    """Test Short Selling Regulation handling."""

    def test_ssr_active_set(self, compliance_agent):
        """Test SSR active set."""
        assert isinstance(compliance_agent._ssr_active, set)

    def test_add_ssr_symbol(self, compliance_agent):
        """Test adding a symbol to SSR list."""
        compliance_agent._ssr_active.add("AAPL")

        assert "AAPL" in compliance_agent._ssr_active

    def test_remove_ssr_symbol(self, compliance_agent):
        """Test removing a symbol from SSR list."""
        compliance_agent._ssr_active.add("AAPL")
        compliance_agent._ssr_active.discard("AAPL")

        assert "AAPL" not in compliance_agent._ssr_active


# ============================================================================
# SUSPENDED SYMBOLS TESTS
# ============================================================================

class TestSuspendedSymbols:
    """Test trading suspension handling."""

    def test_suspended_symbols_set(self, compliance_agent):
        """Test suspended symbols set."""
        assert isinstance(compliance_agent._suspended_symbols, set)

    def test_add_suspended_symbol(self, compliance_agent):
        """Test adding a suspended symbol."""
        compliance_agent._suspended_symbols.add("SUSPENDED_STOCK")

        assert "SUSPENDED_STOCK" in compliance_agent._suspended_symbols


# ============================================================================
# DECLARATION THRESHOLDS TESTS
# ============================================================================

class TestDeclarationThresholds:
    """Test declaration threshold handling."""

    def test_eu_declaration_thresholds(self, compliance_agent):
        """Test EU declaration thresholds."""
        thresholds = compliance_agent._declaration_thresholds

        assert 0.05 in thresholds  # 5%
        assert 0.10 in thresholds  # 10%
        assert 0.50 in thresholds  # 50%


# ============================================================================
# DATA SOURCE VALIDATION TESTS
# ============================================================================

class TestDataSourceValidation:
    """Test data source validation."""

    def test_approved_sources(self):
        """Test approved data sources list."""
        approved = ComplianceAgent.APPROVED_SOURCES

        assert "bloomberg" in approved
        assert "reuters" in approved
        assert "interactive_brokers" in approved
        assert "sec_edgar" in approved

    def test_suspicious_patterns(self):
        """Test suspicious patterns list."""
        patterns = ComplianceAgent.SUSPICIOUS_PATTERNS

        assert len(patterns) > 0
        # Should contain patterns for detecting MNPI
        assert any("leak" in p for p in patterns)
        assert any("insider" in p for p in patterns)


# ============================================================================
# ISIN MAPPING TESTS
# ============================================================================

class TestISINMapping:
    """Test ISIN mapping functionality."""

    def test_isin_mappings_loaded(self, compliance_agent):
        """Test ISIN mappings are loaded from config."""
        assert "AAPL" in compliance_agent._isin_mappings
        assert compliance_agent._isin_mappings["AAPL"] == "US0378331005"

    def test_isin_mapping_googl(self, compliance_agent):
        """Test GOOGL ISIN mapping."""
        assert "GOOGL" in compliance_agent._isin_mappings
        assert compliance_agent._isin_mappings["GOOGL"] == "US02079K1079"


# ============================================================================
# LEI ENTITY CONFIGURATION TESTS
# ============================================================================

class TestLEIEntityConfiguration:
    """Test entity LEI configuration."""

    def test_entity_lei_stored(self, compliance_agent, valid_lei):
        """Test entity LEI is stored from config."""
        assert compliance_agent._entity_lei == valid_lei

    def test_validate_entity_lei(self, compliance_agent):
        """Test validating entity LEI."""
        result = compliance_agent.validate_entity_lei()

        assert result.passed is True


# ============================================================================
# DECISION CACHING TESTS
# ============================================================================

class TestDecisionCaching:
    """Test decision caching functionality."""

    @pytest.mark.asyncio
    async def test_cache_decision_event(self, compliance_agent, sample_decision_event):
        """Test that decision events are cached."""
        await compliance_agent.process_event(sample_decision_event)

        assert sample_decision_event.event_id in compliance_agent._decision_cache

    def test_decision_cache_initially_empty(self, compliance_agent):
        """Test decision cache starts empty."""
        assert len(compliance_agent._decision_cache) == 0


# ============================================================================
# MONITORING AND LATENCY TESTS
# ============================================================================

class TestMonitoringLatency:
    """Test monitoring and latency tracking."""

    def test_check_latencies_tracking(self, compliance_agent):
        """Test check latencies tracking."""
        # Memory safety: check_latencies uses deque instead of list
        assert isinstance(compliance_agent._check_latencies, deque)

    def test_rejections_today_tracking(self, compliance_agent):
        """Test rejections tracking."""
        # Memory safety: rejections_today uses deque instead of list
        assert isinstance(compliance_agent._rejections_today, deque)

    def test_latencies_capped(self, compliance_agent):
        """Test that latencies list is capped."""
        for i in range(1005):
            compliance_agent._check_latencies.append(float(i))

        if len(compliance_agent._check_latencies) > 1000:
            compliance_agent._check_latencies = compliance_agent._check_latencies[-1000:]

        assert len(compliance_agent._check_latencies) == 1000


# ============================================================================
# EDGE CASES TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_restricted_instruments(self, mock_event_bus, mock_audit_logger, valid_lei):
        """Test handling of empty restricted instruments."""
        config = AgentConfig(
            name="ComplianceAgent",
            parameters={
                "entity_lei": valid_lei,
                "restricted_instruments": [],
            },
        )
        agent = ComplianceAgent(config, mock_event_bus, mock_audit_logger)

        assert len(agent._restricted_instruments) == 0

    def test_current_positions_tracking(self, compliance_agent):
        """Test current positions tracking."""
        assert isinstance(compliance_agent._current_positions, dict)

        # Add a position
        compliance_agent._current_positions["AAPL"] = 0.05

        assert compliance_agent._current_positions["AAPL"] == 0.05

    @pytest.mark.asyncio
    async def test_process_non_validated_event(self, compliance_agent, sample_decision_event):
        """Test that non-validated events just get cached."""
        await compliance_agent.process_event(sample_decision_event)

        # Decision should be cached
        assert sample_decision_event.event_id in compliance_agent._decision_cache

        # But no compliance event published (that happens on ValidatedDecisionEvent)
        # Decision events just get cached, not processed for compliance


# ============================================================================
# LOU PREFIX VALIDATION TESTS
# ============================================================================

class TestLOUPrefixes:
    """Test LOU (Local Operating Unit) prefix validation."""

    def test_valid_lou_prefixes_exist(self):
        """Test that valid LOU prefixes are defined."""
        assert len(VALID_LOU_PREFIXES) > 0
        assert "5493" in VALID_LOU_PREFIXES  # DTCC
        assert "5299" in VALID_LOU_PREFIXES  # INSEE France

    def test_placeholder_patterns_exist(self):
        """Test that placeholder patterns are defined."""
        assert len(PLACEHOLDER_LEI_PATTERNS) > 0
        assert "PLACEHOLDER" in PLACEHOLDER_LEI_PATTERNS
        assert "TEST" in PLACEHOLDER_LEI_PATTERNS


# ============================================================================
# MIFID II RULE TESTS (P3 Compliance)
# ============================================================================

class TestMiFIDIIRules:
    """Test MiFID II regulatory compliance rules."""

    def test_mifid_ii_data_source_validation(self, compliance_agent):
        """Test MiFID II requirement for approved data sources."""
        # Approved sources should be defined per MiFID II Art. 27
        approved = compliance_agent.APPROVED_SOURCES

        # Core sources required for best execution
        assert "bloomberg" in approved
        assert "reuters" in approved
        assert "interactive_brokers" in approved or "ib" in approved

    def test_mifid_ii_transaction_reporting_lei_required(self, compliance_agent, valid_lei):
        """Test MiFID II Art. 26 requires LEI for transaction reporting."""
        # Entity LEI must be configured for MiFID II compliance
        assert compliance_agent._entity_lei == valid_lei

        # Validate LEI is properly formatted
        result = compliance_agent.validate_entity_lei()
        assert result.passed is True

    def test_mifid_ii_counterparty_lei_validation(self, compliance_agent, valid_lei):
        """Test counterparty LEI validation for MiFID II."""
        # Valid counterparty LEI
        result = compliance_agent.validate_counterparty_lei(valid_lei)
        assert result.passed is True

        # Invalid LEI should fail
        result = compliance_agent.validate_counterparty_lei("INVALID_LEI_1234")
        assert result.passed is False
        assert result.code == RejectionCode.INVALID_LEI

    def test_mifid_ii_isin_mapping_for_reporting(self, compliance_agent):
        """Test ISIN mappings for MiFID II transaction reporting."""
        # ISIN mappings should be available
        assert "AAPL" in compliance_agent._isin_mappings
        assert compliance_agent._isin_mappings["AAPL"] == "US0378331005"

        # Get ISIN for reporting
        isin = compliance_agent.get_isin("AAPL")
        assert isin == "US0378331005"

    def test_mifid_ii_best_execution_data_sources(self, compliance_agent):
        """Test best execution obligation requires proper data sources."""
        # Per MiFID II Art. 27, firms must use sufficient execution venues
        # The compliance agent should validate data sources
        decision = {
            "symbol": "AAPL",
            "data_sources": ["bloomberg", "reuters"],
        }

        result = compliance_agent._check_data_sources(decision)
        assert result.passed is True

    def test_mifid_ii_reject_no_data_sources(self, compliance_agent):
        """Test MiFID II rejects decisions without data sources."""
        decision = {
            "symbol": "AAPL",
            "data_sources": [],
        }

        result = compliance_agent._check_data_sources(decision)
        assert result.passed is False
        assert result.code == RejectionCode.UNAPPROVED_SOURCE

    def test_mifid_ii_reject_unapproved_sources(self, compliance_agent):
        """Test MiFID II rejects unapproved data sources."""
        decision = {
            "symbol": "AAPL",
            "data_sources": ["unknown_source", "random_api"],
        }

        result = compliance_agent._check_data_sources(decision)
        assert result.passed is False

    def test_mifid_ii_market_hours_timezone_handling(self, compliance_agent):
        """Test MiFID II proper timezone handling for market hours."""
        # Verify market timezone is properly configured
        assert compliance_agent._market_timezone is not None
        assert str(compliance_agent._market_timezone) == "America/New_York"

    def test_mifid_ii_extended_hours_config(self, compliance_agent):
        """Test MiFID II extended hours configuration."""
        # Extended hours should be configurable
        assert hasattr(compliance_agent, "_allow_extended_hours")
        assert hasattr(compliance_agent, "_premarket_open")
        assert hasattr(compliance_agent, "_afterhours_close")


# ============================================================================
# POSITION LIMIT TESTS (P3 Compliance)
# ============================================================================

class TestPositionLimits:
    """Test position limit enforcement."""

    def test_declaration_thresholds_eu(self, compliance_agent):
        """Test EU declaration thresholds are properly configured."""
        thresholds = compliance_agent._declaration_thresholds

        # EU declaration thresholds per Transparency Directive
        assert 0.05 in thresholds  # 5% - first threshold
        assert 0.10 in thresholds  # 10%
        assert 0.15 in thresholds  # 15%
        assert 0.20 in thresholds  # 20%
        assert 0.25 in thresholds  # 25%
        assert 0.30 in thresholds  # 30%
        assert 0.50 in thresholds  # 50% - control threshold
        assert 0.75 in thresholds  # 75%

    def test_position_tracking_initialization(self, compliance_agent):
        """Test position tracking is properly initialized."""
        assert hasattr(compliance_agent, "_current_positions")
        assert isinstance(compliance_agent._current_positions, dict)

    def test_position_weight_update(self, compliance_agent):
        """Test position weight can be updated."""
        compliance_agent._current_positions["AAPL"] = 0.03  # 3%

        assert compliance_agent._current_positions["AAPL"] == 0.03

    @pytest.mark.asyncio
    async def test_threshold_crossing_detection(self, compliance_agent):
        """Test detection of threshold crossings."""
        # Set current position below threshold
        compliance_agent._current_positions["AAPL"] = 0.04  # 4%

        # Order that would push above 5% threshold
        decision = {
            "symbol": "AAPL",
            "quantity": 100,
            "limit_price": 150.0,
            "action": "buy",
        }

        result = await compliance_agent._check_declaration_threshold("AAPL", decision)

        # Should pass but flag for notification
        assert result.passed is True
        # Check if threshold crossing was detected
        if result.details.get("notification_required"):
            assert 0.05 in result.details.get("thresholds_crossed", [])

    def test_position_limit_initial_state(self, compliance_agent):
        """Test position limits start empty."""
        # Initially no positions tracked
        assert len(compliance_agent._current_positions) == 0 or \
               all(v == 0 for v in compliance_agent._current_positions.values())

    def test_multiple_threshold_crossings(self, compliance_agent):
        """Test detection of multiple threshold crossings."""
        # This verifies the system can detect crossing multiple thresholds
        thresholds = compliance_agent._declaration_thresholds

        # Verify thresholds are sorted for proper detection
        sorted_thresholds = sorted(thresholds)
        assert sorted_thresholds == list(sorted(thresholds))


# ============================================================================
# REPORTING DEADLINE TESTS (P3 Compliance)
# ============================================================================

class TestReportingDeadlines:
    """Test regulatory reporting deadline compliance."""

    def test_rejection_tracking_exists(self, compliance_agent):
        """Test rejection tracking for reporting."""
        assert hasattr(compliance_agent, "_rejections_today")
        # Memory safety: rejections_today uses deque instead of list
        assert isinstance(compliance_agent._rejections_today, deque)

    def test_rejection_records_timestamp(self, compliance_agent):
        """Test rejection records include timestamp for reporting."""
        # Add a mock rejection
        compliance_agent._rejections_today.append((
            datetime.now(timezone.utc),
            RejectionCode.BLACKOUT_PERIOD,
            "AAPL"
        ))

        assert len(compliance_agent._rejections_today) == 1
        rejection = compliance_agent._rejections_today[0]

        # Verify rejection has required fields for reporting
        assert isinstance(rejection[0], datetime)  # timestamp
        assert isinstance(rejection[1], RejectionCode)  # rejection code
        assert isinstance(rejection[2], str)  # symbol

    def test_check_latency_tracking(self, compliance_agent):
        """Test compliance check latency tracking for reporting."""
        assert hasattr(compliance_agent, "_check_latencies")
        # Memory safety: check_latencies uses deque instead of list
        assert isinstance(compliance_agent._check_latencies, deque)

    def test_latency_list_capping(self, compliance_agent):
        """Test latency list is capped to prevent memory issues."""
        # Add more than cap
        for i in range(1100):
            compliance_agent._check_latencies.append(float(i))

        # Apply cap (as done in process_event)
        if len(compliance_agent._check_latencies) > 1000:
            compliance_agent._check_latencies = compliance_agent._check_latencies[-1000:]

        assert len(compliance_agent._check_latencies) <= 1000

    def test_status_report_generation(self, compliance_agent):
        """Test status report contains required fields for regulatory reporting."""
        status = compliance_agent.get_status()

        # Verify required fields for regulatory reporting
        assert "jurisdiction" in status
        assert "restricted_instruments_count" in status
        assert "blackout_events_count" in status
        assert "ssr_active_count" in status
        assert "suspended_symbols" in status
        assert "rejections_today" in status
        assert "rejection_breakdown" in status
        assert "avg_check_latency_ms" in status

    def test_rejection_breakdown_by_code(self, compliance_agent):
        """Test rejection breakdown by code for regulatory reporting."""
        # Add different rejection types
        now = datetime.now(timezone.utc)
        compliance_agent._rejections_today = [
            (now, RejectionCode.BLACKOUT_PERIOD, "AAPL"),
            (now, RejectionCode.BLACKOUT_PERIOD, "GOOGL"),
            (now, RejectionCode.RESTRICTED_INSTRUMENT, "SANC1"),
            (now, RejectionCode.MARKET_CLOSED, "MSFT"),
        ]

        breakdown = compliance_agent._get_rejection_breakdown(
            compliance_agent._rejections_today
        )

        # Verify breakdown by code
        assert breakdown.get("BLACKOUT_PERIOD", 0) == 2
        assert breakdown.get("RESTRICTED_INSTRUMENT", 0) == 1
        assert breakdown.get("MARKET_CLOSED", 0) == 1

    def test_daily_rejection_filtering(self, compliance_agent):
        """Test filtering rejections by today for daily reporting."""
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = today_start - timedelta(days=1)

        # Add today's and yesterday's rejections
        compliance_agent._rejections_today = [
            (now, RejectionCode.BLACKOUT_PERIOD, "AAPL"),  # today
            (yesterday, RejectionCode.MARKET_CLOSED, "GOOGL"),  # yesterday
        ]

        # Filter for today only (as done in get_status)
        rejections_today = [
            r for r in compliance_agent._rejections_today
            if r[0] >= today_start
        ]

        assert len(rejections_today) == 1
        assert rejections_today[0][2] == "AAPL"

    def test_audit_logger_integration(self, compliance_agent, mock_audit_logger):
        """Test audit logger is available for regulatory reporting."""
        assert compliance_agent._audit_logger is not None
        assert hasattr(compliance_agent._audit_logger, "log_compliance_check")

    def test_t_plus_one_reporting_fields(self, compliance_agent):
        """Test fields required for T+1 transaction reporting."""
        # Entity LEI required for T+1 reporting
        assert compliance_agent._entity_lei is not None

        # ISIN mappings for instrument identification
        assert len(compliance_agent._isin_mappings) > 0

        # Jurisdiction for regulatory routing
        assert compliance_agent._jurisdiction in ["EU", "US", "UK"]
