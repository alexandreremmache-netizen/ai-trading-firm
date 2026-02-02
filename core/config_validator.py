"""
Configuration Validation Module
===============================

Comprehensive configuration validation at startup (Issue #S8).

Features:
- Schema-based validation
- Type checking
- Range validation
- Cross-field validation
- Environment-specific validation
- Clear error messages
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Validation issue severity."""
    ERROR = "error"  # Must fix before running
    WARNING = "warning"  # Should review
    INFO = "info"  # Informational


@dataclass
class ValidationIssue:
    """Single validation issue."""
    severity: ValidationSeverity
    path: str  # Config path (e.g., "risk.max_position_pct")
    message: str
    suggestion: str | None = None

    def __str__(self) -> str:
        prefix = f"[{self.severity.value.upper()}]"
        msg = f"{prefix} {self.path}: {self.message}"
        if self.suggestion:
            msg += f" Suggestion: {self.suggestion}"
        return msg


@dataclass
class ValidationResult:
    """Complete validation result."""
    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_error(self, path: str, message: str, suggestion: str | None = None) -> None:
        """Add error issue."""
        self.issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            path=path,
            message=message,
            suggestion=suggestion,
        ))
        self.valid = False

    def add_warning(self, path: str, message: str, suggestion: str | None = None) -> None:
        """Add warning issue."""
        self.issues.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            path=path,
            message=message,
            suggestion=suggestion,
        ))

    def add_info(self, path: str, message: str) -> None:
        """Add info issue."""
        self.issues.append(ValidationIssue(
            severity=ValidationSeverity.INFO,
            path=path,
            message=message,
        ))

    def get_errors(self) -> list[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    def get_warnings(self) -> list[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def summary(self) -> str:
        """Get validation summary."""
        errors = len(self.get_errors())
        warnings = len(self.get_warnings())
        status = "VALID" if self.valid else "INVALID"
        return f"Config validation: {status} ({errors} errors, {warnings} warnings)"


@dataclass
class FieldSchema:
    """Schema for a single config field."""
    path: str
    field_type: type | tuple[type, ...]
    required: bool = True
    default: Any = None
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list | None = None
    pattern: str | None = None  # Regex pattern
    validator: Callable[[Any], bool] | None = None
    description: str = ""


class ConfigValidator:
    """
    Configuration validator with schema support.

    Validates config at startup to catch issues early.
    """

    def __init__(self):
        self._schemas: list[FieldSchema] = []
        self._cross_validators: list[Callable[[dict, ValidationResult], None]] = []

        # Register default schemas
        self._register_default_schemas()

    def _register_default_schemas(self) -> None:
        """Register default configuration schemas."""

        # Broker settings
        self.add_schema(FieldSchema(
            path="broker.host",
            field_type=str,
            required=True,
            default="127.0.0.1",
            description="IB Gateway/TWS host address",
        ))
        self.add_schema(FieldSchema(
            path="broker.port",
            field_type=int,
            required=True,
            min_value=1,
            max_value=65535,
            description="IB Gateway/TWS port",
        ))
        self.add_schema(FieldSchema(
            path="broker.client_id",
            field_type=int,
            required=True,
            min_value=1,
            max_value=999,
            description="IB client ID (must be unique per connection)",
        ))
        self.add_schema(FieldSchema(
            path="broker.paper_trading",
            field_type=bool,
            required=False,
            default=True,
            description="Use paper trading mode",
        ))

        # Risk settings
        self.add_schema(FieldSchema(
            path="risk.max_position_pct",
            field_type=(int, float),
            required=True,
            min_value=0.1,
            max_value=100,
            description="Maximum position size as % of portfolio",
        ))
        self.add_schema(FieldSchema(
            path="risk.max_sector_pct",
            field_type=(int, float),
            required=True,
            min_value=1,
            max_value=100,
            description="Maximum sector concentration",
        ))
        self.add_schema(FieldSchema(
            path="risk.max_leverage",
            field_type=(int, float),
            required=True,
            min_value=1,
            max_value=10,
            description="Maximum leverage ratio",
        ))
        self.add_schema(FieldSchema(
            path="risk.var_confidence",
            field_type=float,
            required=False,
            default=0.95,
            min_value=0.90,
            max_value=0.99,
            description="VaR confidence level",
        ))

        # Compliance settings
        self.add_schema(FieldSchema(
            path="compliance.firm_lei",
            field_type=str,
            required=True,
            pattern=r"^[A-Z0-9]{20}$",
            validator=self._validate_lei,
            description="Firm LEI (Legal Entity Identifier)",
        ))
        self.add_schema(FieldSchema(
            path="compliance.jurisdiction",
            field_type=str,
            required=True,
            allowed_values=["EU", "US", "UK", "APAC"],
            description="Regulatory jurisdiction",
        ))

        # Execution settings
        self.add_schema(FieldSchema(
            path="execution.max_slippage_bps",
            field_type=(int, float),
            required=False,
            default=50,
            min_value=1,
            max_value=500,
            description="Maximum allowed slippage in basis points",
        ))
        self.add_schema(FieldSchema(
            path="execution.default_algo",
            field_type=str,
            required=False,
            default="TWAP",
            allowed_values=["MARKET", "LIMIT", "TWAP", "VWAP"],
            description="Default execution algorithm",
        ))

        # Portfolio settings
        self.add_schema(FieldSchema(
            path="portfolio.initial_capital",
            field_type=(int, float),
            required=True,
            min_value=10000,
            description="Initial portfolio capital",
        ))
        self.add_schema(FieldSchema(
            path="portfolio.base_currency",
            field_type=str,
            required=False,
            default="USD",
            pattern=r"^[A-Z]{3}$",
            description="Base currency for portfolio",
        ))

        # Strategy settings
        self.add_schema(FieldSchema(
            path="strategies.enabled",
            field_type=list,
            required=True,
            validator=lambda x: len(x) > 0,
            description="List of enabled strategies",
        ))

        # Register cross-validators
        self.add_cross_validator(self._validate_risk_consistency)
        self.add_cross_validator(self._validate_environment_settings)

    def add_schema(self, schema: FieldSchema) -> None:
        """Add field schema."""
        self._schemas.append(schema)

    def add_cross_validator(
        self,
        validator: Callable[[dict, ValidationResult], None],
    ) -> None:
        """Add cross-field validator."""
        self._cross_validators.append(validator)

    def validate(self, config: dict) -> ValidationResult:
        """
        Validate configuration against schemas.

        Args:
            config: Configuration dictionary

        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult(valid=True)

        # Validate each schema
        for schema in self._schemas:
            self._validate_field(config, schema, result)

        # Run cross-validators
        for validator in self._cross_validators:
            validator(config, result)

        # Log result
        if result.valid:
            logger.info(result.summary())
        else:
            logger.error(result.summary())
            for issue in result.get_errors():
                logger.error(str(issue))
            for issue in result.get_warnings():
                logger.warning(str(issue))

        return result

    def _validate_field(
        self,
        config: dict,
        schema: FieldSchema,
        result: ValidationResult,
    ) -> None:
        """Validate single field against schema."""
        value = self._get_nested_value(config, schema.path)

        # Check required
        if value is None:
            if schema.required:
                result.add_error(
                    schema.path,
                    "Required field is missing",
                    f"Add '{schema.path}' to your configuration",
                )
            return

        # Check type
        if not isinstance(value, schema.field_type):
            result.add_error(
                schema.path,
                f"Invalid type: expected {schema.field_type.__name__ if isinstance(schema.field_type, type) else schema.field_type}, got {type(value).__name__}",
            )
            return

        # Check numeric range
        if schema.min_value is not None and isinstance(value, (int, float)):
            if value < schema.min_value:
                result.add_error(
                    schema.path,
                    f"Value {value} is below minimum {schema.min_value}",
                )

        if schema.max_value is not None and isinstance(value, (int, float)):
            if value > schema.max_value:
                result.add_error(
                    schema.path,
                    f"Value {value} is above maximum {schema.max_value}",
                )

        # Check allowed values
        if schema.allowed_values is not None:
            if value not in schema.allowed_values:
                result.add_error(
                    schema.path,
                    f"Value '{value}' not in allowed values: {schema.allowed_values}",
                )

        # Check pattern
        if schema.pattern is not None and isinstance(value, str):
            if not re.match(schema.pattern, value):
                result.add_error(
                    schema.path,
                    f"Value '{value}' does not match required pattern: {schema.pattern}",
                )

        # Custom validator
        if schema.validator is not None:
            try:
                if not schema.validator(value):
                    result.add_error(
                        schema.path,
                        "Failed custom validation",
                    )
            except Exception as e:
                result.add_error(
                    schema.path,
                    f"Validation error: {e}",
                )

    def _get_nested_value(self, config: dict, path: str) -> Any:
        """Get value from nested dict using dot notation."""
        keys = path.split(".")
        value = config

        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
            if value is None:
                return None

        return value

    def _validate_lei(self, lei: str) -> bool:
        """Validate LEI format and checksum."""
        if not lei or len(lei) != 20:
            return False

        # Check for placeholder
        if "PLACEHOLDER" in lei.upper():
            return False

        # Basic format check
        if not re.match(r"^[A-Z0-9]{20}$", lei):
            return False

        return True

    def _validate_risk_consistency(
        self,
        config: dict,
        result: ValidationResult,
    ) -> None:
        """Cross-validate risk settings."""
        max_position = self._get_nested_value(config, "risk.max_position_pct")
        max_sector = self._get_nested_value(config, "risk.max_sector_pct")

        if max_position and max_sector:
            if max_position > max_sector:
                result.add_warning(
                    "risk",
                    f"max_position_pct ({max_position}%) > max_sector_pct ({max_sector}%)",
                    "Position limit should typically be <= sector limit",
                )

        # Check leverage vs margin
        max_leverage = self._get_nested_value(config, "risk.max_leverage")
        if max_leverage and max_leverage > 2:
            result.add_warning(
                "risk.max_leverage",
                f"High leverage ({max_leverage}x) configured",
                "Consider lower leverage for reduced risk",
            )

    def _validate_environment_settings(
        self,
        config: dict,
        result: ValidationResult,
    ) -> None:
        """Validate environment-specific settings."""
        paper_trading = self._get_nested_value(config, "broker.paper_trading")
        environment = os.environ.get("TRADING_ENV", "development")

        if environment == "production":
            if paper_trading:
                result.add_warning(
                    "broker.paper_trading",
                    "Paper trading enabled in production environment",
                    "Disable paper trading for live trading",
                )

            # Additional production checks
            lei = self._get_nested_value(config, "compliance.firm_lei")
            if lei and "PLACEHOLDER" in lei.upper():
                result.add_error(
                    "compliance.firm_lei",
                    "Placeholder LEI in production",
                    "Configure real LEI before going live",
                )

        if environment == "development":
            if not paper_trading:
                result.add_warning(
                    "broker.paper_trading",
                    "Live trading in development environment",
                    "Enable paper trading for development",
                )


class ConfigMigrator:
    """
    Handles configuration migrations between versions.

    Helps upgrade old configs to new format.
    """

    def __init__(self):
        self._migrations: list[tuple[str, str, Callable[[dict], dict]]] = []

    def add_migration(
        self,
        from_version: str,
        to_version: str,
        migrator: Callable[[dict], dict],
    ) -> None:
        """Add migration function."""
        self._migrations.append((from_version, to_version, migrator))

    def migrate(self, config: dict, target_version: str) -> dict:
        """Migrate config to target version."""
        current_version = config.get("version", "1.0.0")

        if current_version == target_version:
            return config

        # Find migration path
        migrated = config.copy()

        for from_ver, to_ver, migrator in self._migrations:
            if from_ver == current_version:
                logger.info(f"Migrating config from {from_ver} to {to_ver}")
                migrated = migrator(migrated)
                migrated["version"] = to_ver
                current_version = to_ver

                if current_version == target_version:
                    break

        return migrated


def validate_config_at_startup(config: dict) -> bool:
    """
    Convenience function to validate config at startup.

    Returns True if valid, raises on critical errors.
    """
    validator = ConfigValidator()
    result = validator.validate(config)

    if not result.valid:
        errors = result.get_errors()
        error_messages = "\n".join(str(e) for e in errors)
        raise ValueError(f"Configuration validation failed:\n{error_messages}")

    return True
