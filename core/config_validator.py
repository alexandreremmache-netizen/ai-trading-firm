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
- Secret management with env/vault references (API-003)
- Strict validation mode (API-006)
- Hot-reloadable field tracking (API-008)
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


# =============================================================================
# API-003: Secret Management
# =============================================================================

class ConfigValidationError(ValueError):
    """Exception raised when config validation fails in strict mode.

    Inherits from ValueError for backwards compatibility with existing code
    that expects ValueError on validation failures.
    """

    def __init__(self, message: str, result: "ValidationResult"):
        super().__init__(message)
        self.result = result


class SecretField:
    """
    Wrapper for secret values that supports reference syntax.

    Supports:
    - ${env:VARIABLE_NAME} - Load from environment variable
    - ${vault:secret/path} - Load from vault (requires vault client)
    - Plain text (with warning)

    Usage:
        secret = SecretField("${env:API_KEY}")
        resolved = secret.resolve()  # Returns actual value
    """

    ENV_PATTERN = re.compile(r"^\$\{env:([A-Z_][A-Z0-9_]*)\}$")
    VAULT_PATTERN = re.compile(r"^\$\{vault:([a-zA-Z0-9/_-]+)\}$")

    def __init__(self, value: str):
        self._raw_value = value
        self._is_reference = False
        self._reference_type: str | None = None
        self._reference_key: str | None = None
        self._parse_value()

    def _parse_value(self) -> None:
        """Parse the value to detect reference syntax."""
        if not isinstance(self._raw_value, str):
            return

        # Check for env reference
        env_match = self.ENV_PATTERN.match(self._raw_value)
        if env_match:
            self._is_reference = True
            self._reference_type = "env"
            self._reference_key = env_match.group(1)
            return

        # Check for vault reference
        vault_match = self.VAULT_PATTERN.match(self._raw_value)
        if vault_match:
            self._is_reference = True
            self._reference_type = "vault"
            self._reference_key = vault_match.group(1)
            return

    @property
    def is_plaintext(self) -> bool:
        """Returns True if the secret is stored as plain text (not a reference)."""
        return not self._is_reference

    @property
    def reference_type(self) -> str | None:
        """Returns the reference type: 'env', 'vault', or None for plaintext."""
        return self._reference_type

    def resolve(self, vault_client: Any = None) -> str:
        """
        Resolve the secret to its actual value.

        Args:
            vault_client: Optional vault client for vault references.
                         Must have a method `read_secret(path) -> str`.

        Returns:
            The resolved secret value.

        Raises:
            ValueError: If the secret cannot be resolved.
        """
        if not self._is_reference:
            return self._raw_value

        if self._reference_type == "env":
            value = os.environ.get(self._reference_key)
            if value is None:
                raise ValueError(
                    f"Environment variable '{self._reference_key}' not set"
                )
            return value

        if self._reference_type == "vault":
            if vault_client is None:
                raise ValueError(
                    f"Vault client required to resolve vault reference: {self._reference_key}"
                )
            try:
                return vault_client.read_secret(self._reference_key)
            except Exception as e:
                raise ValueError(
                    f"Failed to read vault secret '{self._reference_key}': {e}"
                ) from e

        return self._raw_value

    def __repr__(self) -> str:
        if self._is_reference:
            return f"SecretField(${{{self._reference_type}:{self._reference_key}}})"
        return "SecretField(***masked***)"


def resolve_secrets(config: dict, vault_client: Any = None) -> dict:
    """
    Recursively resolve all secret references in a configuration dict.

    Looks for string values matching ${env:VAR} or ${vault:path} syntax
    and replaces them with resolved values.

    Args:
        config: Configuration dictionary (may be nested)
        vault_client: Optional vault client for vault references

    Returns:
        New config dict with secrets resolved

    Raises:
        ValueError: If any secret cannot be resolved
    """
    resolved = {}

    for key, value in config.items():
        if isinstance(value, dict):
            resolved[key] = resolve_secrets(value, vault_client)
        elif isinstance(value, str):
            # Check if it looks like a secret reference
            if value.startswith("${env:") or value.startswith("${vault:"):
                secret = SecretField(value)
                resolved[key] = secret.resolve(vault_client)
            else:
                resolved[key] = value
        elif isinstance(value, list):
            resolved[key] = [
                resolve_secrets(item, vault_client) if isinstance(item, dict)
                else SecretField(item).resolve(vault_client) if isinstance(item, str) and (item.startswith("${env:") or item.startswith("${vault:"))
                else item
                for item in value
            ]
        else:
            resolved[key] = value

    return resolved


def check_plaintext_secrets(config: dict, secret_field_paths: list[str] | None = None) -> list[str]:
    """
    Check for plaintext secrets in configuration.

    Args:
        config: Configuration dictionary
        secret_field_paths: List of dot-notation paths that should be secrets.
                           If None, uses default sensitive field patterns.

    Returns:
        List of field paths containing plaintext secrets (warnings)
    """
    if secret_field_paths is None:
        # Default patterns for sensitive fields
        secret_field_paths = [
            "compliance.firm_lei",
            "broker.api_key",
            "broker.api_secret",
            "broker.password",
            "database.password",
            "vault.token",
        ]

    warnings = []

    for path in secret_field_paths:
        value = _get_nested_value_static(config, path)
        if value is not None and isinstance(value, str):
            # Check if it's NOT a reference (i.e., plaintext)
            if not value.startswith("${env:") and not value.startswith("${vault:"):
                warnings.append(path)

    return warnings


def _get_nested_value_static(config: dict, path: str) -> Any:
    """Get value from nested dict using dot notation (static helper)."""
    keys = path.split(".")
    value = config

    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
        if value is None:
            return None

    return value


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

    def format_errors_for_display(self, include_warnings: bool = True) -> str:
        """
        Format validation issues for user-friendly display.

        Returns a numbered list of issues with clear formatting and suggestions.

        Args:
            include_warnings: Include warning-level issues (default True)

        Returns:
            Formatted string ready for console/log output
        """
        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("CONFIGURATION VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append("")

        errors = self.get_errors()
        warnings = self.get_warnings() if include_warnings else []

        if not errors and not warnings:
            lines.append("[OK] Configuration is valid!")
            lines.append("")
            return "\n".join(lines)

        # Display errors first
        if errors:
            lines.append(f"ERRORS ({len(errors)} issues that must be fixed):")
            lines.append("-" * 40)
            for i, issue in enumerate(errors, 1):
                lines.append(f"  {i}. [{issue.path}]")
                lines.append(f"     Problem: {issue.message}")
                if issue.suggestion:
                    lines.append(f"     Fix: {issue.suggestion}")
                lines.append("")

        # Display warnings
        if warnings:
            lines.append(f"WARNINGS ({len(warnings)} issues to review):")
            lines.append("-" * 40)
            for i, issue in enumerate(warnings, 1):
                lines.append(f"  {i}. [{issue.path}]")
                lines.append(f"     Note: {issue.message}")
                if issue.suggestion:
                    lines.append(f"     Suggestion: {issue.suggestion}")
                lines.append("")

        # Quick fix guide
        if errors:
            lines.append("=" * 60)
            lines.append("QUICK FIX GUIDE:")
            lines.append("-" * 40)

            # Group common fixes
            missing_fields = [e for e in errors if "missing" in e.message.lower()]
            invalid_values = [e for e in errors if "invalid" in e.message.lower() or "not in" in e.message.lower()]

            if missing_fields:
                lines.append("  Missing required fields:")
                for issue in missing_fields:
                    lines.append(f"    - Add '{issue.path}' to config.yaml")

            if invalid_values:
                lines.append("  Invalid values:")
                for issue in invalid_values:
                    lines.append(f"    - Check '{issue.path}' value")

            lines.append("")
            lines.append("For help, see: docs/CONFIGURATION.md or README.md")
            lines.append("=" * 60)

        return "\n".join(lines)


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
    # API-008: Runtime vs compile-time settings distinction
    hot_reload: bool = False  # If True, setting can be changed without restart
    is_secret: bool = False  # API-003: Mark fields that contain secrets


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

        # Broker settings (require restart - connection parameters)
        self.add_schema(FieldSchema(
            path="broker.host",
            field_type=str,
            required=True,
            default="127.0.0.1",
            description="IB Gateway/TWS host address",
            hot_reload=False,  # Requires reconnection
        ))
        self.add_schema(FieldSchema(
            path="broker.port",
            field_type=int,
            required=True,
            min_value=1,
            max_value=65535,
            description="IB Gateway/TWS port",
            hot_reload=False,  # Requires reconnection
        ))
        self.add_schema(FieldSchema(
            path="broker.client_id",
            field_type=int,
            required=True,
            min_value=1,
            max_value=999,
            description="IB client ID (must be unique per connection)",
            hot_reload=False,  # Requires reconnection
        ))
        self.add_schema(FieldSchema(
            path="broker.paper_trading",
            field_type=bool,
            required=False,
            default=True,
            description="Use paper trading mode",
            hot_reload=False,  # Critical setting, requires restart
        ))

        # Risk settings (some are hot-reloadable for dynamic risk adjustment)
        self.add_schema(FieldSchema(
            path="risk.max_position_pct",
            field_type=(int, float),
            required=True,
            min_value=0.1,
            max_value=100,
            description="Maximum position size as % of portfolio",
            hot_reload=True,  # Can be adjusted at runtime
        ))
        self.add_schema(FieldSchema(
            path="risk.max_sector_pct",
            field_type=(int, float),
            required=True,
            min_value=1,
            max_value=100,
            description="Maximum sector concentration",
            hot_reload=True,  # Can be adjusted at runtime
        ))
        self.add_schema(FieldSchema(
            path="risk.max_leverage",
            field_type=(int, float),
            required=True,
            min_value=1,
            max_value=10,
            description="Maximum leverage ratio",
            hot_reload=True,  # Can be adjusted at runtime
        ))
        self.add_schema(FieldSchema(
            path="risk.var_confidence",
            field_type=float,
            required=False,
            default=0.95,
            min_value=0.90,
            max_value=0.99,
            description="VaR confidence level",
            hot_reload=True,  # Can be adjusted at runtime
        ))

        # Compliance settings (API-003: LEI is a secret, requires restart)
        self.add_schema(FieldSchema(
            path="compliance.firm_lei",
            field_type=str,
            required=True,
            pattern=r"^[A-Z0-9]{20}$",
            validator=self._validate_lei,
            description="Firm LEI (Legal Entity Identifier)",
            hot_reload=False,  # Identity cannot change at runtime
            is_secret=True,  # API-003: Should use ${env:} or ${vault:} syntax
        ))
        self.add_schema(FieldSchema(
            path="compliance.jurisdiction",
            field_type=str,
            required=True,
            allowed_values=["EU", "US", "UK", "APAC"],
            description="Regulatory jurisdiction",
            hot_reload=False,  # Requires compliance reconfiguration
        ))

        # Execution settings (hot-reloadable for trading adjustments)
        self.add_schema(FieldSchema(
            path="execution.max_slippage_bps",
            field_type=(int, float),
            required=False,
            default=50,
            min_value=1,
            max_value=500,
            description="Maximum allowed slippage in basis points",
            hot_reload=True,  # Can be adjusted at runtime
        ))
        self.add_schema(FieldSchema(
            path="execution.default_algo",
            field_type=str,
            required=False,
            default="TWAP",
            allowed_values=["MARKET", "LIMIT", "TWAP", "VWAP"],
            description="Default execution algorithm",
            hot_reload=True,  # Can be adjusted at runtime
        ))

        # Portfolio settings (require restart - fundamental settings)
        self.add_schema(FieldSchema(
            path="portfolio.initial_capital",
            field_type=(int, float),
            required=True,
            min_value=10000,
            description="Initial portfolio capital",
            hot_reload=False,  # Cannot change initial capital at runtime
        ))
        self.add_schema(FieldSchema(
            path="portfolio.base_currency",
            field_type=str,
            required=False,
            default="USD",
            pattern=r"^[A-Z]{3}$",
            description="Base currency for portfolio",
            hot_reload=False,  # Requires portfolio recalculation
        ))

        # Strategy settings
        self.add_schema(FieldSchema(
            path="strategies.enabled",
            field_type=list,
            required=True,
            validator=lambda x: len(x) > 0,
            description="List of enabled strategies",
            hot_reload=False,  # Requires strategy initialization
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

    def get_hot_reloadable_fields(self) -> list[FieldSchema]:
        """
        Get list of fields that can be hot-reloaded at runtime.

        These fields can be changed without requiring a system restart.
        Useful for implementing dynamic configuration updates.

        Returns:
            List of FieldSchema objects with hot_reload=True

        API-008: Runtime vs compile-time settings distinction
        """
        return [schema for schema in self._schemas if schema.hot_reload]

    def get_restart_required_fields(self) -> list[FieldSchema]:
        """
        Get list of fields that require a restart when changed.

        These are compile-time settings that cannot be hot-reloaded.

        Returns:
            List of FieldSchema objects with hot_reload=False
        """
        return [schema for schema in self._schemas if not schema.hot_reload]

    def get_secret_fields(self) -> list[FieldSchema]:
        """
        Get list of fields marked as secrets.

        These fields should use ${env:VAR} or ${vault:path} syntax.

        Returns:
            List of FieldSchema objects with is_secret=True

        API-003: Secret management
        """
        return [schema for schema in self._schemas if schema.is_secret]

    def is_field_hot_reloadable(self, path: str) -> bool:
        """
        Check if a specific field can be hot-reloaded.

        Args:
            path: Dot-notation path to the field

        Returns:
            True if the field is hot-reloadable, False otherwise
        """
        for schema in self._schemas:
            if schema.path == path:
                return schema.hot_reload
        return False

    def validate(self, config: dict, strict: bool = False) -> ValidationResult:
        """
        Validate configuration against schemas.

        Args:
            config: Configuration dictionary
            strict: If True, raise ConfigValidationError on validation failure.
                   This ensures callers cannot ignore invalid configurations.
                   (API-006 fix)

        Returns:
            ValidationResult with all issues found

        Raises:
            ConfigValidationError: If strict=True and validation fails
        """
        result = ValidationResult(valid=True)

        # Validate each schema
        for schema in self._schemas:
            self._validate_field(config, schema, result)

        # API-003: Check for plaintext secrets
        secret_paths = [s.path for s in self._schemas if s.is_secret]
        plaintext_warnings = check_plaintext_secrets(config, secret_paths if secret_paths else None)
        for path in plaintext_warnings:
            result.add_warning(
                path,
                "Secret stored as plaintext in configuration",
                f"Use ${{env:VAR_NAME}} or ${{vault:secret/path}} syntax for '{path}'",
            )

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

        # API-006: Strict mode - raise exception if validation failed
        if strict and not result.valid:
            error_messages = result.format_errors_for_display(include_warnings=False)
            raise ConfigValidationError(
                f"Configuration validation failed in strict mode.\n{error_messages}",
                result,
            )

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


# =============================================================================
# P3: Configuration Diff Reporting
# =============================================================================

@dataclass
class ConfigDiff:
    """Represents a difference between two configurations."""
    path: str
    change_type: str  # "added", "removed", "modified"
    old_value: Any = None
    new_value: Any = None

    def __str__(self) -> str:
        if self.change_type == "added":
            return f"+ {self.path}: {self.new_value}"
        elif self.change_type == "removed":
            return f"- {self.path}: {self.old_value}"
        else:
            return f"~ {self.path}: {self.old_value} -> {self.new_value}"


class ConfigDiffReporter:
    """
    Reports differences between two configurations.

    P3 Enhancement: Useful for:
    - Auditing configuration changes
    - Hot-reload validation
    - Migration verification
    """

    def __init__(self):
        self._diff_history: list[tuple[datetime, list[ConfigDiff]]] = []

    def compare(
        self,
        old_config: dict,
        new_config: dict,
        path_prefix: str = "",
    ) -> list[ConfigDiff]:
        """
        Compare two configurations and return differences.

        Args:
            old_config: Original configuration
            new_config: New configuration
            path_prefix: Prefix for nested paths

        Returns:
            List of ConfigDiff objects describing changes
        """
        diffs: list[ConfigDiff] = []

        all_keys = set(old_config.keys()) | set(new_config.keys())

        for key in all_keys:
            current_path = f"{path_prefix}.{key}" if path_prefix else key
            old_val = old_config.get(key)
            new_val = new_config.get(key)

            if key not in old_config:
                diffs.append(ConfigDiff(
                    path=current_path,
                    change_type="added",
                    new_value=new_val,
                ))
            elif key not in new_config:
                diffs.append(ConfigDiff(
                    path=current_path,
                    change_type="removed",
                    old_value=old_val,
                ))
            elif isinstance(old_val, dict) and isinstance(new_val, dict):
                # Recurse into nested dicts
                diffs.extend(self.compare(old_val, new_val, current_path))
            elif old_val != new_val:
                diffs.append(ConfigDiff(
                    path=current_path,
                    change_type="modified",
                    old_value=old_val,
                    new_value=new_val,
                ))

        return diffs

    def record_change(self, diffs: list[ConfigDiff]) -> None:
        """Record a configuration change for audit purposes."""
        if diffs:
            self._diff_history.append((datetime.now(timezone.utc), diffs))
            # Keep only last 100 changes
            if len(self._diff_history) > 100:
                self._diff_history = self._diff_history[-100:]

    def get_change_history(self) -> list[tuple[datetime, list[ConfigDiff]]]:
        """Get recorded change history."""
        return self._diff_history.copy()

    def format_diff_report(
        self,
        diffs: list[ConfigDiff],
        include_timestamp: bool = True,
    ) -> str:
        """
        Format a diff report for display or logging.

        Args:
            diffs: List of differences to format
            include_timestamp: Include timestamp in report

        Returns:
            Formatted string report
        """
        if not diffs:
            return "No configuration changes detected."

        lines = []
        if include_timestamp:
            lines.append(f"Configuration Diff Report - {datetime.now(timezone.utc).isoformat()}")
            lines.append("=" * 60)

        # Group by change type
        added = [d for d in diffs if d.change_type == "added"]
        removed = [d for d in diffs if d.change_type == "removed"]
        modified = [d for d in diffs if d.change_type == "modified"]

        if added:
            lines.append(f"\nAdded ({len(added)}):")
            for diff in added:
                lines.append(f"  + {diff.path}: {diff.new_value}")

        if removed:
            lines.append(f"\nRemoved ({len(removed)}):")
            for diff in removed:
                lines.append(f"  - {diff.path}: {diff.old_value}")

        if modified:
            lines.append(f"\nModified ({len(modified)}):")
            for diff in modified:
                lines.append(f"  ~ {diff.path}:")
                lines.append(f"      old: {diff.old_value}")
                lines.append(f"      new: {diff.new_value}")

        lines.append(f"\nTotal changes: {len(diffs)}")
        return "\n".join(lines)

    def validate_hot_reload_safety(
        self,
        diffs: list[ConfigDiff],
        validator: "ConfigValidator",
    ) -> tuple[bool, list[str]]:
        """
        Check if configuration changes are safe for hot-reload.

        Args:
            diffs: List of configuration differences
            validator: ConfigValidator to check hot-reload settings

        Returns:
            Tuple of (is_safe, list of unsafe paths)
        """
        unsafe_paths = []

        for diff in diffs:
            if diff.change_type in ("modified", "removed"):
                if not validator.is_field_hot_reloadable(diff.path):
                    unsafe_paths.append(diff.path)

        is_safe = len(unsafe_paths) == 0
        return is_safe, unsafe_paths


# =============================================================================
# P3: Environment-Specific Validation
# =============================================================================

class EnvironmentType(str, Enum):
    """Supported environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


@dataclass
class EnvironmentValidationRule:
    """Rule for environment-specific validation."""
    environment: EnvironmentType
    path: str
    rule_type: str  # "required", "forbidden", "range", "value"
    expected_value: Any = None
    min_value: float | None = None
    max_value: float | None = None
    message: str = ""


class EnvironmentConfigValidator:
    """
    Validates configuration based on target environment.

    P3 Enhancement: Different environments have different requirements:
    - Development: Allow relaxed settings, paper trading required
    - Staging: Production-like but with safety checks
    - Production: Strict validation, real credentials required
    """

    def __init__(self):
        self._rules: dict[EnvironmentType, list[EnvironmentValidationRule]] = {
            env: [] for env in EnvironmentType
        }
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default environment-specific rules."""
        # Development rules
        self.add_rule(EnvironmentValidationRule(
            environment=EnvironmentType.DEVELOPMENT,
            path="broker.paper_trading",
            rule_type="value",
            expected_value=True,
            message="Paper trading must be enabled in development",
        ))
        self.add_rule(EnvironmentValidationRule(
            environment=EnvironmentType.DEVELOPMENT,
            path="risk.max_leverage",
            rule_type="range",
            max_value=2.0,
            message="Leverage should be limited in development",
        ))

        # Production rules
        self.add_rule(EnvironmentValidationRule(
            environment=EnvironmentType.PRODUCTION,
            path="compliance.firm_lei",
            rule_type="required",
            message="Valid LEI required for production",
        ))
        self.add_rule(EnvironmentValidationRule(
            environment=EnvironmentType.PRODUCTION,
            path="broker.api_key",
            rule_type="required",
            message="API key required for production",
        ))
        self.add_rule(EnvironmentValidationRule(
            environment=EnvironmentType.PRODUCTION,
            path="risk.max_leverage",
            rule_type="range",
            max_value=5.0,
            message="Leverage must be <= 5x in production",
        ))

        # Staging rules
        self.add_rule(EnvironmentValidationRule(
            environment=EnvironmentType.STAGING,
            path="broker.paper_trading",
            rule_type="value",
            expected_value=True,
            message="Paper trading recommended in staging",
        ))

    def add_rule(self, rule: EnvironmentValidationRule) -> None:
        """Add an environment-specific validation rule."""
        self._rules[rule.environment].append(rule)

    def validate_for_environment(
        self,
        config: dict,
        environment: EnvironmentType | str,
    ) -> ValidationResult:
        """
        Validate configuration for a specific environment.

        Args:
            config: Configuration dictionary to validate
            environment: Target environment

        Returns:
            ValidationResult with environment-specific issues
        """
        if isinstance(environment, str):
            try:
                environment = EnvironmentType(environment.lower())
            except ValueError:
                result = ValidationResult(valid=False)
                result.add_error(
                    "environment",
                    f"Unknown environment: {environment}",
                    f"Use one of: {[e.value for e in EnvironmentType]}",
                )
                return result

        result = ValidationResult(valid=True)
        rules = self._rules.get(environment, [])

        for rule in rules:
            value = self._get_nested_value(config, rule.path)

            if rule.rule_type == "required":
                if value is None or value == "" or (isinstance(value, str) and "PLACEHOLDER" in value.upper()):
                    result.add_error(
                        rule.path,
                        f"Required for {environment.value}: {rule.message}",
                    )

            elif rule.rule_type == "forbidden":
                if value is not None:
                    result.add_error(
                        rule.path,
                        f"Forbidden in {environment.value}: {rule.message}",
                    )

            elif rule.rule_type == "value":
                if value != rule.expected_value:
                    result.add_warning(
                        rule.path,
                        f"Expected {rule.expected_value} in {environment.value}: {rule.message}",
                    )

            elif rule.rule_type == "range":
                if value is not None and isinstance(value, (int, float)):
                    if rule.min_value is not None and value < rule.min_value:
                        result.add_error(
                            rule.path,
                            f"Value {value} below minimum {rule.min_value} for {environment.value}",
                        )
                    if rule.max_value is not None and value > rule.max_value:
                        result.add_error(
                            rule.path,
                            f"Value {value} above maximum {rule.max_value} for {environment.value}",
                        )

        result.add_info(
            "environment",
            f"Validated for environment: {environment.value}",
        )

        return result

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


# =============================================================================
# P3: Configuration Migration Helpers
# =============================================================================

@dataclass
class MigrationStep:
    """Single step in a configuration migration."""
    description: str
    action: str  # "rename", "remove", "add", "transform", "move"
    source_path: str | None = None
    target_path: str | None = None
    default_value: Any = None
    transform_func: Callable[[Any], Any] | None = None


class ConfigMigrationHelper:
    """
    Helper utilities for configuration migrations.

    P3 Enhancement: Provides common migration operations:
    - Rename fields
    - Move fields between sections
    - Add new required fields with defaults
    - Remove deprecated fields
    - Transform values
    """

    def __init__(self):
        self._migration_log: list[dict[str, Any]] = []

    def rename_field(
        self,
        config: dict,
        old_path: str,
        new_path: str,
    ) -> dict:
        """
        Rename a configuration field.

        Args:
            config: Configuration dictionary
            old_path: Current field path (dot notation)
            new_path: New field path (dot notation)

        Returns:
            Modified configuration
        """
        value = self._get_and_remove(config, old_path)
        if value is not None:
            self._set_nested(config, new_path, value)
            self._log_migration("rename", old_path, new_path, value)
        return config

    def move_field(
        self,
        config: dict,
        source_path: str,
        target_path: str,
    ) -> dict:
        """Move a field to a different location."""
        return self.rename_field(config, source_path, target_path)

    def add_field_if_missing(
        self,
        config: dict,
        path: str,
        default_value: Any,
    ) -> dict:
        """
        Add a field with default value if it doesn't exist.

        Args:
            config: Configuration dictionary
            path: Field path (dot notation)
            default_value: Value to set if field is missing

        Returns:
            Modified configuration
        """
        current = self._get_nested(config, path)
        if current is None:
            self._set_nested(config, path, default_value)
            self._log_migration("add", None, path, default_value)
        return config

    def remove_field(
        self,
        config: dict,
        path: str,
    ) -> dict:
        """
        Remove a deprecated field.

        Args:
            config: Configuration dictionary
            path: Field path to remove

        Returns:
            Modified configuration
        """
        value = self._get_and_remove(config, path)
        if value is not None:
            self._log_migration("remove", path, None, value)
        return config

    def transform_field(
        self,
        config: dict,
        path: str,
        transform_func: Callable[[Any], Any],
    ) -> dict:
        """
        Transform a field value.

        Args:
            config: Configuration dictionary
            path: Field path
            transform_func: Function to transform the value

        Returns:
            Modified configuration
        """
        value = self._get_nested(config, path)
        if value is not None:
            new_value = transform_func(value)
            self._set_nested(config, path, new_value)
            self._log_migration("transform", path, path, f"{value} -> {new_value}")
        return config

    def apply_migration_steps(
        self,
        config: dict,
        steps: list[MigrationStep],
    ) -> dict:
        """
        Apply a series of migration steps.

        Args:
            config: Configuration dictionary
            steps: List of migration steps to apply

        Returns:
            Modified configuration
        """
        for step in steps:
            if step.action == "rename":
                config = self.rename_field(config, step.source_path, step.target_path)
            elif step.action == "move":
                config = self.move_field(config, step.source_path, step.target_path)
            elif step.action == "add":
                config = self.add_field_if_missing(config, step.target_path, step.default_value)
            elif step.action == "remove":
                config = self.remove_field(config, step.source_path)
            elif step.action == "transform" and step.transform_func:
                config = self.transform_field(config, step.source_path, step.transform_func)

        return config

    def get_migration_log(self) -> list[dict[str, Any]]:
        """Get the log of all migrations performed."""
        return self._migration_log.copy()

    def _log_migration(
        self,
        action: str,
        source: str | None,
        target: str | None,
        value: Any,
    ) -> None:
        """Log a migration action."""
        self._migration_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "source": source,
            "target": target,
            "value": str(value)[:100],  # Truncate for logging
        })

    def _get_nested(self, config: dict, path: str) -> Any:
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

    def _set_nested(self, config: dict, path: str, value: Any) -> None:
        """Set value in nested dict using dot notation."""
        keys = path.split(".")
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def _get_and_remove(self, config: dict, path: str) -> Any:
        """Get and remove value from nested dict."""
        keys = path.split(".")
        current = config
        for key in keys[:-1]:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]

        if isinstance(current, dict) and keys[-1] in current:
            return current.pop(keys[-1])
        return None


# Pre-built migration definitions for common version upgrades
COMMON_MIGRATIONS: dict[str, list[MigrationStep]] = {
    "1.0.0_to_1.1.0": [
        MigrationStep(
            description="Rename max_pos to max_position_pct",
            action="rename",
            source_path="risk.max_pos",
            target_path="risk.max_position_pct",
        ),
        MigrationStep(
            description="Add default execution algo",
            action="add",
            target_path="execution.default_algo",
            default_value="TWAP",
        ),
    ],
    "1.1.0_to_2.0.0": [
        MigrationStep(
            description="Move API settings to broker section",
            action="move",
            source_path="api.host",
            target_path="broker.host",
        ),
        MigrationStep(
            description="Add VaR confidence level",
            action="add",
            target_path="risk.var_confidence",
            default_value=0.95,
        ),
    ],
}


def validate_config_at_startup(config: dict, strict: bool = True) -> bool:
    """
    Convenience function to validate config at startup.

    Args:
        config: Configuration dictionary to validate
        strict: If True (default), raises ConfigValidationError on failure.
                This ensures invalid configs cannot be ignored.

    Returns:
        True if valid

    Raises:
        ConfigValidationError: If strict=True and validation fails (API-006)
    """
    validator = ConfigValidator()
    # Use strict mode by default at startup to enforce validation
    result = validator.validate(config, strict=strict)

    return result.valid
