"""
Configuration Manager
=====================

Configuration viewing and management component for the trading system dashboard.

Features:
- Load and parse configuration from YAML files
- View configuration by section (firm, broker, risk, compliance, etc.)
- Safe parameter validation before changes
- Change history tracking
- Config diff view (current vs default)
- Export current config as YAML
- Read-only mode for production safety
- Dangerous parameter warnings (require confirmation)

Security:
- Read-only mode by default in production
- Dangerous parameters require explicit confirmation
- All changes are logged and audited
- Restart-required parameters are flagged
"""

from __future__ import annotations

import copy
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import yaml

from core.config_validator import (
    ConfigValidator,
    ConfigDiffReporter,
    ConfigDiff,
    FieldSchema,
    ValidationResult,
    ValidationSeverity,
)


logger = logging.getLogger(__name__)


class ParameterType(str, Enum):
    """Types of configuration parameters."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    PATH = "path"
    SECRET = "secret"


class DangerLevel(str, Enum):
    """Danger level of a configuration parameter."""
    SAFE = "safe"              # Can be changed freely
    REVIEW = "review"          # Should review before changing
    DANGEROUS = "dangerous"    # Requires explicit confirmation
    CRITICAL = "critical"      # Cannot be changed (read-only)


class ChangeStatus(str, Enum):
    """Status of a configuration change."""
    PENDING = "pending"        # Change requested but not applied
    VALIDATED = "validated"    # Change validated successfully
    APPLIED = "applied"        # Change has been applied
    REJECTED = "rejected"      # Change was rejected (validation failed)
    REVERTED = "reverted"      # Change was reverted


@dataclass
class ConfigParameter:
    """
    Represents a single configuration parameter.

    Captures all metadata needed for display, validation, and safe modification.
    """
    name: str
    value: Any
    param_type: ParameterType
    default: Any = None
    min_value: float | None = None
    max_value: float | None = None
    description: str = ""
    requires_restart: bool = False
    danger_level: DangerLevel = DangerLevel.SAFE
    allowed_values: list[Any] | None = None
    pattern: str | None = None  # Regex pattern for validation
    section: str = ""
    full_path: str = ""  # Dot-notation path (e.g., "risk.max_leverage")
    is_editable: bool = True
    is_secret: bool = False
    last_modified: datetime | None = None
    modified_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "name": self.name,
            "value": "***" if self.is_secret else self.value,
            "type": self.param_type.value,
            "default": "***" if self.is_secret else self.default,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "description": self.description,
            "requires_restart": self.requires_restart,
            "danger_level": self.danger_level.value,
            "allowed_values": self.allowed_values,
            "pattern": self.pattern,
            "section": self.section,
            "full_path": self.full_path,
            "is_editable": self.is_editable,
            "is_secret": self.is_secret,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "modified_by": self.modified_by,
        }

    def validate_value(self, new_value: Any) -> tuple[bool, str | None]:
        """
        Validate a proposed new value for this parameter.

        Args:
            new_value: The proposed new value

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check type
        type_valid, type_error = self._validate_type(new_value)
        if not type_valid:
            return False, type_error

        # Check range for numeric types
        if self.param_type in (ParameterType.INTEGER, ParameterType.FLOAT):
            if self.min_value is not None and new_value < self.min_value:
                return False, f"Value {new_value} is below minimum {self.min_value}"
            if self.max_value is not None and new_value > self.max_value:
                return False, f"Value {new_value} is above maximum {self.max_value}"

        # Check allowed values
        if self.allowed_values is not None:
            if new_value not in self.allowed_values:
                return False, f"Value '{new_value}' not in allowed values: {self.allowed_values}"

        # Check pattern for strings
        if self.param_type == ParameterType.STRING and self.pattern:
            if not re.match(self.pattern, str(new_value)):
                return False, f"Value '{new_value}' does not match pattern: {self.pattern}"

        return True, None

    def _validate_type(self, value: Any) -> tuple[bool, str | None]:
        """Validate the type of a value."""
        expected_types = {
            ParameterType.STRING: str,
            ParameterType.INTEGER: int,
            ParameterType.FLOAT: (int, float),
            ParameterType.BOOLEAN: bool,
            ParameterType.LIST: list,
            ParameterType.DICT: dict,
            ParameterType.PATH: str,
            ParameterType.SECRET: str,
        }

        expected = expected_types.get(self.param_type)
        if expected and not isinstance(value, expected):
            return False, f"Expected type {self.param_type.value}, got {type(value).__name__}"

        return True, None


@dataclass
class ConfigSection:
    """
    Represents a section of the configuration.

    Groups related parameters together for organized viewing.
    """
    section_name: str
    parameters: dict[str, ConfigParameter] = field(default_factory=dict)
    description: str = ""
    editable: bool = True
    danger_level: DangerLevel = DangerLevel.SAFE
    order: int = 0  # For display ordering

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "section_name": self.section_name,
            "parameters": {
                name: param.to_dict()
                for name, param in self.parameters.items()
            },
            "description": self.description,
            "editable": self.editable,
            "danger_level": self.danger_level.value,
            "order": self.order,
            "parameter_count": len(self.parameters),
        }

    def get_parameter(self, name: str) -> ConfigParameter | None:
        """Get a parameter by name."""
        return self.parameters.get(name)

    def add_parameter(self, param: ConfigParameter) -> None:
        """Add a parameter to this section."""
        param.section = self.section_name
        self.parameters[param.name] = param


@dataclass
class ConfigChange:
    """
    Record of a configuration change.

    Tracks all changes for audit and rollback purposes.
    """
    change_id: str
    timestamp: datetime
    path: str
    old_value: Any
    new_value: Any
    status: ChangeStatus
    changed_by: str
    reason: str = ""
    requires_restart: bool = False
    danger_level: DangerLevel = DangerLevel.SAFE
    validation_result: str | None = None
    applied_at: datetime | None = None
    reverted_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WebSocket streaming."""
        return {
            "change_id": self.change_id,
            "timestamp": self.timestamp.isoformat(),
            "path": self.path,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "status": self.status.value,
            "changed_by": self.changed_by,
            "reason": self.reason,
            "requires_restart": self.requires_restart,
            "danger_level": self.danger_level.value,
            "validation_result": self.validation_result,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "reverted_at": self.reverted_at.isoformat() if self.reverted_at else None,
        }


# Section definitions with metadata
SECTION_DEFINITIONS: dict[str, dict[str, Any]] = {
    "firm": {
        "description": "Basic firm settings including name, version, and trading mode",
        "order": 1,
        "danger_level": DangerLevel.REVIEW,
        "editable": True,
    },
    "broker": {
        "description": "Interactive Brokers connection settings (host, port, client ID)",
        "order": 2,
        "danger_level": DangerLevel.DANGEROUS,
        "editable": True,
    },
    "risk": {
        "description": "Risk management limits (VaR, position size, leverage, loss limits)",
        "order": 3,
        "danger_level": DangerLevel.REVIEW,
        "editable": True,
    },
    "compliance": {
        "description": "Regulatory compliance settings (LEI, jurisdiction, audit retention)",
        "order": 4,
        "danger_level": DangerLevel.DANGEROUS,
        "editable": True,
    },
    "agents": {
        "description": "Trading agent configuration (signal agents, CIO, execution)",
        "order": 5,
        "danger_level": DangerLevel.REVIEW,
        "editable": True,
    },
    "execution": {
        "description": "Order execution settings (algorithms, slippage limits)",
        "order": 6,
        "danger_level": DangerLevel.REVIEW,
        "editable": True,
    },
    "monitoring": {
        "description": "Monitoring and alerting configuration (thresholds, log levels)",
        "order": 7,
        "danger_level": DangerLevel.SAFE,
        "editable": True,
    },
    "strategies": {
        "description": "Strategy configuration (enabled strategies, weights)",
        "order": 8,
        "danger_level": DangerLevel.REVIEW,
        "editable": True,
    },
    "universe": {
        "description": "Trading universe (equities, ETFs, futures, forex)",
        "order": 9,
        "danger_level": DangerLevel.SAFE,
        "editable": True,
    },
    "portfolio": {
        "description": "Portfolio settings (initial capital, base currency)",
        "order": 10,
        "danger_level": DangerLevel.DANGEROUS,
        "editable": True,
    },
    "logging": {
        "description": "Logging configuration (level, format, output files)",
        "order": 11,
        "danger_level": DangerLevel.SAFE,
        "editable": True,
    },
    "event_bus": {
        "description": "Event bus configuration (queue size, timeouts)",
        "order": 12,
        "danger_level": DangerLevel.REVIEW,
        "editable": True,
    },
    "surveillance": {
        "description": "Market surveillance settings (abuse detection thresholds)",
        "order": 13,
        "danger_level": DangerLevel.DANGEROUS,
        "editable": True,
    },
    "transaction_reporting": {
        "description": "Transaction reporting settings (ESMA RTS 22/23)",
        "order": 14,
        "danger_level": DangerLevel.DANGEROUS,
        "editable": True,
    },
    "var": {
        "description": "Value at Risk calculation settings",
        "order": 15,
        "danger_level": DangerLevel.REVIEW,
        "editable": True,
    },
    "position_sizing": {
        "description": "Position sizing method and parameters",
        "order": 16,
        "danger_level": DangerLevel.REVIEW,
        "editable": True,
    },
}


# Dangerous parameters that require confirmation
DANGEROUS_PARAMETERS: dict[str, str] = {
    "firm.mode": "Changing trading mode can result in real money trades",
    "broker.port": "Changing port can connect to live trading environment",
    "broker.readonly": "Disabling read-only allows order placement",
    "risk.max_leverage": "Increasing leverage increases risk of significant losses",
    "risk.max_daily_loss_pct": "Increasing daily loss limit reduces capital protection",
    "risk.max_drawdown_pct": "Increasing drawdown limit reduces capital protection",
    "compliance.firm_lei": "LEI is a regulatory identifier that must be valid",
    "surveillance.wash_trading_detection": "Disabling may violate MAR 2014/596/EU",
    "surveillance.spoofing_detection": "Disabling may violate market manipulation rules",
    "portfolio.initial_capital": "Changing capital affects all risk calculations",
}


# Parameters that require restart
RESTART_REQUIRED_PARAMETERS: set[str] = {
    "broker.host",
    "broker.port",
    "broker.client_id",
    "firm.mode",
    "compliance.firm_lei",
    "compliance.jurisdiction",
    "portfolio.initial_capital",
    "portfolio.base_currency",
    "event_bus.max_queue_size",
    "strategies.enabled",
}


class ConfigManager:
    """
    Configuration viewing and management component.

    Provides safe configuration viewing, validation, and modification
    with full audit trail and rollback capabilities.

    Usage:
        # Create manager (read-only mode by default in production)
        manager = ConfigManager(read_only=True)

        # Load configuration
        await manager.load_config("config.yaml")

        # Get a section
        risk_section = manager.get_section("risk")

        # Get a specific parameter
        max_leverage = manager.get_parameter("risk.max_leverage")

        # Validate a proposed change
        result = manager.validate_change("risk.max_leverage", 3.0)

        # Apply change (if not read-only)
        if not manager.read_only:
            success = manager.apply_change("risk.max_leverage", 3.0, "admin", "Increased for testing")

        # Get change history
        history = manager.get_change_history()

        # Export configuration
        yaml_str = manager.export_yaml()

        # Get diff from default
        diffs = manager.get_diff_from_default()

        # Export for WebSocket streaming
        data = manager.to_dict()

    Security:
        - Read-only mode is enforced in production
        - Dangerous parameters require explicit confirmation
        - All changes are logged with user attribution
        - Changes can be reverted from history
    """

    # Maximum number of changes to keep in history
    MAX_CHANGE_HISTORY = 500

    def __init__(
        self,
        read_only: bool = True,
        config_path: str | Path | None = None,
        default_config_path: str | Path | None = None,
    ):
        """
        Initialize the configuration manager.

        Args:
            read_only: If True, no changes can be applied (default True for safety)
            config_path: Path to the configuration file
            default_config_path: Path to the default configuration file for diffs
        """
        self._read_only = read_only
        self._config_path = Path(config_path) if config_path else None
        self._default_config_path = Path(default_config_path) if default_config_path else None

        # Configuration state
        self._raw_config: dict[str, Any] = {}
        self._default_config: dict[str, Any] = {}
        self._sections: dict[str, ConfigSection] = {}
        self._parameters: dict[str, ConfigParameter] = {}  # Full path -> parameter

        # Change tracking
        self._change_history: deque[ConfigChange] = deque(maxlen=self.MAX_CHANGE_HISTORY)
        self._change_counter = 0

        # Validators
        self._validator = ConfigValidator()
        self._diff_reporter = ConfigDiffReporter()

        # Custom validators for specific parameters
        self._custom_validators: dict[str, Callable[[Any], tuple[bool, str | None]]] = {}

        # Initialize sections
        self._initialize_sections()

        logger.info(
            f"ConfigManager initialized (read_only={read_only}, "
            f"config_path={config_path})"
        )

    def _initialize_sections(self) -> None:
        """Initialize section definitions."""
        for section_name, definition in SECTION_DEFINITIONS.items():
            self._sections[section_name] = ConfigSection(
                section_name=section_name,
                description=definition.get("description", ""),
                editable=definition.get("editable", True) and not self._read_only,
                danger_level=DangerLevel(definition.get("danger_level", DangerLevel.SAFE)),
                order=definition.get("order", 99),
            )

    @property
    def read_only(self) -> bool:
        """Check if manager is in read-only mode."""
        return self._read_only

    @property
    def config_loaded(self) -> bool:
        """Check if configuration has been loaded."""
        return len(self._raw_config) > 0

    def load_config(
        self,
        config_path: str | Path | None = None,
        config_dict: dict[str, Any] | None = None,
    ) -> bool:
        """
        Load configuration from file or dictionary.

        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to file)

        Returns:
            True if loaded successfully
        """
        try:
            if config_dict is not None:
                self._raw_config = copy.deepcopy(config_dict)
            elif config_path is not None:
                path = Path(config_path)
                if not path.exists():
                    logger.error(f"Configuration file not found: {path}")
                    return False

                with open(path, "r", encoding="utf-8") as f:
                    self._raw_config = yaml.safe_load(f) or {}

                self._config_path = path
            elif self._config_path is not None:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    self._raw_config = yaml.safe_load(f) or {}
            else:
                logger.error("No configuration source provided")
                return False

            # Parse configuration into sections and parameters
            self._parse_config()

            logger.info(
                f"Configuration loaded: {len(self._sections)} sections, "
                f"{len(self._parameters)} parameters"
            )
            return True

        except yaml.YAMLError as e:
            logger.exception(f"YAML parsing error: {e}")
            return False
        except Exception as e:
            logger.exception(f"Error loading configuration: {e}")
            return False

    def load_default_config(
        self,
        default_path: str | Path | None = None,
        default_dict: dict[str, Any] | None = None,
    ) -> bool:
        """
        Load default configuration for comparison.

        Args:
            default_path: Path to default YAML configuration file
            default_dict: Default configuration dictionary

        Returns:
            True if loaded successfully
        """
        try:
            if default_dict is not None:
                self._default_config = copy.deepcopy(default_dict)
            elif default_path is not None:
                path = Path(default_path)
                if not path.exists():
                    logger.warning(f"Default configuration file not found: {path}")
                    return False

                with open(path, "r", encoding="utf-8") as f:
                    self._default_config = yaml.safe_load(f) or {}

                self._default_config_path = path
            elif self._default_config_path is not None:
                with open(self._default_config_path, "r", encoding="utf-8") as f:
                    self._default_config = yaml.safe_load(f) or {}
            else:
                # Use a minimal default config if none provided
                self._default_config = self._generate_minimal_defaults()

            logger.info("Default configuration loaded for comparison")
            return True

        except Exception as e:
            logger.exception(f"Error loading default configuration: {e}")
            return False

    def _generate_minimal_defaults(self) -> dict[str, Any]:
        """Generate minimal default configuration values."""
        return {
            "firm": {
                "name": "AI Trading Firm",
                "version": "1.0.0",
                "mode": "paper",
            },
            "broker": {
                "host": "127.0.0.1",
                "port": 4002,
                "client_id": 1,
                "timeout_seconds": 30,
                "readonly": False,
                "use_delayed_data": True,
            },
            "risk": {
                "max_portfolio_var_pct": 2.0,
                "max_position_pct": 5.0,
                "max_sector_pct": 20.0,
                "max_daily_loss_pct": 3.0,
                "max_drawdown_pct": 10.0,
                "max_leverage": 2.0,
                "max_orders_per_minute": 10,
            },
            "compliance": {
                "jurisdiction": "EU",
                "regulator": "AMF",
                "require_rationale": True,
                "audit_retention_days": 2555,
            },
            "portfolio": {
                "initial_capital": 100000.0,
                "base_currency": "USD",
            },
            "monitoring": {
                "log_dir": "logs/agents",
                "metrics_history_size": 10000,
            },
            "logging": {
                "level": "INFO",
            },
        }

    def _parse_config(self) -> None:
        """Parse raw configuration into sections and parameters."""
        self._parameters.clear()

        # Re-initialize sections to clear old parameters
        self._initialize_sections()

        for section_name, section_data in self._raw_config.items():
            if not isinstance(section_data, dict):
                # Handle non-dict values (like version strings)
                param = self._create_parameter(
                    name=section_name,
                    value=section_data,
                    full_path=section_name,
                    section=section_name,
                )
                self._parameters[section_name] = param
                continue

            # Ensure section exists
            if section_name not in self._sections:
                self._sections[section_name] = ConfigSection(
                    section_name=section_name,
                    description=f"Configuration section: {section_name}",
                    editable=not self._read_only,
                )

            section = self._sections[section_name]

            # Parse parameters in this section
            self._parse_section_parameters(section, section_data, section_name)

    def _parse_section_parameters(
        self,
        section: ConfigSection,
        data: dict[str, Any],
        path_prefix: str,
    ) -> None:
        """Recursively parse parameters in a section."""
        for key, value in data.items():
            full_path = f"{path_prefix}.{key}"

            if isinstance(value, dict) and not self._is_complex_value(value):
                # Recurse into nested sections
                self._parse_section_parameters(section, value, full_path)
            else:
                # Create parameter
                param = self._create_parameter(
                    name=key,
                    value=value,
                    full_path=full_path,
                    section=section.section_name,
                )
                section.add_parameter(param)
                self._parameters[full_path] = param

    def _is_complex_value(self, value: dict) -> bool:
        """Check if a dict value should be treated as a single complex value."""
        # Lists of dicts (like universe instruments) are complex values
        if any(isinstance(v, list) for v in value.values()):
            return True
        # Dicts with specific keys that indicate a single value
        single_value_keys = {"symbol", "exchange", "currency"}
        if single_value_keys.issubset(set(value.keys())):
            return True
        return False

    def _create_parameter(
        self,
        name: str,
        value: Any,
        full_path: str,
        section: str,
    ) -> ConfigParameter:
        """Create a ConfigParameter from a value."""
        # Determine type
        param_type = self._infer_type(value)

        # Determine danger level
        danger_level = DangerLevel.SAFE
        if full_path in DANGEROUS_PARAMETERS:
            danger_level = DangerLevel.DANGEROUS
        elif section in SECTION_DEFINITIONS:
            section_danger = SECTION_DEFINITIONS[section].get("danger_level", DangerLevel.SAFE)
            if isinstance(section_danger, DangerLevel):
                danger_level = section_danger

        # Check if requires restart
        requires_restart = full_path in RESTART_REQUIRED_PARAMETERS

        # Get default value
        default = self._get_nested_value(self._default_config, full_path)

        # Determine if editable
        is_editable = not self._read_only and danger_level != DangerLevel.CRITICAL

        # Check if secret
        is_secret = "lei" in name.lower() or "password" in name.lower() or "key" in name.lower() or "secret" in name.lower()

        # Get min/max from validator schemas if available
        min_value = None
        max_value = None
        allowed_values = None
        pattern = None
        description = ""

        for schema in self._validator._schemas:
            if schema.path == full_path:
                min_value = schema.min_value
                max_value = schema.max_value
                allowed_values = schema.allowed_values
                pattern = schema.pattern
                description = schema.description
                break

        # Add description from dangerous parameters
        if full_path in DANGEROUS_PARAMETERS and not description:
            description = DANGEROUS_PARAMETERS[full_path]

        return ConfigParameter(
            name=name,
            value=value,
            param_type=param_type,
            default=default,
            min_value=min_value,
            max_value=max_value,
            description=description,
            requires_restart=requires_restart,
            danger_level=danger_level,
            allowed_values=allowed_values,
            pattern=pattern,
            section=section,
            full_path=full_path,
            is_editable=is_editable,
            is_secret=is_secret,
        )

    def _infer_type(self, value: Any) -> ParameterType:
        """Infer parameter type from value."""
        if isinstance(value, bool):
            return ParameterType.BOOLEAN
        elif isinstance(value, int):
            return ParameterType.INTEGER
        elif isinstance(value, float):
            return ParameterType.FLOAT
        elif isinstance(value, str):
            return ParameterType.STRING
        elif isinstance(value, list):
            return ParameterType.LIST
        elif isinstance(value, dict):
            return ParameterType.DICT
        else:
            return ParameterType.STRING

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

    def _set_nested_value(self, config: dict, path: str, value: Any) -> None:
        """Set value in nested dict using dot notation."""
        keys = path.split(".")
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def get_section(self, section_name: str) -> ConfigSection | None:
        """
        Get a configuration section by name.

        Args:
            section_name: Name of the section (e.g., "risk", "broker")

        Returns:
            ConfigSection or None if not found
        """
        return self._sections.get(section_name)

    def get_all_sections(self) -> list[ConfigSection]:
        """
        Get all configuration sections.

        Returns:
            List of ConfigSection sorted by order
        """
        return sorted(self._sections.values(), key=lambda s: s.order)

    def get_parameter(self, path: str) -> ConfigParameter | None:
        """
        Get a parameter by its full path.

        Args:
            path: Dot-notation path (e.g., "risk.max_leverage")

        Returns:
            ConfigParameter or None if not found
        """
        return self._parameters.get(path)

    def get_parameters_by_section(self, section_name: str) -> list[ConfigParameter]:
        """
        Get all parameters in a section.

        Args:
            section_name: Name of the section

        Returns:
            List of ConfigParameter in the section
        """
        section = self._sections.get(section_name)
        if section:
            return list(section.parameters.values())
        return []

    def get_dangerous_parameters(self) -> list[ConfigParameter]:
        """
        Get all parameters marked as dangerous.

        Returns:
            List of ConfigParameter with danger_level >= DANGEROUS
        """
        return [
            p for p in self._parameters.values()
            if p.danger_level in (DangerLevel.DANGEROUS, DangerLevel.CRITICAL)
        ]

    def get_restart_required_parameters(self) -> list[ConfigParameter]:
        """
        Get all parameters that require restart when changed.

        Returns:
            List of ConfigParameter with requires_restart=True
        """
        return [p for p in self._parameters.values() if p.requires_restart]

    def validate_change(
        self,
        path: str,
        new_value: Any,
    ) -> tuple[bool, str | None, bool]:
        """
        Validate a proposed configuration change.

        Args:
            path: Dot-notation path of the parameter
            new_value: Proposed new value

        Returns:
            Tuple of (is_valid, error_message, requires_confirmation)
        """
        # Check if parameter exists
        param = self._parameters.get(path)
        if param is None:
            return False, f"Parameter not found: {path}", False

        # Check if editable
        if not param.is_editable:
            return False, f"Parameter is not editable: {path}", False

        # Check if read-only mode
        if self._read_only:
            return False, "Configuration is in read-only mode", False

        # Validate value
        is_valid, error = param.validate_value(new_value)
        if not is_valid:
            return False, error, False

        # Run custom validators
        if path in self._custom_validators:
            validator = self._custom_validators[path]
            is_valid, error = validator(new_value)
            if not is_valid:
                return False, error, False

        # Check if confirmation required (dangerous parameter)
        requires_confirmation = param.danger_level in (DangerLevel.DANGEROUS, DangerLevel.CRITICAL)

        return True, None, requires_confirmation

    def apply_change(
        self,
        path: str,
        new_value: Any,
        changed_by: str,
        reason: str = "",
        confirmed: bool = False,
    ) -> tuple[bool, str | None]:
        """
        Apply a configuration change.

        Args:
            path: Dot-notation path of the parameter
            new_value: New value to set
            changed_by: User or system applying the change
            reason: Reason for the change
            confirmed: If True, bypass dangerous parameter confirmation

        Returns:
            Tuple of (success, error_message)
        """
        # Check read-only mode
        if self._read_only:
            return False, "Configuration is in read-only mode"

        # Validate the change
        is_valid, error, requires_confirmation = self.validate_change(path, new_value)
        if not is_valid:
            return False, error

        # Check confirmation for dangerous parameters
        if requires_confirmation and not confirmed:
            param = self._parameters.get(path)
            warning = DANGEROUS_PARAMETERS.get(path, "This is a dangerous parameter")
            return False, f"Confirmation required: {warning}"

        # Get current value
        param = self._parameters.get(path)
        if param is None:
            return False, f"Parameter not found: {path}"

        old_value = param.value

        # Create change record
        self._change_counter += 1
        change = ConfigChange(
            change_id=f"CHG-{self._change_counter:06d}",
            timestamp=datetime.now(timezone.utc),
            path=path,
            old_value=old_value,
            new_value=new_value,
            status=ChangeStatus.PENDING,
            changed_by=changed_by,
            reason=reason,
            requires_restart=param.requires_restart,
            danger_level=param.danger_level,
        )

        try:
            # Apply change to raw config
            self._set_nested_value(self._raw_config, path, new_value)

            # Update parameter
            param.value = new_value
            param.last_modified = datetime.now(timezone.utc)
            param.modified_by = changed_by

            # Update change record
            change.status = ChangeStatus.APPLIED
            change.applied_at = datetime.now(timezone.utc)
            change.validation_result = "Success"

            self._change_history.append(change)

            logger.info(
                f"Configuration changed: {path} = {new_value} "
                f"(by {changed_by}, reason: {reason})"
            )

            # Record diff for audit
            self._diff_reporter.record_change([
                ConfigDiff(
                    path=path,
                    change_type="modified",
                    old_value=old_value,
                    new_value=new_value,
                )
            ])

            return True, None

        except Exception as e:
            change.status = ChangeStatus.REJECTED
            change.validation_result = str(e)
            self._change_history.append(change)

            logger.exception(f"Error applying configuration change: {e}")
            return False, str(e)

    def revert_change(self, change_id: str, reverted_by: str) -> tuple[bool, str | None]:
        """
        Revert a previous configuration change.

        Args:
            change_id: ID of the change to revert
            reverted_by: User or system reverting the change

        Returns:
            Tuple of (success, error_message)
        """
        if self._read_only:
            return False, "Configuration is in read-only mode"

        # Find the change
        change = None
        for c in self._change_history:
            if c.change_id == change_id:
                change = c
                break

        if change is None:
            return False, f"Change not found: {change_id}"

        if change.status != ChangeStatus.APPLIED:
            return False, f"Change is not in applied state: {change.status.value}"

        # Apply revert
        success, error = self.apply_change(
            path=change.path,
            new_value=change.old_value,
            changed_by=reverted_by,
            reason=f"Reverting change {change_id}",
            confirmed=True,  # Bypass confirmation for reverts
        )

        if success:
            change.status = ChangeStatus.REVERTED
            change.reverted_at = datetime.now(timezone.utc)

        return success, error

    def get_change_history(
        self,
        limit: int = 100,
        path_filter: str | None = None,
        status_filter: ChangeStatus | None = None,
    ) -> list[ConfigChange]:
        """
        Get configuration change history.

        Args:
            limit: Maximum number of changes to return
            path_filter: Filter by parameter path (prefix match)
            status_filter: Filter by change status

        Returns:
            List of ConfigChange, most recent first
        """
        changes = list(self._change_history)
        changes.reverse()

        # Apply filters
        if path_filter:
            changes = [c for c in changes if c.path.startswith(path_filter)]

        if status_filter:
            changes = [c for c in changes if c.status == status_filter]

        return changes[:limit]

    def get_diff_from_default(self) -> list[ConfigDiff]:
        """
        Get differences between current and default configuration.

        Returns:
            List of ConfigDiff objects
        """
        if not self._default_config:
            self.load_default_config()

        return self._diff_reporter.compare(self._default_config, self._raw_config)

    def export_yaml(self, include_comments: bool = True) -> str:
        """
        Export current configuration as YAML string.

        Args:
            include_comments: If True, add helpful comments

        Returns:
            YAML string representation
        """
        if include_comments:
            lines = [
                "# AI Trading Firm Configuration",
                f"# Exported at: {datetime.now(timezone.utc).isoformat()}",
                "# ",
                "# WARNING: Review all changes before using in production",
                "",
            ]

            # Add sections in order
            sections = self.get_all_sections()
            processed_sections = set()

            for section in sections:
                if section.section_name not in self._raw_config:
                    continue

                if section.section_name in processed_sections:
                    continue

                processed_sections.add(section.section_name)

                lines.append(f"# {section.description}")
                section_yaml = yaml.dump(
                    {section.section_name: self._raw_config[section.section_name]},
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
                lines.append(section_yaml)

            # Add any remaining sections not in SECTION_DEFINITIONS
            for section_name in self._raw_config:
                if section_name not in processed_sections:
                    section_yaml = yaml.dump(
                        {section_name: self._raw_config[section_name]},
                        default_flow_style=False,
                        sort_keys=False,
                        allow_unicode=True,
                    )
                    lines.append(section_yaml)

            return "\n".join(lines)
        else:
            return yaml.dump(
                self._raw_config,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    def save_config(self, path: str | Path | None = None) -> bool:
        """
        Save current configuration to file.

        Args:
            path: Path to save to (uses original path if not provided)

        Returns:
            True if saved successfully
        """
        if self._read_only:
            logger.warning("Cannot save configuration in read-only mode")
            return False

        save_path = Path(path) if path else self._config_path
        if save_path is None:
            logger.error("No save path specified")
            return False

        try:
            # Create backup
            if save_path.exists():
                backup_path = save_path.with_suffix(
                    f".backup.{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.yaml"
                )
                import shutil
                shutil.copy(save_path, backup_path)
                logger.info(f"Created backup: {backup_path}")

            # Write new config
            yaml_content = self.export_yaml(include_comments=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)

            logger.info(f"Configuration saved to: {save_path}")
            return True

        except Exception as e:
            logger.exception(f"Error saving configuration: {e}")
            return False

    def validate_full_config(self) -> ValidationResult:
        """
        Validate the entire configuration.

        Returns:
            ValidationResult with all validation issues
        """
        return self._validator.validate(self._raw_config)

    def add_custom_validator(
        self,
        path: str,
        validator: Callable[[Any], tuple[bool, str | None]],
    ) -> None:
        """
        Add a custom validator for a specific parameter.

        Args:
            path: Parameter path
            validator: Function that takes a value and returns (is_valid, error_message)
        """
        self._custom_validators[path] = validator

    def set_read_only(self, read_only: bool) -> None:
        """
        Set read-only mode.

        Args:
            read_only: If True, disables all modifications
        """
        old_value = self._read_only
        self._read_only = read_only

        # Update editability of all parameters and sections
        for param in self._parameters.values():
            param.is_editable = not read_only and param.danger_level != DangerLevel.CRITICAL

        for section in self._sections.values():
            section.editable = not read_only

        logger.info(f"Read-only mode changed: {old_value} -> {read_only}")

    def search_parameters(
        self,
        query: str,
        include_values: bool = False,
    ) -> list[ConfigParameter]:
        """
        Search for parameters by name or description.

        Args:
            query: Search query (case-insensitive)
            include_values: Also search in values

        Returns:
            List of matching ConfigParameter
        """
        query_lower = query.lower()
        results = []

        for param in self._parameters.values():
            # Search in name and path
            if query_lower in param.name.lower() or query_lower in param.full_path.lower():
                results.append(param)
                continue

            # Search in description
            if query_lower in param.description.lower():
                results.append(param)
                continue

            # Search in values
            if include_values:
                value_str = str(param.value).lower()
                if query_lower in value_str:
                    results.append(param)

        return results

    def get_config_summary(self) -> dict[str, Any]:
        """
        Get a summary of the configuration.

        Returns:
            Dictionary with configuration summary
        """
        return {
            "total_sections": len(self._sections),
            "total_parameters": len(self._parameters),
            "dangerous_parameters": len(self.get_dangerous_parameters()),
            "restart_required": len(self.get_restart_required_parameters()),
            "read_only": self._read_only,
            "config_loaded": self.config_loaded,
            "config_path": str(self._config_path) if self._config_path else None,
            "change_history_count": len(self._change_history),
            "last_change": (
                self._change_history[-1].to_dict()
                if self._change_history else None
            ),
            "sections": {
                s.section_name: {
                    "parameter_count": len(s.parameters),
                    "danger_level": s.danger_level.value,
                    "editable": s.editable,
                }
                for s in self._sections.values()
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Export manager state to dictionary for WebSocket streaming.

        Returns:
            Complete manager state as dict
        """
        return {
            "summary": self.get_config_summary(),
            "sections": {
                name: section.to_dict()
                for name, section in sorted(
                    self._sections.items(),
                    key=lambda x: x[1].order
                )
            },
            "dangerous_parameters": [
                p.to_dict() for p in self.get_dangerous_parameters()
            ],
            "restart_required_parameters": [
                p.to_dict() for p in self.get_restart_required_parameters()
            ],
            "recent_changes": [
                c.to_dict() for c in self.get_change_history(limit=10)
            ],
            "diff_from_default": [
                {
                    "path": d.path,
                    "change_type": d.change_type,
                    "old_value": d.old_value,
                    "new_value": d.new_value,
                }
                for d in self.get_diff_from_default()[:20]  # Limit for streaming
            ],
            "validation_result": self.validate_full_config().summary(),
            "read_only": self._read_only,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# Factory function for common use cases
def create_config_manager(
    config_path: str | Path = "config.yaml",
    read_only: bool | None = None,
    environment: str = "development",
) -> ConfigManager:
    """
    Create a ConfigManager with appropriate settings for the environment.

    Args:
        config_path: Path to configuration file
        read_only: Override read-only setting (None = auto-detect from environment)
        environment: Environment name ("development", "staging", "production")

    Returns:
        Configured ConfigManager instance
    """
    # Auto-detect read-only based on environment
    if read_only is None:
        read_only = environment in ("production", "prod")

    manager = ConfigManager(
        read_only=read_only,
        config_path=config_path,
    )

    # Load configuration
    manager.load_config()

    # Load default for comparison
    default_path = Path(config_path).parent / "config.simple.yaml"
    if default_path.exists():
        manager.load_default_config(default_path)

    return manager
