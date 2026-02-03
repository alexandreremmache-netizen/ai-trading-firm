# config_validator

**Path**: `C:\Users\Alexa\ai-trading-firm\core\config_validator.py`

## Overview

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

## Classes

### ValidationSeverity

**Inherits from**: str, Enum

Validation issue severity.

### ValidationIssue

Single validation issue.

#### Methods

##### `def __str__(self) -> str`

### ValidationResult

Complete validation result.

#### Methods

##### `def add_error(self, path: str, message: str, suggestion: ) -> None`

Add error issue.

##### `def add_warning(self, path: str, message: str, suggestion: ) -> None`

Add warning issue.

##### `def add_info(self, path: str, message: str) -> None`

Add info issue.

##### `def get_errors(self) -> list[ValidationIssue]`

Get only error-level issues.

##### `def get_warnings(self) -> list[ValidationIssue]`

Get only warning-level issues.

##### `def summary(self) -> str`

Get validation summary.

##### `def format_errors_for_display(self, include_warnings: bool) -> str`

Format validation issues for user-friendly display.

Returns a numbered list of issues with clear formatting and suggestions.

Args:
    include_warnings: Include warning-level issues (default True)

Returns:
    Formatted string ready for console/log output

### FieldSchema

Schema for a single config field.

### ConfigValidator

Configuration validator with schema support.

Validates config at startup to catch issues early.

#### Methods

##### `def __init__(self)`

##### `def add_schema(self, schema: FieldSchema) -> None`

Add field schema.

##### `def add_cross_validator(self, validator: Callable[, None]) -> None`

Add cross-field validator.

##### `def validate(self, config: dict) -> ValidationResult`

Validate configuration against schemas.

Args:
    config: Configuration dictionary

Returns:
    ValidationResult with all issues found

### ConfigMigrator

Handles configuration migrations between versions.

Helps upgrade old configs to new format.

#### Methods

##### `def __init__(self)`

##### `def add_migration(self, from_version: str, to_version: str, migrator: Callable[, dict]) -> None`

Add migration function.

##### `def migrate(self, config: dict, target_version: str) -> dict`

Migrate config to target version.

## Functions

### `def validate_config_at_startup(config: dict) -> bool`

Convenience function to validate config at startup.

Returns True if valid, raises on critical errors.
