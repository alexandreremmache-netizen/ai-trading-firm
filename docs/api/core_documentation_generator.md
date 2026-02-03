# documentation_generator

**Path**: `C:\Users\Alexa\ai-trading-firm\core\documentation_generator.py`

## Overview

Auto-Generated Documentation System.

This module provides automated documentation generation from code,
including API documentation, module summaries, and architecture diagrams.

Addresses:
- #I24 - Documentation not auto-generated

## Classes

### DocFormat

**Inherits from**: Enum

Documentation output formats.

### FunctionDoc

Documentation for a function.

#### Methods

##### `def to_dict(self) -> Dict[str, Any]`

Convert to dictionary.

### ClassDoc

Documentation for a class.

#### Methods

##### `def to_dict(self) -> Dict[str, Any]`

Convert to dictionary.

### ModuleDoc

Documentation for a module.

#### Methods

##### `def to_dict(self) -> Dict[str, Any]`

Convert to dictionary.

### DocstringParser

Parser for Python docstrings (Google style).

#### Methods

##### `def parse(docstring: Optional[str]) -> Dict[str, Any]`

Parse a Google-style docstring.

Args:
    docstring: The docstring to parse

Returns:
    Dictionary with parsed sections

### CodeAnalyzer

Analyzes Python source code for documentation extraction.

#### Methods

##### `def __init__(self)`

##### `def analyze_file(self, file_path: str) -> Optional[ModuleDoc]`

Analyze a Python file and extract documentation.

Args:
    file_path: Path to the Python file

Returns:
    ModuleDoc object with extracted documentation

### DocumentationGenerator

Generates documentation from analyzed code.

Supports multiple output formats including Markdown, HTML, and JSON.

#### Methods

##### `def __init__(self, project_root: str)`

##### `def scan_directory(self, directory: str, exclude_patterns: Optional[List[str]])`

Scan a directory for Python files and analyze them.

Args:
    directory: Directory path to scan
    exclude_patterns: List of patterns to exclude

##### `def generate_markdown(self, output_dir: str)`

Generate Markdown documentation.

Args:
    output_dir: Output directory for generated docs

##### `def generate_json(self, output_file: str)`

Generate JSON documentation.

Args:
    output_file: Output file path

##### `def generate_summary(self) -> Dict[str, Any]`

Generate documentation summary.

## Functions

### `def generate_project_documentation(project_root: str, directories: List[str], output_dir: str)`

Generate full project documentation.

Args:
    project_root: Root directory of the project
    directories: List of directories to document
    output_dir: Output directory for documentation
