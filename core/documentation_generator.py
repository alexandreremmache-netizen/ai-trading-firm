"""
Auto-Generated Documentation System.

This module provides automated documentation generation from code,
including API documentation, module summaries, and architecture diagrams.

Addresses:
- #I24 - Documentation not auto-generated
"""

import ast
import inspect
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type


logger = logging.getLogger(__name__)


class DocFormat(Enum):
    """Documentation output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    RST = "rst"


@dataclass
class FunctionDoc:
    """Documentation for a function."""
    name: str
    module: str
    signature: str
    docstring: str
    parameters: List[Dict[str, str]] = field(default_factory=list)
    return_type: str = ""
    raises: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    line_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "module": self.module,
            "signature": self.signature,
            "docstring": self.docstring,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "raises": self.raises,
            "examples": self.examples,
            "decorators": self.decorators,
            "is_async": self.is_async,
            "line_number": self.line_number
        }


@dataclass
class ClassDoc:
    """Documentation for a class."""
    name: str
    module: str
    docstring: str
    bases: List[str] = field(default_factory=list)
    methods: List[FunctionDoc] = field(default_factory=list)
    class_attributes: List[Dict[str, str]] = field(default_factory=list)
    instance_attributes: List[Dict[str, str]] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    line_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "module": self.module,
            "docstring": self.docstring,
            "bases": self.bases,
            "methods": [m.to_dict() for m in self.methods],
            "class_attributes": self.class_attributes,
            "instance_attributes": self.instance_attributes,
            "decorators": self.decorators,
            "line_number": self.line_number
        }


@dataclass
class ModuleDoc:
    """Documentation for a module."""
    name: str
    path: str
    docstring: str
    classes: List[ClassDoc] = field(default_factory=list)
    functions: List[FunctionDoc] = field(default_factory=list)
    constants: List[Dict[str, Any]] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "docstring": self.docstring,
            "classes": [c.to_dict() for c in self.classes],
            "functions": [f.to_dict() for f in self.functions],
            "constants": self.constants,
            "imports": self.imports,
            "dependencies": self.dependencies
        }


class DocstringParser:
    """Parser for Python docstrings (Google style)."""

    @staticmethod
    def parse(docstring: Optional[str]) -> Dict[str, Any]:
        """
        Parse a Google-style docstring.

        Args:
            docstring: The docstring to parse

        Returns:
            Dictionary with parsed sections
        """
        if not docstring:
            return {
                "summary": "",
                "description": "",
                "args": [],
                "returns": "",
                "raises": [],
                "examples": []
            }

        lines = docstring.strip().split("\n")
        result = {
            "summary": "",
            "description": "",
            "args": [],
            "returns": "",
            "raises": [],
            "examples": []
        }

        current_section = "summary"
        current_content = []
        current_arg = None

        for line in lines:
            stripped = line.strip()

            # Check for section headers
            if stripped.lower() in ["args:", "arguments:", "parameters:"]:
                if current_content:
                    result["summary"] = " ".join(current_content)
                current_section = "args"
                current_content = []
                continue
            elif stripped.lower() in ["returns:", "return:"]:
                current_section = "returns"
                current_content = []
                continue
            elif stripped.lower() in ["raises:", "raise:", "exceptions:"]:
                current_section = "raises"
                current_content = []
                continue
            elif stripped.lower() in ["example:", "examples:"]:
                current_section = "examples"
                current_content = []
                continue

            # Process content based on section
            if current_section == "args":
                # Check if this is a new argument
                if ":" in stripped and not stripped.startswith(" "):
                    if current_arg:
                        result["args"].append(current_arg)
                    parts = stripped.split(":", 1)
                    current_arg = {
                        "name": parts[0].strip(),
                        "description": parts[1].strip() if len(parts) > 1 else ""
                    }
                elif current_arg and stripped:
                    current_arg["description"] += " " + stripped
            elif current_section == "returns":
                current_content.append(stripped)
            elif current_section == "raises":
                if ":" in stripped:
                    parts = stripped.split(":", 1)
                    result["raises"].append({
                        "exception": parts[0].strip(),
                        "description": parts[1].strip() if len(parts) > 1 else ""
                    })
            elif current_section == "examples":
                current_content.append(line)
            else:
                current_content.append(stripped)

        # Finalize
        if current_section == "args" and current_arg:
            result["args"].append(current_arg)
        elif current_section == "returns":
            result["returns"] = " ".join(current_content)
        elif current_section == "examples":
            result["examples"] = current_content
        elif current_section == "summary" and current_content:
            result["summary"] = " ".join(current_content)

        return result


class CodeAnalyzer:
    """Analyzes Python source code for documentation extraction."""

    def __init__(self):
        self.docstring_parser = DocstringParser()

    def analyze_file(self, file_path: str) -> Optional[ModuleDoc]:
        """
        Analyze a Python file and extract documentation.

        Args:
            file_path: Path to the Python file

        Returns:
            ModuleDoc object with extracted documentation
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)
            module_name = Path(file_path).stem

            module_doc = ModuleDoc(
                name=module_name,
                path=file_path,
                docstring=ast.get_docstring(tree) or ""
            )

            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_doc.imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module_name_import = node.module or ""
                    module_doc.imports.append(module_name_import)

            # Extract top-level items
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_doc = self._analyze_class(node, module_name)
                    module_doc.classes.append(class_doc)
                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    func_doc = self._analyze_function(node, module_name)
                    module_doc.functions.append(func_doc)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            module_doc.constants.append({
                                "name": target.id,
                                "line": node.lineno
                            })

            # Identify dependencies
            module_doc.dependencies = self._identify_dependencies(module_doc.imports)

            return module_doc

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return None

    def _analyze_class(self, node: ast.ClassDef, module_name: str) -> ClassDoc:
        """Analyze a class definition."""
        class_doc = ClassDoc(
            name=node.name,
            module=module_name,
            docstring=ast.get_docstring(node) or "",
            bases=[self._get_name(base) for base in node.bases],
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            line_number=node.lineno
        )

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_doc = self._analyze_function(item, module_name)
                class_doc.methods.append(method_doc)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                class_doc.class_attributes.append({
                    "name": item.target.id,
                    "type": self._get_annotation(item.annotation)
                })

        return class_doc

    def _analyze_function(self, node, module_name: str) -> FunctionDoc:
        """Analyze a function definition."""
        is_async = isinstance(node, ast.AsyncFunctionDef)

        # Build parameter list
        params = []
        for arg in node.args.args:
            param = {"name": arg.arg}
            if arg.annotation:
                param["type"] = self._get_annotation(arg.annotation)
            params.append(param)

        # Get return type
        return_type = ""
        if node.returns:
            return_type = self._get_annotation(node.returns)

        # Build signature
        param_strs = []
        for p in params:
            if "type" in p:
                param_strs.append(f"{p['name']}: {p['type']}")
            else:
                param_strs.append(p["name"])

        prefix = "async " if is_async else ""
        signature = f"{prefix}def {node.name}({', '.join(param_strs)})"
        if return_type:
            signature += f" -> {return_type}"

        func_doc = FunctionDoc(
            name=node.name,
            module=module_name,
            signature=signature,
            docstring=ast.get_docstring(node) or "",
            parameters=params,
            return_type=return_type,
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            is_async=is_async,
            line_number=node.lineno
        )

        return func_doc

    def _get_name(self, node) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[...]"
        return str(node)

    def _get_annotation(self, node) -> str:
        """Get annotation as string."""
        if node is None:
            return ""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Subscript):
            base = self._get_annotation(node.value)
            if isinstance(node.slice, ast.Tuple):
                args = ", ".join(self._get_annotation(e) for e in node.slice.elts)
            else:
                args = self._get_annotation(node.slice)
            return f"{base}[{args}]"
        elif isinstance(node, ast.Attribute):
            return f"{self._get_annotation(node.value)}.{node.attr}"
        return ""

    def _get_decorator_name(self, node) -> str:
        """Get decorator name."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return ""

    def _identify_dependencies(self, imports: List[str]) -> List[str]:
        """Identify external dependencies."""
        stdlib = {
            "os", "sys", "json", "datetime", "logging", "typing",
            "dataclasses", "enum", "pathlib", "asyncio", "collections",
            "functools", "itertools", "re", "math", "statistics"
        }
        return [i for i in imports if i.split(".")[0] not in stdlib]


class DocumentationGenerator:
    """
    Generates documentation from analyzed code.

    Supports multiple output formats including Markdown, HTML, and JSON.
    """

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.analyzer = CodeAnalyzer()
        self.modules: Dict[str, ModuleDoc] = {}

    def scan_directory(self, directory: str, exclude_patterns: Optional[List[str]] = None):
        """
        Scan a directory for Python files and analyze them.

        Args:
            directory: Directory path to scan
            exclude_patterns: List of patterns to exclude
        """
        exclude_patterns = exclude_patterns or ["__pycache__", ".git", "venv", "test"]
        dir_path = self.project_root / directory

        for file_path in dir_path.rglob("*.py"):
            # Check exclusions
            if any(pattern in str(file_path) for pattern in exclude_patterns):
                continue

            module_doc = self.analyzer.analyze_file(str(file_path))
            if module_doc:
                rel_path = file_path.relative_to(self.project_root)
                module_key = str(rel_path).replace(os.sep, ".").replace(".py", "")
                self.modules[module_key] = module_doc

    def generate_markdown(self, output_dir: str):
        """
        Generate Markdown documentation.

        Args:
            output_dir: Output directory for generated docs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate index
        index_content = self._generate_index_md()
        (output_path / "INDEX.md").write_text(index_content, encoding="utf-8")

        # Generate module docs
        for module_key, module_doc in self.modules.items():
            module_content = self._generate_module_md(module_doc)
            module_file = output_path / f"{module_key.replace('.', '_')}.md"
            module_file.write_text(module_content, encoding="utf-8")

        logger.info(f"Generated Markdown documentation in {output_path}")

    def _generate_index_md(self) -> str:
        """Generate index Markdown."""
        lines = [
            "# API Documentation",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Modules",
            ""
        ]

        # Group by package
        packages: Dict[str, List[str]] = {}
        for module_key in sorted(self.modules.keys()):
            parts = module_key.split(".")
            package = parts[0] if len(parts) > 1 else "root"
            if package not in packages:
                packages[package] = []
            packages[package].append(module_key)

        for package, modules in sorted(packages.items()):
            lines.append(f"### {package.title()}")
            lines.append("")
            for module in modules:
                doc = self.modules[module]
                summary = doc.docstring.split("\n")[0] if doc.docstring else "No description"
                link = f"{module.replace('.', '_')}.md"
                lines.append(f"- [{module}]({link}) - {summary}")
            lines.append("")

        return "\n".join(lines)

    def _generate_module_md(self, module_doc: ModuleDoc) -> str:
        """Generate Markdown for a single module."""
        lines = [
            f"# {module_doc.name}",
            "",
            f"**Path**: `{module_doc.path}`",
            "",
        ]

        if module_doc.docstring:
            lines.extend([
                "## Overview",
                "",
                module_doc.docstring,
                ""
            ])

        # Classes
        if module_doc.classes:
            lines.append("## Classes")
            lines.append("")

            for class_doc in module_doc.classes:
                lines.append(f"### {class_doc.name}")
                lines.append("")

                if class_doc.bases:
                    lines.append(f"**Inherits from**: {', '.join(class_doc.bases)}")
                    lines.append("")

                if class_doc.docstring:
                    lines.append(class_doc.docstring)
                    lines.append("")

                # Methods
                if class_doc.methods:
                    lines.append("#### Methods")
                    lines.append("")

                    for method in class_doc.methods:
                        if method.name.startswith("_") and not method.name.startswith("__"):
                            continue  # Skip private methods

                        lines.append(f"##### `{method.signature}`")
                        lines.append("")

                        if method.docstring:
                            lines.append(method.docstring)
                            lines.append("")

        # Functions
        if module_doc.functions:
            lines.append("## Functions")
            lines.append("")

            for func_doc in module_doc.functions:
                if func_doc.name.startswith("_"):
                    continue  # Skip private functions

                lines.append(f"### `{func_doc.signature}`")
                lines.append("")

                if func_doc.docstring:
                    lines.append(func_doc.docstring)
                    lines.append("")

        # Constants
        if module_doc.constants:
            lines.append("## Constants")
            lines.append("")

            for const in module_doc.constants:
                lines.append(f"- `{const['name']}`")
            lines.append("")

        return "\n".join(lines)

    def generate_json(self, output_file: str):
        """
        Generate JSON documentation.

        Args:
            output_file: Output file path
        """
        output = {
            "generated": datetime.now().isoformat(),
            "modules": {k: v.to_dict() for k, v in self.modules.items()}
        }

        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Generated JSON documentation: {output_file}")

    def generate_summary(self) -> Dict[str, Any]:
        """Generate documentation summary."""
        total_classes = sum(len(m.classes) for m in self.modules.values())
        total_functions = sum(len(m.functions) for m in self.modules.values())
        total_methods = sum(
            sum(len(c.methods) for c in m.classes)
            for m in self.modules.values()
        )

        documented_functions = sum(
            sum(1 for f in m.functions if f.docstring)
            for m in self.modules.values()
        )
        documented_classes = sum(
            sum(1 for c in m.classes if c.docstring)
            for m in self.modules.values()
        )

        return {
            "total_modules": len(self.modules),
            "total_classes": total_classes,
            "total_functions": total_functions,
            "total_methods": total_methods,
            "documentation_coverage": {
                "functions": documented_functions / total_functions if total_functions > 0 else 0,
                "classes": documented_classes / total_classes if total_classes > 0 else 0
            }
        }


def generate_project_documentation(project_root: str,
                                   directories: List[str],
                                   output_dir: str):
    """
    Generate full project documentation.

    Args:
        project_root: Root directory of the project
        directories: List of directories to document
        output_dir: Output directory for documentation
    """
    generator = DocumentationGenerator(project_root)

    for directory in directories:
        generator.scan_directory(directory)

    generator.generate_markdown(output_dir)
    generator.generate_json(os.path.join(output_dir, "api.json"))

    summary = generator.generate_summary()
    logger.info(f"Documentation generated: {summary}")

    return summary


# =============================================================================
# P3: API DOCUMENTATION AUTO-GENERATION
# =============================================================================

class APIDocumentationGenerator:
    """
    Auto-generates API documentation from code (P3).

    Features:
    - OpenAPI/Swagger-style documentation
    - Endpoint documentation extraction
    - Request/Response schema generation
    - API versioning support
    """

    def __init__(self, project_name: str, version: str = "1.0.0"):
        """
        Initialize API documentation generator.

        Args:
            project_name: Name of the project/API
            version: API version string
        """
        self.project_name = project_name
        self.version = version
        self.endpoints: List[Dict[str, Any]] = []
        self.schemas: Dict[str, Dict[str, Any]] = {}

    def add_endpoint(
        self,
        path: str,
        method: str,
        summary: str,
        description: str = "",
        parameters: List[Dict[str, Any]] | None = None,
        request_body: Dict[str, Any] | None = None,
        responses: Dict[str, Dict[str, Any]] | None = None,
        tags: List[str] | None = None,
    ) -> None:
        """
        Add an API endpoint to documentation.

        Args:
            path: Endpoint path (e.g., "/api/v1/trades")
            method: HTTP method (GET, POST, PUT, DELETE)
            summary: Short description
            description: Detailed description
            parameters: Query/path parameters
            request_body: Request body schema
            responses: Response schemas by status code
            tags: Endpoint tags for grouping
        """
        self.endpoints.append({
            "path": path,
            "method": method.upper(),
            "summary": summary,
            "description": description,
            "parameters": parameters or [],
            "request_body": request_body,
            "responses": responses or {"200": {"description": "Successful response"}},
            "tags": tags or [],
        })

    def add_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """
        Add a reusable schema definition.

        Args:
            name: Schema name
            schema: JSON Schema definition
        """
        self.schemas[name] = schema

    def extract_from_function(self, func: Callable) -> Dict[str, Any]:
        """
        Extract API documentation from a function's signature and docstring.

        Args:
            func: Function to extract documentation from

        Returns:
            Endpoint documentation dict
        """
        sig = inspect.signature(func)
        docstring = func.__doc__ or ""
        parsed_doc = DocstringParser.parse(docstring)

        parameters = []
        for param_name, param in sig.parameters.items():
            if param_name in ["self", "cls"]:
                continue

            param_doc = next(
                (a for a in parsed_doc.get("args", []) if a.get("name") == param_name),
                {}
            )

            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                param_type = self._python_type_to_json_type(param.annotation)

            parameters.append({
                "name": param_name,
                "in": "query",
                "required": param.default == inspect.Parameter.empty,
                "schema": {"type": param_type},
                "description": param_doc.get("description", ""),
            })

        return {
            "summary": parsed_doc.get("summary", func.__name__),
            "description": parsed_doc.get("description", ""),
            "parameters": parameters,
            "responses": {
                "200": {
                    "description": parsed_doc.get("returns", "Successful response"),
                },
            },
        }

    def _python_type_to_json_type(self, python_type) -> str:
        """Convert Python type annotation to JSON Schema type."""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null",
        }

        if hasattr(python_type, "__origin__"):
            # Handle generic types like List[str], Dict[str, int]
            origin = python_type.__origin__
            if origin is list:
                return "array"
            elif origin is dict:
                return "object"

        return type_map.get(python_type, "string")

    def generate_openapi_spec(self) -> Dict[str, Any]:
        """
        Generate OpenAPI 3.0 specification.

        Returns:
            OpenAPI specification as dict
        """
        # Group endpoints by path
        paths: Dict[str, Dict[str, Any]] = {}
        for endpoint in self.endpoints:
            path = endpoint["path"]
            method = endpoint["method"].lower()

            if path not in paths:
                paths[path] = {}

            paths[path][method] = {
                "summary": endpoint["summary"],
                "description": endpoint["description"],
                "parameters": endpoint["parameters"],
                "responses": endpoint["responses"],
                "tags": endpoint["tags"],
            }

            if endpoint["request_body"]:
                paths[path][method]["requestBody"] = endpoint["request_body"]

        # Collect all tags
        all_tags = set()
        for endpoint in self.endpoints:
            all_tags.update(endpoint["tags"])

        return {
            "openapi": "3.0.0",
            "info": {
                "title": self.project_name,
                "version": self.version,
                "description": f"API documentation for {self.project_name}",
            },
            "paths": paths,
            "components": {
                "schemas": self.schemas,
            },
            "tags": [{"name": tag} for tag in sorted(all_tags)],
        }

    def generate_markdown_docs(self) -> str:
        """
        Generate Markdown API documentation.

        Returns:
            Markdown string
        """
        lines = [
            f"# {self.project_name} API Documentation",
            "",
            f"**Version**: {self.version}",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Table of Contents",
            "",
        ]

        # Group by tags
        endpoints_by_tag: Dict[str, List[Dict]] = {}
        for endpoint in self.endpoints:
            for tag in endpoint["tags"] or ["General"]:
                if tag not in endpoints_by_tag:
                    endpoints_by_tag[tag] = []
                endpoints_by_tag[tag].append(endpoint)

        # Generate TOC
        for tag in sorted(endpoints_by_tag.keys()):
            lines.append(f"- [{tag}](#{tag.lower().replace(' ', '-')})")
        lines.append("")

        # Generate endpoint documentation
        for tag in sorted(endpoints_by_tag.keys()):
            lines.extend([
                f"## {tag}",
                "",
            ])

            for endpoint in endpoints_by_tag[tag]:
                method = endpoint["method"]
                path = endpoint["path"]
                lines.extend([
                    f"### {method} {path}",
                    "",
                    endpoint["summary"],
                    "",
                ])

                if endpoint["description"]:
                    lines.extend([endpoint["description"], ""])

                # Parameters
                if endpoint["parameters"]:
                    lines.extend(["**Parameters:**", ""])
                    lines.append("| Name | Type | Required | Description |")
                    lines.append("|------|------|----------|-------------|")
                    for param in endpoint["parameters"]:
                        required = "Yes" if param.get("required") else "No"
                        param_type = param.get("schema", {}).get("type", "string")
                        lines.append(
                            f"| `{param['name']}` | {param_type} | {required} | "
                            f"{param.get('description', '')} |"
                        )
                    lines.append("")

                # Request body
                if endpoint["request_body"]:
                    lines.extend([
                        "**Request Body:**",
                        "",
                        "```json",
                        json.dumps(endpoint["request_body"], indent=2),
                        "```",
                        "",
                    ])

                # Responses
                lines.extend(["**Responses:**", ""])
                for status_code, response in endpoint["responses"].items():
                    lines.append(f"- **{status_code}**: {response.get('description', '')}")
                lines.append("")

        # Schemas section
        if self.schemas:
            lines.extend([
                "## Schemas",
                "",
            ])
            for schema_name, schema in self.schemas.items():
                lines.extend([
                    f"### {schema_name}",
                    "",
                    "```json",
                    json.dumps(schema, indent=2),
                    "```",
                    "",
                ])

        return "\n".join(lines)

    def save_documentation(self, output_dir: str) -> Dict[str, str]:
        """
        Save all documentation formats.

        Args:
            output_dir: Output directory

        Returns:
            Dict of generated file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated_files = {}

        # Save OpenAPI spec
        openapi_path = output_path / "openapi.json"
        with open(openapi_path, "w") as f:
            json.dump(self.generate_openapi_spec(), f, indent=2)
        generated_files["openapi"] = str(openapi_path)

        # Save Markdown docs
        md_path = output_path / "API.md"
        md_path.write_text(self.generate_markdown_docs(), encoding="utf-8")
        generated_files["markdown"] = str(md_path)

        logger.info(f"API documentation saved to {output_dir}")
        return generated_files


# =============================================================================
# P3: CONFIGURATION SCHEMA DOCUMENTATION
# =============================================================================

class ConfigSchemaDocumenter:
    """
    Documents configuration schemas (P3).

    Features:
    - YAML/JSON schema documentation
    - Default values documentation
    - Environment variable mapping
    - Validation rules documentation
    """

    def __init__(self, schema_name: str = "Configuration"):
        """
        Initialize configuration schema documenter.

        Args:
            schema_name: Name of the configuration schema
        """
        self.schema_name = schema_name
        self.sections: Dict[str, Dict[str, Any]] = {}
        self.env_mappings: Dict[str, str] = {}

    def add_section(
        self,
        name: str,
        description: str,
        fields: List[Dict[str, Any]],
    ) -> None:
        """
        Add a configuration section.

        Args:
            name: Section name (e.g., "database", "logging")
            description: Section description
            fields: List of field definitions
        """
        self.sections[name] = {
            "description": description,
            "fields": fields,
        }

    def add_field(
        self,
        section: str,
        name: str,
        field_type: str,
        description: str,
        default: Any = None,
        required: bool = False,
        env_var: str | None = None,
        validation: str | None = None,
        example: Any = None,
    ) -> None:
        """
        Add a field to a section.

        Args:
            section: Section name
            name: Field name
            field_type: Field type (string, integer, boolean, array, object)
            description: Field description
            default: Default value
            required: Whether the field is required
            env_var: Environment variable that can override this field
            validation: Validation rules description
            example: Example value
        """
        if section not in self.sections:
            self.sections[section] = {"description": "", "fields": []}

        field_def = {
            "name": name,
            "type": field_type,
            "description": description,
            "default": default,
            "required": required,
            "validation": validation,
            "example": example,
        }

        self.sections[section]["fields"].append(field_def)

        if env_var:
            self.env_mappings[f"{section}.{name}"] = env_var

    def generate_markdown(self) -> str:
        """
        Generate Markdown documentation for the configuration schema.

        Returns:
            Markdown string
        """
        lines = [
            f"# {self.schema_name} Schema Documentation",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Table of Contents",
            "",
        ]

        # TOC
        for section_name in sorted(self.sections.keys()):
            anchor = section_name.lower().replace(" ", "-").replace(".", "-")
            lines.append(f"- [{section_name}](#{anchor})")
        lines.append("")

        # Sections
        for section_name, section_data in sorted(self.sections.items()):
            lines.extend([
                f"## {section_name}",
                "",
                section_data["description"],
                "",
            ])

            if section_data["fields"]:
                lines.append("| Field | Type | Required | Default | Description |")
                lines.append("|-------|------|----------|---------|-------------|")

                for field in section_data["fields"]:
                    required = "Yes" if field["required"] else "No"
                    default = f"`{field['default']}`" if field["default"] is not None else "-"
                    lines.append(
                        f"| `{field['name']}` | {field['type']} | {required} | "
                        f"{default} | {field['description']} |"
                    )

                lines.append("")

                # Detailed field documentation
                for field in section_data["fields"]:
                    lines.extend([
                        f"### {section_name}.{field['name']}",
                        "",
                        field["description"],
                        "",
                    ])

                    if field["validation"]:
                        lines.extend([
                            f"**Validation**: {field['validation']}",
                            "",
                        ])

                    if field["example"] is not None:
                        lines.extend([
                            "**Example**:",
                            "",
                            "```yaml",
                            f"{field['name']}: {field['example']}",
                            "```",
                            "",
                        ])

                    env_key = f"{section_name}.{field['name']}"
                    if env_key in self.env_mappings:
                        lines.extend([
                            f"**Environment Variable**: `{self.env_mappings[env_key]}`",
                            "",
                        ])

        # Environment variables summary
        if self.env_mappings:
            lines.extend([
                "## Environment Variables",
                "",
                "The following environment variables can be used to override configuration values:",
                "",
                "| Config Path | Environment Variable |",
                "|-------------|---------------------|",
            ])

            for config_path, env_var in sorted(self.env_mappings.items()):
                lines.append(f"| `{config_path}` | `{env_var}` |")

            lines.append("")

        return "\n".join(lines)

    def generate_json_schema(self) -> Dict[str, Any]:
        """
        Generate JSON Schema for the configuration.

        Returns:
            JSON Schema dict
        """
        properties = {}
        required_fields = []

        for section_name, section_data in self.sections.items():
            section_properties = {}
            section_required = []

            for field in section_data["fields"]:
                field_schema = {
                    "type": field["type"],
                    "description": field["description"],
                }

                if field["default"] is not None:
                    field_schema["default"] = field["default"]

                if field["example"] is not None:
                    field_schema["examples"] = [field["example"]]

                section_properties[field["name"]] = field_schema

                if field["required"]:
                    section_required.append(field["name"])

            properties[section_name] = {
                "type": "object",
                "description": section_data["description"],
                "properties": section_properties,
            }

            if section_required:
                properties[section_name]["required"] = section_required

        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": self.schema_name,
            "type": "object",
            "properties": properties,
        }

    def generate_example_config(self, format: str = "yaml") -> str:
        """
        Generate example configuration file.

        Args:
            format: Output format ("yaml" or "json")

        Returns:
            Example configuration string
        """
        config = {}

        for section_name, section_data in self.sections.items():
            section_config = {}

            for field in section_data["fields"]:
                value = field["example"] if field["example"] is not None else field["default"]
                if value is not None:
                    section_config[field["name"]] = value

            if section_config:
                config[section_name] = section_config

        if format == "yaml":
            try:
                import yaml
                return yaml.dump(config, default_flow_style=False, sort_keys=False)
            except ImportError:
                # Fallback to simple YAML-like format
                lines = []
                for section, fields in config.items():
                    lines.append(f"{section}:")
                    for key, value in fields.items():
                        if isinstance(value, str):
                            lines.append(f"  {key}: \"{value}\"")
                        else:
                            lines.append(f"  {key}: {value}")
                return "\n".join(lines)
        else:
            return json.dumps(config, indent=2)

    def save_documentation(self, output_dir: str) -> Dict[str, str]:
        """
        Save configuration documentation.

        Args:
            output_dir: Output directory

        Returns:
            Dict of generated file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated_files = {}

        # Save Markdown
        md_path = output_path / "CONFIG.md"
        md_path.write_text(self.generate_markdown(), encoding="utf-8")
        generated_files["markdown"] = str(md_path)

        # Save JSON Schema
        schema_path = output_path / "config-schema.json"
        with open(schema_path, "w") as f:
            json.dump(self.generate_json_schema(), f, indent=2)
        generated_files["json_schema"] = str(schema_path)

        # Save example config
        example_path = output_path / "config.example.yaml"
        example_path.write_text(self.generate_example_config("yaml"), encoding="utf-8")
        generated_files["example"] = str(example_path)

        logger.info(f"Configuration documentation saved to {output_dir}")
        return generated_files


# =============================================================================
# P3: EXAMPLE CODE SNIPPETS
# =============================================================================

class CodeSnippetGenerator:
    """
    Generates and manages code example snippets (P3).

    Features:
    - Extract examples from docstrings
    - Generate usage examples
    - Create runnable code snippets
    - Language-aware syntax highlighting hints
    """

    def __init__(self):
        """Initialize code snippet generator."""
        self.snippets: Dict[str, List[Dict[str, Any]]] = {}

    def add_snippet(
        self,
        category: str,
        title: str,
        code: str,
        language: str = "python",
        description: str = "",
        tags: List[str] | None = None,
        output: str | None = None,
    ) -> None:
        """
        Add a code snippet.

        Args:
            category: Snippet category (e.g., "getting-started", "advanced")
            title: Snippet title
            code: Code content
            language: Programming language
            description: Description of what the code does
            tags: Tags for searching
            output: Expected output (optional)
        """
        if category not in self.snippets:
            self.snippets[category] = []

        self.snippets[category].append({
            "title": title,
            "code": code,
            "language": language,
            "description": description,
            "tags": tags or [],
            "output": output,
        })

    def extract_from_docstring(self, docstring: str) -> List[Dict[str, Any]]:
        """
        Extract code examples from a docstring.

        Args:
            docstring: The docstring to parse

        Returns:
            List of extracted code snippets
        """
        snippets = []

        if not docstring:
            return snippets

        lines = docstring.split("\n")
        in_example = False
        current_code = []
        current_description = ""

        for line in lines:
            stripped = line.strip()

            if stripped.lower() in ["example:", "examples:", "usage:"]:
                in_example = True
                current_code = []
                continue

            if in_example:
                if stripped.startswith(">>>"):
                    # Python interactive example
                    current_code.append(stripped[4:] if stripped.startswith(">>> ") else stripped[3:])
                elif stripped.startswith("..."):
                    # Continuation
                    current_code.append(stripped[4:] if stripped.startswith("... ") else stripped[3:])
                elif line.startswith("    ") or line.startswith("\t"):
                    # Indented code block
                    current_code.append(line[4:] if line.startswith("    ") else line[1:])
                elif stripped == "" and current_code:
                    # End of example block
                    if current_code:
                        snippets.append({
                            "code": "\n".join(current_code),
                            "language": "python",
                            "description": current_description,
                        })
                    current_code = []
                    current_description = ""
                    in_example = False
                elif not stripped.startswith(("Args:", "Returns:", "Raises:", "Note:", "Warning:")):
                    current_description = stripped

        # Handle trailing example
        if current_code:
            snippets.append({
                "code": "\n".join(current_code),
                "language": "python",
                "description": current_description,
            })

        return snippets

    def generate_usage_example(
        self,
        class_or_func: Any,
        include_imports: bool = True,
    ) -> str:
        """
        Generate a usage example for a class or function.

        Args:
            class_or_func: The class or function to generate example for
            include_imports: Whether to include import statements

        Returns:
            Example code string
        """
        lines = []

        if include_imports:
            module = class_or_func.__module__
            name = class_or_func.__name__
            lines.append(f"from {module} import {name}")
            lines.append("")

        if inspect.isclass(class_or_func):
            # Class example
            class_name = class_or_func.__name__
            init_sig = inspect.signature(class_or_func.__init__)
            params = []

            for param_name, param in init_sig.parameters.items():
                if param_name == "self":
                    continue
                if param.default != inspect.Parameter.empty:
                    continue  # Skip optional params in example
                params.append(f"{param_name}=...")

            lines.append(f"# Create instance")
            lines.append(f"instance = {class_name}({', '.join(params)})")
            lines.append("")

            # Add method examples
            methods = [
                m for m in dir(class_or_func)
                if not m.startswith("_") and callable(getattr(class_or_func, m))
            ]

            for method_name in methods[:3]:  # Limit to first 3 methods
                method = getattr(class_or_func, method_name)
                if hasattr(method, "__func__"):
                    lines.append(f"# Call {method_name}")
                    lines.append(f"result = instance.{method_name}(...)")
                    lines.append("")

        elif inspect.isfunction(class_or_func):
            # Function example
            func_name = class_or_func.__name__
            sig = inspect.signature(class_or_func)
            params = []

            for param_name, param in sig.parameters.items():
                if param.default != inspect.Parameter.empty:
                    continue
                params.append(f"{param_name}=...")

            lines.append(f"# Call function")
            lines.append(f"result = {func_name}({', '.join(params)})")

        return "\n".join(lines)

    def generate_markdown(self, category: str | None = None) -> str:
        """
        Generate Markdown documentation for snippets.

        Args:
            category: Specific category to generate (None for all)

        Returns:
            Markdown string
        """
        lines = [
            "# Code Examples",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        categories = [category] if category else sorted(self.snippets.keys())

        for cat in categories:
            if cat not in self.snippets:
                continue

            # Convert category to title case
            title = cat.replace("-", " ").replace("_", " ").title()
            lines.extend([
                f"## {title}",
                "",
            ])

            for snippet in self.snippets[cat]:
                lines.extend([
                    f"### {snippet['title']}",
                    "",
                ])

                if snippet["description"]:
                    lines.extend([snippet["description"], ""])

                lines.extend([
                    f"```{snippet['language']}",
                    snippet["code"],
                    "```",
                    "",
                ])

                if snippet["output"]:
                    lines.extend([
                        "**Output:**",
                        "",
                        "```",
                        snippet["output"],
                        "```",
                        "",
                    ])

                if snippet["tags"]:
                    lines.extend([
                        f"*Tags: {', '.join(snippet['tags'])}*",
                        "",
                    ])

        return "\n".join(lines)

    def search_snippets(
        self,
        query: str,
        tags: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Search snippets by query and/or tags.

        Args:
            query: Search query (searches title, description, code)
            tags: Filter by tags

        Returns:
            List of matching snippets
        """
        results = []
        query_lower = query.lower()

        for category, snippets in self.snippets.items():
            for snippet in snippets:
                # Check tags
                if tags and not any(t in snippet["tags"] for t in tags):
                    continue

                # Check query
                if query:
                    searchable = (
                        snippet["title"].lower() +
                        snippet["description"].lower() +
                        snippet["code"].lower()
                    )
                    if query_lower not in searchable:
                        continue

                results.append({
                    "category": category,
                    **snippet,
                })

        return results

    def save_snippets(self, output_dir: str) -> Dict[str, str]:
        """
        Save snippets to files.

        Args:
            output_dir: Output directory

        Returns:
            Dict of generated file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated_files = {}

        # Save full Markdown
        md_path = output_path / "EXAMPLES.md"
        md_path.write_text(self.generate_markdown(), encoding="utf-8")
        generated_files["markdown"] = str(md_path)

        # Save individual category files
        for category in self.snippets:
            cat_path = output_path / f"examples_{category}.md"
            cat_path.write_text(self.generate_markdown(category), encoding="utf-8")
            generated_files[f"category_{category}"] = str(cat_path)

        # Save JSON index
        index_path = output_path / "examples.json"
        with open(index_path, "w") as f:
            json.dump(self.snippets, f, indent=2)
        generated_files["json_index"] = str(index_path)

        logger.info(f"Code snippets saved to {output_dir}")
        return generated_files


# Convenience function to create pre-populated snippet generator
def create_trading_snippets() -> CodeSnippetGenerator:
    """
    Create a code snippet generator with trading system examples.

    Returns:
        Pre-populated CodeSnippetGenerator
    """
    generator = CodeSnippetGenerator()

    # Getting Started examples
    generator.add_snippet(
        category="getting-started",
        title="Initialize Trading System",
        code='''from main import TradingSystem

# Create trading system with default config
system = TradingSystem(config_path="config.yaml")

# Initialize all components
await system.initialize()

# Start trading
await system.start()''',
        description="Basic setup for the trading system.",
        tags=["setup", "initialization"],
    )

    generator.add_snippet(
        category="getting-started",
        title="Connect to Interactive Brokers",
        code='''from core.event_bus import EventBus
from agents.execution_agent import ExecutionAgent

# Setup event bus
event_bus = EventBus()

# Create execution agent with IB connection
execution_agent = ExecutionAgent(
    event_bus=event_bus,
    config={
        "broker": "interactive_brokers",
        "host": "127.0.0.1",
        "port": 7497,  # Paper trading port
        "client_id": 1
    }
)

# Connect
await execution_agent.connect()''',
        description="Connect to Interactive Brokers for execution.",
        tags=["ib", "broker", "connection"],
    )

    # Strategy examples
    generator.add_snippet(
        category="strategies",
        title="Create Custom Signal",
        code='''from strategies.base_strategy import BaseStrategy, Signal

class MyStrategy(BaseStrategy):
    def generate_signal(self, market_data):
        # Analyze market data
        price = market_data["close"]
        sma_20 = market_data["sma_20"]

        # Generate signal
        if price > sma_20 * 1.02:
            return Signal(
                symbol=market_data["symbol"],
                direction="long",
                strength=0.8,
                source="my_strategy"
            )
        return None''',
        description="Create a custom trading signal strategy.",
        tags=["strategy", "signals", "custom"],
    )

    # Risk management examples
    generator.add_snippet(
        category="risk-management",
        title="Calculate Portfolio VaR",
        code='''from core.var_calculator import VaRCalculator

calculator = VaRCalculator(
    confidence_level=0.99,
    time_horizon_days=1,
    method="historical"
)

# Calculate VaR for portfolio
positions = {
    "AAPL": 10000,
    "GOOGL": 15000,
    "MSFT": 12000
}

var_result = calculator.calculate_var(
    positions=positions,
    returns_data=historical_returns
)

print(f"1-day 99% VaR: ${var_result['var']:,.2f}")''',
        description="Calculate Value at Risk for a portfolio.",
        tags=["risk", "var", "portfolio"],
    )

    # Attribution examples
    generator.add_snippet(
        category="attribution",
        title="Track Trade Performance",
        code='''from core.attribution import PerformanceAttribution

attribution = PerformanceAttribution()

# Record trade entry
trade_id = attribution.record_trade_entry(
    strategy="momentum",
    symbol="AAPL",
    side="buy",
    quantity=100,
    entry_price=150.00,
    commission=1.00
)

# Later, record exit
trade = attribution.record_trade_exit(
    trade_id=trade_id,
    exit_price=155.00,
    commission=1.00
)

# Get metrics
metrics = attribution.get_strategy_metrics("momentum")
print(f"Win rate: {metrics.win_rate:.1%}")
print(f"Sharpe ratio: {metrics.sharpe_ratio:.2f}")''',
        description="Track and attribute trade performance.",
        tags=["attribution", "performance", "tracking"],
    )

    return generator
