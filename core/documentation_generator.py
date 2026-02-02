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
        (output_path / "INDEX.md").write_text(index_content)

        # Generate module docs
        for module_key, module_doc in self.modules.items():
            module_content = self._generate_module_md(module_doc)
            module_file = output_path / f"{module_key.replace('.', '_')}.md"
            module_file.write_text(module_content)

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
