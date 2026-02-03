#!/usr/bin/env python3
"""
Documentation Generator Script
==============================

Generates API documentation from source code using the core documentation_generator module.

Usage:
    python scripts/generate_docs.py [--output-dir docs/api] [--format markdown]

This script scans the following directories:
    - agents/     : Trading agent implementations
    - core/       : Core infrastructure modules
    - strategies/ : Trading strategy implementations
    - data/       : Market data handling

Output is written to docs/api/ by default.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.documentation_generator import (
    DocumentationGenerator,
    generate_project_documentation,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    """Main entry point for documentation generation."""
    parser = argparse.ArgumentParser(
        description="Generate API documentation from source code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate docs with defaults (output to docs/api/)
    python scripts/generate_docs.py

    # Generate to custom directory
    python scripts/generate_docs.py --output-dir ./my-docs

    # Verbose output
    python scripts/generate_docs.py -v
        """,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="docs/api",
        help="Output directory for generated documentation (default: docs/api)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--scan-dirs",
        "-s",
        nargs="+",
        default=["agents", "core", "strategies", "data"],
        help="Directories to scan (default: agents core strategies data)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Resolve paths
    project_root = PROJECT_ROOT
    output_dir = project_root / args.output_dir

    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Scanning directories: {args.scan_dirs}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = DocumentationGenerator(str(project_root))

    # Scan directories
    for directory in args.scan_dirs:
        dir_path = project_root / directory
        if dir_path.exists():
            logger.info(f"Scanning: {directory}/")
            generator.scan_directory(directory, exclude_patterns=["__pycache__", ".git", "test_"])
        else:
            logger.warning(f"Directory not found: {directory}")

    # Generate documentation
    if args.format in ("markdown", "both"):
        logger.info("Generating Markdown documentation...")
        generator.generate_markdown(str(output_dir))
        logger.info(f"Markdown docs written to: {output_dir}")

    if args.format in ("json", "both"):
        json_file = output_dir / "api.json"
        logger.info("Generating JSON documentation...")
        generator.generate_json(str(json_file))
        logger.info(f"JSON docs written to: {json_file}")

    # Print summary
    summary = generator.generate_summary()
    logger.info("")
    logger.info("=" * 50)
    logger.info("DOCUMENTATION GENERATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"  Modules documented:    {summary['total_modules']}")
    logger.info(f"  Classes documented:    {summary['total_classes']}")
    logger.info(f"  Functions documented:  {summary['total_functions']}")
    logger.info(f"  Total methods:         {summary['total_methods']}")
    logger.info("")
    logger.info("Coverage:")
    coverage = summary.get("documentation_coverage", {})
    logger.info(f"  Functions with docs:   {coverage.get('functions', 0):.1%}")
    logger.info(f"  Classes with docs:     {coverage.get('classes', 0):.1%}")
    logger.info("=" * 50)
    logger.info(f"Output location: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
