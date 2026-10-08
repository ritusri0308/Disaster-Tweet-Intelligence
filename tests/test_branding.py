"""
Branding verification tests for CrisisLens.
Ensures zero traces of academic or course terminology remain in codebase.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_TERMS = [
    r"csi3007",
    r"advanced python programming",
    r"\bcourse\b",
    r"\buniversity\b",
    r"\bstudent\b",
    r"academic prototype",
    r"python project",
    r"course project"
]

def test_branding_cleanliness():
    """Scan all Python, Markdown, and configuration files for forbidden legacy strings."""
    forbidden_pattern = re.compile("|".join(FORBIDDEN_TERMS), re.IGNORECASE)
    violations = []

    # Files to check: all .py, .md, .toml, .yaml, .yml
    scan_extensions = {".py", ".md", ".toml", ".yaml", ".yml"}

    # Exclude directories
    exclude_dirs = {".git", ".pytest_cache", "__pycache__", "data", "venv", ".venv"}

    for path in PROJECT_ROOT.rglob("*"):
        if path.is_file() and path.suffix in scan_extensions:
            # Check if any parent is in exclude_dirs
            if any(part in exclude_dirs for part in path.parts):
                continue

            # Skip Licensing.txt or Readme.txt under data
            if "Licensing" in path.name or "Readme.txt" in path.name:
                continue

            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue

            for line_idx, line in enumerate(content.splitlines(), 1):
                # If this test file itself contains the pattern strings, skip the definition line
                if path.name == "test_branding.py":
                    continue
                match = forbidden_pattern.search(line)
                if match:
                    violations.append(f"{path.relative_to(PROJECT_ROOT)}:{line_idx} - Found '{match.group(0)}': {line.strip()[:80]}")

    assert not violations, f"Branding violations found ({len(violations)}):\n" + "\n".join(violations)
